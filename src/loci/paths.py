"""Where loci keeps its own artifacts.

Index artifacts live in ONE place, not per-project, because routing is a
cross-scope decision: answering "which project is this question about" requires
every scope's vocabulary in the same lookup. Per-project indexes would make the
cheap question (one dict lookup) into the expensive one (open N indexes).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "LOCI_HOME"
LOCK_STALE_SECONDS = 3600
REPLACE_RETRY_SECONDS = 5.0


def home() -> Path:
    """Data root. ``$LOCI_HOME`` wins; otherwise ``~/.loci``."""
    raw = os.environ.get(ENV_HOME)
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            raise ValueError(f"{ENV_HOME} must be an absolute path, got {raw!r}")
        return p
    return Path.home() / ".loci"


def ensure_home() -> Path:
    p = home()
    p.mkdir(parents=True, exist_ok=True)
    return p


def registry_file() -> Path:
    """The scope registry.

    JSON rather than TOML on purpose: ``tomllib`` is read-only and 3.11+, and
    hand-rolling a TOML *writer* is a known way to produce files the parser then
    rejects. One machine-managed file, human-editable, no encoder to get wrong.
    """
    return home() / "scopes.json"


def groups_file() -> Path:
    """Group policy: which mode each group runs in.

    Separate from the registry because `scopes.json` is machine-managed and
    rewritten by every scan, while this is user-authored and must survive one.
    """
    return home() / "groups.json"


def scope_index_file() -> Path:
    return home() / "scope_index.json"


def episode_store_file() -> Path:
    return home() / "episodes.json"


def embeddings_file() -> Path:
    return home() / "embeddings.npz"


def rankers_dir() -> Path:
    """Fitted lexical rankers, one file per scope.

    Per-scope rather than one bundle so a query loads only what it asks for:
    routing usually selects one or two scopes out of many, and deserializing
    every scope's matrix to answer about one is the same waste as loading every
    graph to decide which project a question is about.
    """
    return home() / "rankers"


# ---------------------------------------------------------------------------
# durable writes
# ---------------------------------------------------------------------------
def _replace(tmp: "str | Path", path: Path) -> None:
    """`os.replace`, retried while Windows holds the destination open.

    `os.replace` is atomic on POSIX and on Windows, but only POSIX lets it
    proceed while another handle has the destination open. Windows raises
    PermissionError instead, which converts the failure this module exists to
    prevent -- a reader seeing half a file -- into a different one: a write that
    never lands. A long-lived MCP server reading the store while `loci index`
    runs holds exactly that handle, so the durable path was the one that broke.

    Retrying is the whole mitigation, and it works because the competing handle
    is a reader that opens, reads and closes: the window is milliseconds, not a
    held lock. The retry is bounded so a genuinely locked file fails loudly
    rather than hanging -- an unbounded wait here cost four CI jobs six hours
    each before it was diagnosed.
    """
    import time

    deadline = time.monotonic() + REPLACE_RETRY_SECONDS
    delay = 0.001
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # POSIX never raises this merely because the destination is open,
            # so anywhere but Windows it is a real permissions error and
            # retrying would only delay the report.
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.05)


def atomic_write(path: Path, data: "str | bytes") -> None:
    """Write via a sibling temp file and rename.

    `Path.write_text` truncates and then fills, so a reader can observe a
    half-written file. Measured on an 8MB store: 11 torn reads in a few seconds
    of concurrent access. A long-lived MCP server reading while `loci index`
    runs hits exactly that window, and a JSONDecodeError there is indistinguishable
    from a corrupt index.

    The temp file is created beside the target so the rename never crosses a
    filesystem boundary, and the rename goes through `_replace`, which absorbs
    the one way Windows differs here.
    """
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    kwargs = {} if isinstance(data, bytes) else {"encoding": "utf-8"}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode, **kwargs) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_via(path: Path, writer) -> None:
    """Same guarantee for libraries that insist on writing to a path themselves."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=str(path.parent), prefix=f".{path.name}.")) / path.name
    try:
        writer(tmp)
        _replace(tmp, path)
    finally:
        try:
            tmp.parent.rmdir()
        except OSError:
            pass


class BuildLock:
    """Advisory lock so two `loci index` runs cannot interleave their writes.

    Atomic writes stop a reader seeing half a file; they do not stop two writers
    producing an index and a store that disagree with each other, because those
    are separate files written at separate moments.
    """

    def __init__(self, name: str = "index") -> None:
        self.path = home() / f".{name}.lock"

    def __enter__(self) -> "BuildLock":
        import os
        import time

        ensure_home()
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {int(time.time())}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                if self._stale():
                    try:
                        self.path.unlink()
                        continue
                    except OSError:
                        pass
                raise SystemExit(
                    f"error: another loci index is running (lock: {self.path}). "
                    f"If that is wrong, delete the lock file and retry.")
        return self

    def _stale(self) -> bool:
        import os
        import time

        try:
            pid_s, ts_s = self.path.read_text().split()
            pid, ts = int(pid_s), int(ts_s)
        except (OSError, ValueError):
            return True
        if time.time() - ts > LOCK_STALE_SECONDS:
            return True
        # signal 0 is a POSIX liveness probe; on Windows os.kill actually
        # terminates, so never send one there -- fall back to the age check
        # already applied above.
        if os.name != "posix":
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def __exit__(self, *exc) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass
