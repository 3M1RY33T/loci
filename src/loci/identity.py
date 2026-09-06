"""What a project IS, from the outside — its signboard.

A scope's aliases say what to call it in a question. They are not what another
project calls it in a dependency, an import, or an argv. loci is the case that
forced this module: it is `loci-mem` on PyPI, `loci` to import, `loci` to run
and `3M1RY33T/loci` on GitHub. Four names, one project, and the only one an
edge ever mentions is the third.

Read at the scope root only. A monorepo's inner manifests belong to its
sub-scopes, and descending is what made fingerprinting one repository take
32.9s before `SKIP_DIRS` pruning existed (`defaults.py`).

Nothing here feeds the routing index. A signboard is matched against edges by
`edges.py`, never tokenized into a scope's vocabulary: putting `loci-mem` in
the loci scope's postings is how a question that merely names a tool routes to
the tool's own source, which is the failure `ALIAS_BOOST` already has to be
careful about.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:                                  # 3.11+
    import tomllib as _tomllib
except ModuleNotFoundError:           # 3.10; see `_toml_keys`
    _tomllib = None                   # type: ignore[assignment]

from .defaults import SKIP_DIRS

# Directories that are never the importable package, even when they carry an
# `__init__.py`. A test package is importable and is not what another project
# imports.
_NOT_A_PACKAGE = frozenset({"tests", "test", "docs", "doc", "evals", "eval",
                            "examples", "example", "benchmarks", "scripts"})

EMPTY: dict[str, list[str]] = {"dist": [], "imports": [], "command": []}


def _add(out: dict[str, list[str]], field: str, value: str | None) -> None:
    """Append, deduplicated and order-preserving. Order is the manifest's."""
    v = (value or "").strip()
    if v and v not in out[field]:
        out[field].append(v)


# -- TOML ------------------------------------------------------------------
_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_KEY = re.compile(r"""^\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z0-9_.\-]+))\s*=""")
_STR = re.compile(r"""^\s*[^=]+=\s*(?:"([^"]*)"|'([^']*)')""")


def _toml_keys(text: str, section: str) -> tuple[str | None, list[str]]:
    """`(name, key_names)` for one TOML table, without a TOML parser.

    3.10 has no `tomllib` and loci supports 3.10. Three keys out of two tables
    is not worth a `tomli` dependency, but a hand-rolled reader that ignores
    section boundaries is worse than none: `[tool.other] name` is a different
    project's name and reading it as this one's puts a wrong identity on the
    signboard, which resolves edges to the wrong scope. So this tracks the
    current table and answers only for the one asked about.
    """
    name: str | None = None
    keys: list[str] = []
    cur = ""
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _SECTION.match(line)
        if m:
            cur = m.group(1).strip()
            continue
        if cur != section:
            continue
        km = _KEY.match(line)
        if not km:
            continue
        key = km.group(1) or km.group(2) or km.group(3)
        keys.append(key)
        if key == "name" and name is None:
            sm = _STR.match(line)
            if sm:
                name = sm.group(1) if sm.group(1) is not None else sm.group(2)
    return name, keys


def _toml_table(text: str, path: tuple[str, ...]) -> dict:
    if _tomllib is None:
        return {}
    try:
        data = _tomllib.loads(text)
    except Exception:
        return {}
    for part in path:
        data = data.get(part) if isinstance(data, dict) else None
        if not isinstance(data, dict):
            return {}
    return data


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# -- per-manifest readers --------------------------------------------------
def _normalize(name: str) -> str:
    """PEP 503-ish: the form in which `tensor-serve` and `tensor_serve` are
    the same name."""
    return re.sub(r"[-_.]+", "_", name).lower()


def _python_packages(root: Path, dist: str | None) -> list[str]:
    """Importable top-level packages, `src/` layout first.

    A filesystem question, not a manifest one: setuptools, hatchling, poetry
    and flit each declare packages differently or not at all, and the directory
    holding `__init__.py` is the same answer under all four.

    A root layout is filtered against the distribution name, and the reason is
    a measurement rather than taste. Scanning every root directory with an
    `__init__.py` across the real corpus put `api`, `cli`, `search` and
    `agent_tools` on two signboards. Those are internal packages of
    applications nobody installs, and an edge naming `search` would then
    resolve to a project that merely has a directory by that name -- a
    fabricated cross-project dependency, which is worse than the missing edge
    it replaces.

    A declared distribution is the precondition for having any import identity
    at all, in either layout. Trusting `src/` on its own was measured wrong on
    the same corpus: `odysseus` keeps application code in `src/` under a
    pyproject declaring only `[tool.pytest.ini_options]`, so `src/` there means
    "where the code lives" and not "what is published". Only a distribution
    name says the code is published under a name -- and once one exists, a
    `src/` layout is trusted whole, because a distribution may ship several
    packages and each directory under `src/` exists to be one.
    """
    if not dist:
        return []                    # not distributed: no import identity
    found = [d.name for d in sorted((root / "src").glob("*"))
             if (d / "__init__.py").exists()]
    if found:
        return found
    want = _normalize(dist)
    return [d.name for d in sorted(root.glob("*"))
            if d.is_dir() and d.name not in SKIP_DIRS
            and d.name not in _NOT_A_PACKAGE
            and _normalize(d.name) == want
            and (d / "__init__.py").exists()]


def _from_pyproject(root: Path, out: dict[str, list[str]]) -> None:
    text = _read(root / "pyproject.toml")
    if not text:
        return
    project = _toml_table(text, ("project",))
    if project:
        _add(out, "dist", project.get("name"))
        for script in (project.get("scripts") or {}):
            _add(out, "command", script)
    else:
        name, _ = _toml_keys(text, "project")
        _add(out, "dist", name)
        _, scripts = _toml_keys(text, "project.scripts")
        for script in scripts:
            _add(out, "command", script)
    for pkg in _python_packages(root, out["dist"][0] if out["dist"] else None):
        _add(out, "imports", pkg)


def _from_package_json(root: Path, out: dict[str, list[str]]) -> None:
    text = _read(root / "package.json")
    if not text:
        return
    try:
        data = json.loads(text)
    except Exception:
        return                       # one bad manifest must not end a scan
    if not isinstance(data, dict):
        return
    name = data.get("name") if isinstance(data.get("name"), str) else None
    _add(out, "dist", name)
    _add(out, "imports", name)       # the npm specifier IS the import
    bin_ = data.get("bin")
    if isinstance(bin_, dict):
        for cmd in bin_:
            _add(out, "command", cmd)
    elif isinstance(bin_, str) and name:
        # `"bin": "cli.js"` means one command, named after the package -- and
        # for a scoped package it is the tail, not `@acme/thing`.
        _add(out, "command", name.rsplit("/", 1)[-1])


def _from_cargo(root: Path, out: dict[str, list[str]]) -> None:
    text = _read(root / "Cargo.toml")
    if not text:
        return
    package = _toml_table(text, ("package",))
    name = package.get("name") if package else _toml_keys(text, "package")[0]
    if not isinstance(name, str):
        return
    _add(out, "dist", name)
    _add(out, "imports", name.replace("-", "_"))   # cargo's own mangling
    _add(out, "command", name)


_GO_MODULE = re.compile(r"^\s*module\s+(\S+)", re.M)


def _from_go_mod(root: Path, out: dict[str, list[str]]) -> None:
    text = _read(root / "go.mod")
    m = _GO_MODULE.search(text) if text else None
    if not m:
        return
    path = m.group(1)
    _add(out, "dist", path)
    _add(out, "imports", path)
    _add(out, "command", path.rsplit("/", 1)[-1])   # what `go install` leaves


def packaging_identity(root: Path) -> dict[str, list[str]]:
    """Distribution, import and command names declared at `root`.

    Every reader is best-effort and independent: a repository with a valid
    pyproject.toml and a truncated package.json reports what the first one
    said. `loci scan` walks every repository on the disk, so one unparseable
    manifest anywhere must not be able to end the scan.
    """
    out: dict[str, list[str]] = {"dist": [], "imports": [], "command": []}
    root = Path(root)
    for reader in (_from_pyproject, _from_package_json, _from_cargo, _from_go_mod):
        try:
            reader(root, out)
        except Exception:
            continue
    return out


def signboard(root: Path) -> dict:
    """What this project is called from the outside, for edge resolution.

    `remote` is `org/repo` and stays whole: the org is what makes two
    same-named repositories from different owners distinguishable, and an edge
    citing a git URL carries both.
    """
    from .provenance import remote_slug

    out: dict = dict(packaging_identity(root))
    slug = remote_slug(Path(root))
    out["remote"] = f"{slug[0]}/{slug[1]}" if slug else ""
    return out


def targets(sign: dict) -> set[str]:
    """Every string an edge could name this project by, lowercased.

    The repository name is admitted on its own as well as as `org/repo`: an
    edge that says `github:3M1RY33T/loci` carries the pair, and one that says
    `shutil.which("loci")` carries neither -- it carries a command that happens
    to equal the repository name, and on this corpus that is the only edge
    there is.
    """
    out: set[str] = set()
    for field in ("dist", "imports", "command"):
        out.update(t.lower() for t in (sign.get(field) or []) if t)
    remote = (sign.get("remote") or "").lower()
    if remote:
        out.add(remote)
        out.add(remote.rsplit("/", 1)[-1])
    return {t for t in out if t}
