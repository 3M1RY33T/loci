"""Directory traversal that prunes instead of filtering.

`Path.glob("**/*.py")` has no way to skip a subtree: it descends into
node_modules and .venv in full and leaves the caller to discard the results
afterwards. Measured on one real repo that cost 32.9s to enumerate; pruning the
walk brings the same enumeration to well under a second.

The traversal is Rust, in ``rust/loci-core/src/walk.rs``. Two things stay here:

* **Absolute patterns.** They are rare -- a scope pulling in prose that lives
  outside its own tree -- and they need ``expanduser`` and ``Path.glob``
  semantics that are fiddly to reproduce for no measurable gain. The result is
  sorted at the end, so the two halves merge without caring which found what.
* **``_matches``**, because the test suite imports it directly.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from loci._core import glob_matches as _rs_glob_matches
from loci._core import iter_files as _rs_iter_files

from .defaults import SKIP_DIRS


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern:
    """Translate a glob to a regex with pathlib semantics, not fnmatch's.

    Kept in Python because the tests reach for it. The traversal uses the Rust
    translation, and `test_rust_parity.py` asserts the two agree.

    Two differences from fnmatch matter and both caused real bugs:

    `fnmatch` lets a single ``*`` cross a path separator, so ``"*.md"`` -- which
    pathlib treats as root-only -- would match every nested markdown file.

    `fnmatch` has no notion of ``**`` at all, so ``"docs/**/*.md"`` compiled to
    something requiring an intervening directory and silently skipped
    ``docs/guide.md``. Measured: 28 documentation files dropped across the test
    corpus, 25 of them from one project. pathlib's ``**`` matches ZERO or more
    directories, and so does this.
    """
    i, out = 0, []
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")     # zero or more directories
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _matches(rel: str, pattern: str) -> bool:
    # Windows hands back backslashes; every pattern here is written with
    # forward slashes, so normalize rather than branch on os.sep.
    return _compile(pattern).match(rel.replace("\\", "/")) is not None


def _absolute_matches(patterns: list[str]) -> list[Path]:
    """Files named by absolute or ``~`` patterns, honoured as-is."""
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat).expanduser()
        if not p.is_absolute():
            continue
        try:
            base = Path(p.anchor)
            for f in sorted(base.glob(str(p.relative_to(base)))):
                if f.is_file():
                    out.append(f)
        except (OSError, ValueError):
            continue
    return out


def iter_files(root: Path, patterns: list[str], *,
               exclude: "list[Path] | tuple[Path, ...]" = ()) -> list[Path]:
    """Files under `root` matching any glob, skipping vendored subtrees.

    Absolute patterns are honoured as-is so a scope can pull in prose that
    lives outside its own tree.

    `exclude` names subtrees owned by other scopes. They are PRUNED rather than
    filtered afterwards, for the same reason SKIP_DIRS is: descending a
    sub-scope's node_modules to discard the result is the cost this module
    exists to avoid.
    """
    pats = list(patterns or [])
    absolute = [p for p in pats if Path(p).expanduser().is_absolute()]
    relative = [p for p in pats if p not in absolute]

    found = _absolute_matches(absolute)
    if relative:
        found += [Path(p) for p in _rs_iter_files(
            str(root), relative, [str(e) for e in (exclude or ())],
            sorted(SKIP_DIRS))]

    seen: set[Path] = set()
    out: list[Path] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return sorted(out)
