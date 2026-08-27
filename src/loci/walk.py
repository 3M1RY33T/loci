"""Directory traversal that prunes instead of filtering.

`Path.glob("**/*.py")` has no way to skip a subtree: it descends into
node_modules and .venv in full and leaves the caller to discard the results
afterwards. Measured on one real repo that cost 32.9s to enumerate; pruning the
walk brings the same enumeration to well under a second.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .defaults import SKIP_DIRS


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern:
    """Translate a glob to a regex with pathlib semantics, not fnmatch's.

    Two differences matter and both caused real bugs:

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
    excluded = set()
    for e in exclude or ():
        try:
            excluded.add(Path(e).resolve())
        except (OSError, ValueError):
            continue

    out: list[Path] = []
    seen: set[Path] = set()
    rel_patterns = []
    for pat in patterns or []:
        p = Path(pat).expanduser()
        if p.is_absolute():
            try:
                base = Path(p.anchor)
                for f in sorted(base.glob(str(p.relative_to(base)))):
                    if f.is_file() and f not in seen:
                        seen.add(f)
                        out.append(f)
            except (OSError, ValueError):
                continue
        else:
            rel_patterns.append(pat)
    if not rel_patterns:
        return out

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        d = Path(dirpath)
        dirnames[:] = [
            x for x in dirnames
            if x not in SKIP_DIRS and not x.startswith(".")
            # `resolve` is a syscall per directory, so it is asked only when
            # there is something for it to be compared against.
            and not (excluded and (d / x).resolve() in excluded)
        ]
        for name in filenames:
            f = d / name
            if f in seen:
                continue
            try:
                rel = str(f.relative_to(root))
            except ValueError:
                continue
            for pat in rel_patterns:
                if _matches(rel, pat):
                    seen.add(f)
                    out.append(f)
                    break
    return sorted(out)
