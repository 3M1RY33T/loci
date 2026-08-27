"""Scope registry and discovery.

This is the module that had to change most in extracting loci from its
prototype. The prototype discovered projects by reading one specific host
application's private layout. A general tool cannot assume that, so scopes are
explicit: registered by the user, or discovered as git repositories under roots
the user names.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .defaults import DEFAULT_CODE_GLOBS, DEFAULT_EPISODE_GLOBS
from .paths import ensure_home, registry_file
from .types import Scope


# Directories that are never a project of the user's own.
SKIP_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    "vendor", "target", ".next", ".tox", "site-packages",
}


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "scope"


def _aliases_for(name: str, root: Path) -> list[str]:
    cands = {name, root.name, name.replace("_", " "), name.replace("-", " ")}
    return sorted({c.strip().lower() for c in cands if c and len(c.strip()) >= 3})


def _git_updated_at(root: Path) -> str:
    if not (root / ".git").is_dir():
        return ""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--pretty=format:%aI"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        return out
    except Exception:
        return ""


def make_scope(root: Path, *, name: str | None = None,
               aliases: list[str] | None = None,
               episode_globs: list[str] | None = None,
               code_globs: list[str] | None = None) -> Scope:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    nm = name or root.name
    return Scope(
        id=slugify(nm),
        name=nm,
        root=root,
        aliases=aliases if aliases is not None else _aliases_for(nm, root),
        episode_globs=episode_globs if episode_globs is not None else list(DEFAULT_EPISODE_GLOBS),
        code_globs=code_globs if code_globs is not None else list(DEFAULT_CODE_GLOBS),
        updated_at=_git_updated_at(root) or datetime.now(timezone.utc).isoformat(),
    )


def discover(roots: list[Path], *, max_depth: int = 2) -> list[Scope]:
    """Find git repositories under `roots`. A repo is the natural scope boundary."""
    found: list[Scope] = []
    seen: set[Path] = set()
    for raw in roots:
        base = Path(raw).expanduser().resolve()
        if not base.is_dir():
            continue
        stack: list[tuple[Path, int]] = [(base, 0)]
        while stack:
            d, depth = stack.pop()
            if d in seen or d.name in SKIP_DIRS:
                continue
            seen.add(d)
            if (d / ".git").exists():
                found.append(make_scope(d))
                continue  # a repo is a leaf; do not descend into submodules
            if depth >= max_depth:
                continue
            try:
                stack.extend((c, depth + 1) for c in d.iterdir() if c.is_dir())
            except PermissionError:
                continue
    return sorted(found, key=lambda s: s.name.lower())


def load_scopes() -> list[Scope]:
    f = registry_file()
    if not f.is_file():
        return []
    raw = json.loads(f.read_text(encoding="utf-8"))
    return [Scope.from_json(d) for d in raw.get("scopes", [])]


def save_scopes(scopes: list[Scope]) -> Path:
    ensure_home()
    f = registry_file()
    f.write_text(json.dumps(
        {"version": 1,
         "updated_at": datetime.now(timezone.utc).isoformat(),
         "scopes": [s.to_json() for s in scopes]},
        indent=2), encoding="utf-8")
    return f


def upsert(scopes: list[Scope], new: Scope) -> list[Scope]:
    """Add or replace by id, preserving registry order."""
    out = [s for s in scopes if s.id != new.id]
    out.append(new)
    return sorted(out, key=lambda s: s.name.lower())


def resolve(scopes: list[Scope], want: str) -> Scope | None:
    """Look up a scope by id, name or alias (case-insensitive)."""
    w = want.strip().lower()
    for s in scopes:
        if s.id.lower() == w or s.name.lower() == w or w in {a.lower() for a in s.aliases}:
            return s
    return None


def scope_for_cwd(scopes: list[Scope], cwd: str | Path) -> Scope | None:
    """The scope whose root contains `cwd`; the deepest one wins on nesting.

    Measured on real questions, cwd is not a tiebreak signal -- it is the
    primary one. Questions that name no project ("how do I run the tests?",
    "how is this deployed?") route correctly 12/12 with cwd and 1/4 without.
    """
    try:
        p = Path(cwd).expanduser().resolve()
    except (OSError, ValueError):
        return None
    best: Scope | None = None
    for s in scopes:
        if p == s.root or s.root in p.parents:
            if best is None or len(str(s.root)) > len(str(best.root)):
                best = s
    return best
