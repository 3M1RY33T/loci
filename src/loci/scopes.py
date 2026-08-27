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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .defaults import DEFAULT_CODE_GLOBS, DEFAULT_EPISODE_GLOBS, SKIP_DIRS
from .paths import atomic_write, ensure_home, registry_file
from .types import Scope





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


# A depth-1 directory carrying one of these is a project in its own right.
# Depth-1 only, deliberately: descending further turns `client/static/package.json`
# into a scope called `static` and two `benchmarks/*/pyproject.toml` into
# benchmark harnesses nobody asked to route to.
WORKSPACE_MARKERS = ("package.json", "pyproject.toml", "Cargo.toml", "go.mod")

# JSON, not TOML. `tomllib` is read-only and 3.11+, the floor here is 3.10, and
# paths.py already argues this out for the registry.
DECLARATION_FILE = ".loci.json"


def declared_subscopes(root: Path) -> list[Path]:
    """Sub-scopes named in `<root>/.loci.json`, for what markers cannot see.

    A malformed file warns nobody and blocks nothing: a scan that aborts because
    of a stray comma is worse than one that misses a declaration. Well-formed
    JSON of the wrong shape is malformed too -- `{"scopes": "client"}` reaches
    `.get` on a string and raises AttributeError, which json.loads never does --
    so every level is type-checked rather than trusted.
    """
    f = root / DECLARATION_FILE
    if not f.is_file():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = raw.get("scopes") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        return []
    resolved_root = root.resolve()
    out: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rel = str(entry.get("path", "")).strip()
        if not rel:
            continue
        p = (root / rel).resolve()
        try:
            p.relative_to(resolved_root)
        except ValueError:
            continue                      # never escape the repository
        if p != resolved_root and p.is_dir() and p not in out:
            out.append(p)
    return out


def subscopes(root: Path) -> list[Path]:
    """Depth-1 directories inside `root` that are projects in their own right."""
    root = Path(root)
    out: list[Path] = []
    try:
        children = sorted(c for c in root.iterdir() if c.is_dir())
    except OSError:
        children = []
    for c in children:
        if c.name in SKIP_DIRS or c.name.startswith("."):
            continue
        if any((c / m).is_file() for m in WORKSPACE_MARKERS):
            out.append(c.resolve())
    for d in declared_subscopes(root):
        if d not in out:
            out.append(d)
    return out


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


def _unique_id(base: str, taken: set[str]) -> str:
    """A registry-unique id. Two repos may each hold a `client/`."""
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}-{n}", n + 1
    taken.add(candidate)
    return candidate


def discover(roots: list[Path], *, max_depth: int = 2) -> list[Scope]:
    """Find git repositories under `roots`, and the projects inside them.

    A repo is still the traversal boundary, but it is no longer necessarily one
    scope: a monorepo holding six independent projects fused into a single scope
    large enough to win every routing decision by size alone.
    """
    found: list[Scope] = []
    seen: set[Path] = set()
    taken: set[str] = set()
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
                parent = make_scope(d)
                parent.id = _unique_id(parent.id, taken)
                found.append(parent)
                for sub in subscopes(d):
                    sc = make_scope(sub, name=f"{parent.name}/{sub.name}")
                    sc.id = _unique_id(slugify(f"{parent.name}-{sub.name}"), taken)
                    # Containment is a label, not a parent pointer. The
                    # parent's own groups come along: a sub-project of a
                    # client's monorepo is the client's too.
                    sc.groups = sorted(set(parent.groups or []) | {parent.id})
                    found.append(sc)
                continue  # a repo is a leaf for TRAVERSAL; sub-scopes are handled here
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
    atomic_write(f, json.dumps(
        {"version": 1,
         "updated_at": datetime.now(timezone.utc).isoformat(),
         "scopes": [s.to_json() for s in scopes]},
        indent=2))
    return f


# Fields a re-scan must not clobber. `make_scope` regenerates defaults for all
# of them, so without this a user's `loci add --alias` is discarded the next
# time `loci scan` runs over the same directory.
PRESERVED_FIELDS = ("groups", "aliases", "episode_globs", "code_globs")


def upsert(scopes: list[Scope], new: Scope, *,
           preserve: tuple[str, ...] = PRESERVED_FIELDS) -> list[Scope]:
    """Add or replace by id, keeping user-edited fields from the existing entry.

    `new` is never modified: the caller still holds the scope it asked for, and
    editing it under them made `loci add` print values nobody had requested.

    Preservation is per field. `groups` and `aliases` carry over whole; a glob
    list carries over only the entries the current defaults do not provide, so
    a re-scan still delivers a newly shipped default (see the comment below).
    """
    old = next((s for s in scopes if s.id == new.id), None)
    if old is not None:
        carried: dict[str, Any] = {}
        # Read here, not at import: a module-level table would bind the lists
        # that existed when this module was first imported, which is the
        # staleness the glob branch below exists to prevent.
        defaults = {"episode_globs": DEFAULT_EPISODE_GLOBS,
                    "code_globs": DEFAULT_CODE_GLOBS}
        for field_name in preserve:
            current = getattr(old, field_name)
            # `groups` is tri-state: None means never inferred, so only an
            # inferred value (including a deliberate []) is worth preserving.
            if field_name == "groups":
                if current is not None:
                    carried["groups"] = list(current)
            elif not current:
                continue
            elif field_name in defaults:
                # Carry the DIFFERENCE, not the list. Preserving a stored glob
                # list wholesale freezes the defaults at whatever they were when
                # the scope was registered, and a re-scan is the only path by
                # which a newly shipped glob reaches an existing install -- there
                # is no CLI surface for code_globs at all. Comparing the stored
                # list against the current default cannot tell the two apart
                # either: once the defaults change, an untouched list differs
                # from them exactly as an edited one does.
                base = getattr(new, field_name)
                extras = [g for g in current
                          if g not in defaults[field_name] and g not in base]
                if extras:
                    carried[field_name] = list(base) + extras
            else:
                carried[field_name] = list(current)
        if carried:
            new = replace(new, **carried)
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
