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
from .provenance import is_provenance
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
#
# OFF by default -- `--split` opts in -- and the reason is `_aliases_for`, not
# anything in this function. A new scope's aliases include its bare directory
# name, and `ALIAS_BOOST` is 6.0 against `CWD_BOOST`'s 4.0, so a sub-project
# called `glasses` promotes an ordinary English noun to the strongest routing
# signal in the corpus. Measured on the real corpus (evals/RESULTS.md, "Does a
# question about `glasses` reach `delroy-glasses`?"): eight hand-written
# questions about a DIFFERENT project that merely contain the word, 7/8 routed
# correctly before the split and 0/8 after, six of the eight demonstrably caused
# by the alias -- and `_site`, a Jekyll build directory, became a scope of its
# own. Nothing here is deleted or weakened: the split works, and the follow-up
# that answers the alias question turns this back on.
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


def subscopes(root: Path, *, markers: bool = False) -> list[Path]:
    """Depth-1 directories inside `root` that are projects in their own right.

    A `.loci.json` declaration is honoured whatever `markers` says, and that
    asymmetry is the point of the flag. Writing that file is the user asserting
    "these are separate projects" about one repository they know; a workspace
    marker is loci guessing it about every repository in the corpus, and the
    guess costs what WORKSPACE_MARKERS records above.
    """
    root = Path(root)
    out: list[Path] = []
    if markers:
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


def discover(roots: list[Path], *, max_depth: int = 2,
             markers: bool = False) -> list[Scope]:
    """Find git repositories under `roots`, and the projects inside them.

    A repo is still the traversal boundary, but it is no longer necessarily one
    scope: a monorepo holding six independent projects fused into a single scope
    large enough to win every routing decision by size alone.

    `markers` is threaded straight to `subscopes` and nothing else here reads it,
    so with it off a repository declaring nothing is one scope again, exactly as
    it was before the split; a repository carrying `.loci.json` splits either
    way. See WORKSPACE_MARKERS for what the default costs and why it is paid.
    """
    repos: list[Path] = []
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
                repos.append(d)
                continue  # a repo is a leaf for TRAVERSAL; sub-scopes come later
            if depth >= max_depth:
                continue
            try:
                # Sorted, and reversed because the stack is LIFO: unordered
                # `iterdir` made id assignment depend on filesystem order, and
                # `upsert` matches on id while `root` is not preserved -- so two
                # scopes trading which one holds the bare id would land one
                # project's root in the other's registry entry.
                children = sorted((c for c in d.iterdir() if c.is_dir()), reverse=True)
            except PermissionError:
                continue
            stack.extend((c, depth + 1) for c in children)

    # Two passes, deliberately: every real repository claims its id before any
    # generated sub-scope id exists to take it.
    found: list[Scope] = []
    taken: set[str] = set()
    families: list[tuple[Scope, list[Path]]] = []
    for d in repos:
        parent = make_scope(d)
        parent.id = _unique_id(parent.id, taken)
        subs = subscopes(d, markers=markers)
        if subs:
            # The container is a member of its own containment group. Excluding
            # it makes `members(["delroy"])` return the sub-projects but not the
            # monorepo root, which still owns the code no sub-project claimed.
            parent.groups = [parent.id]
        found.append(parent)
        families.append((parent, subs))
    for parent, subs in families:
        for sub in subs:
            sc = make_scope(sub, name=f"{parent.name}/{sub.name}")
            sc.id = _unique_id(slugify(f"{parent.name}-{sub.name}"), taken)
            # Containment is a label, not a parent pointer, and it is structural:
            # computed from the filesystem, like everything else `discover` reads.
            # Inheriting a user's own groups belongs at registration time, where
            # the loaded registry is in hand; `discover` never reads it.
            sc.groups = [parent.id]
            found.append(sc)
    return sorted(found, key=lambda s: s.name.lower())


def load_scopes() -> list[Scope]:
    f = registry_file()
    if not f.is_file():
        return []
    raw = json.loads(f.read_text(encoding="utf-8"))
    return [Scope.from_json(d) for d in raw.get("scopes", [])]


def load_roots() -> list[Path]:
    """Directories a previous scan was pointed at.

    The registry records every scope's own root, and none of them answers the
    question `loci update` has to ask: where would a repository that did not
    exist yet turn up? A scope root is the answer for a project already known.
    Deriving the search root from it -- taking each scope's parent -- guesses,
    and guesses wrong in the one case that matters: a scope registered at the
    home directory would nominate the whole home directory for scanning.

    Returned verbatim, including entries that are not directories right now. An
    unmounted volume or an unplugged disk is not a reason to forget where the
    user keeps their code; the caller decides what to do about a root it cannot
    reach today.
    """
    f = registry_file()
    if not f.is_file():
        return []
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [Path(r) for r in raw.get("roots", []) if isinstance(r, str)]


def save_scopes(scopes: list[Scope], *, roots: list[Path] | None = None) -> Path:
    """Write the registry, and remember where its scopes were scanned from.

    `roots` accumulates rather than replaces: someone who scanned `~/code` in
    March and `~/work` in June wants `loci update` to look in both. Passing
    None keeps what is already recorded, which is what makes the default safe
    -- `add`, `group` and `groups infer` all rewrite the scope list without
    scanning anything, and a default of "forget" would blind the next update
    every time one of them ran.

    `loci add` deliberately records nothing: it registers one directory, and
    entering it here would send a later scan hunting for repositories INSIDE a
    project the user named on purpose.
    """
    ensure_home()
    f = registry_file()
    known = {str(p) for p in load_roots()}
    for r in roots or []:
        try:
            known.add(str(Path(r).expanduser().resolve()))
        except OSError:
            continue
    atomic_write(f, json.dumps(
        {"version": 1,
         "updated_at": datetime.now(timezone.utc).isoformat(),
         "roots": sorted(known),
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

    Preservation is per field. `aliases` carries over whole; `groups` unions,
    because discovery now computes labels of its own that a user's list must not
    erase; a glob list carries over only the entries the current defaults do not
    provide, so a re-scan still delivers a newly shipped default (see below).
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
                    # Union, not replace. `discover` computes structural labels
                    # -- a monorepo's containment group -- and a stored user list
                    # must not be able to erase them: `group set` on a sub-scope
                    # would otherwise drop its parent label on the next scan,
                    # permanently. Removal still works for every label discovery
                    # does not re-assert. Appended rather than sorted, so a list
                    # the user ordered by hand comes back in that order.
                    carried["groups"] = list(current) + [
                        g for g in (new.groups or []) if g not in current]
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


def repositories(scopes: list[Scope]) -> list[Scope]:
    """The scopes in `scopes` that no other one contains.

    `discover` returns a monorepo AND the projects inside it, so `len(found)`
    counts scopes and not repositories -- the setup line read "5 git
    repositories" for one repo holding four projects, in the onboarding flow,
    where a wrong number is least likely to be questioned.

    It is also the wrong input to `infer_identity`: git resolves `origin` by
    walking upward, so every sub-scope votes again for its parent's org and one
    monorepo can outvote the rest of the corpus.
    """
    return [s for s in scopes
            if not any(o.root in s.root.parents for o in scopes)]


def inherit_parent_groups(scopes: list[Scope]) -> list[Scope]:
    """Give each sub-scope the groups of the container it lives in.

    `discover` cannot do this and the attempt was deleted from it: discovery
    reads the filesystem and never the registry, so the parent it constructs
    always has `groups is None` and the union there is unreachable. Here the
    registry IS loaded, which is why this runs at registration time -- a user
    who tagged the monorepo `client:acme` gets its sub-projects into
    `client:acme` too, and a hard group then confines to the client's work
    rather than to one directory of it.

    Containment is matched on the LABEL and on the PATH, both. A scope is free
    to join a group whose name happens to be some container's id without being
    anything of that container's (`loci group add solo mono` is legal), and
    inheriting there would hand it labels nobody asserted.

    PROVENANCE is never inherited. It is read from one repository's own git, and
    a repository vendored into a monorepo -- `third_party/`, `external/`, a
    submodule carrying a workspace marker -- is still the vendor's however it
    got there. Inheriting `me` onto it silently undoes the classification the
    user was asked about, and then `doctor` says `loci group set me --mode hard`
    keeps it out of the routable set when it no longer would. Every scope scan
    registers gets its own label from its own git, so there is nothing for
    inheritance to supply; a scope missing one is what `loci groups infer` is
    for, and that reads git too rather than guessing from a container.

    Purely additive, which `groups infer` no longer is: that command retracts a
    `vendor:` label its fresh reading of git contradicts. Nothing here reads git,
    so there is nothing to contradict -- a label inherited once survives the
    parent losing it. Membership is a claim, and the user removes their own
    claims with `loci group rm`.
    """
    by_id = {s.id: s for s in scopes}
    # Snapshotted so the result does not depend on registry order: a sub-scope
    # visited before its parent must inherit the same set as one visited after.
    frozen = {s.id: s.group_set() for s in scopes}
    for s in scopes:
        inherited: set[str] = set()
        for g in frozen[s.id]:
            parent = by_id.get(g)
            if parent is None or parent.id == s.id:
                continue
            if parent.root in s.root.parents:
                inherited |= {p for p in frozen[parent.id]
                              if not is_provenance(p)}
        if inherited - frozen[s.id]:
            s.groups = sorted(frozen[s.id] | inherited)
    return scopes


def resolve(scopes: list[Scope], want: str) -> Scope | None:
    """Look up a scope by id, name or alias (case-insensitive).

    An exact ID match wins over every name and alias in the registry, and that
    ordering is what makes an id a HANDLE rather than a hint. Only the id is
    uniquified -- `make_scope` takes `name` from the directory, `_unique_id`
    suffixes the id -- so two clones of one upstream repo register as
    (`utils`, "utils") and (`utils-2`, "utils"). Scanning them in one pass, the
    three-way `or` below returned whichever came first in registry order, which
    is filesystem order: `resolve("utils")` answered `utils-2`, leaving the
    scope whose id is literally `utils` unreachable by any string at all.
    `loci group rm utils me` then edited the wrong scope, printed success, and
    left the intended one exactly as it was -- forever, since the next run
    resolved the same way.

    Two passes, not one, because the fix has to be about the whole registry:
    "prefer this scope's id to this scope's name" inside the loop still lets an
    earlier scope's NAME beat a later scope's id.
    """
    w = want.strip().lower()
    for s in scopes:
        if s.id.lower() == w:
            return s
    for s in scopes:
        if s.name.lower() == w or w in {a.lower() for a in s.aliases}:
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


def nested_roots(scope: Scope, all_scopes: list[Scope]) -> list[Path]:
    """Roots of other registered scopes lying inside this one.

    Computed, never stored: exclusion is a property of the registry as a whole,
    and a stored copy is wrong the moment a sibling is added or removed.
    """
    return [s.root for s in all_scopes
            if s.id != scope.id and scope.root in s.root.parents]
