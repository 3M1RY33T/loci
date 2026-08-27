"""Group policy, and what it admits.

Membership is a property of a scope and lives in the registry. Mode is a
property of a group and lives here. Splitting them means a re-scan -- which
rewrites the registry wholesale -- cannot discard policy.

Nothing in this module shells out or touches the index. It turns
(policy, scopes, anchor) into two optional sets that `route` understands.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import atomic_write, ensure_home, groups_file
from .types import Scope

# Ordered loosest to strictest. MODES is derived rather than written out again:
# the two listed the same three names twice, and a mode added to one but not the
# other resolves to a KeyError deep inside `confining_groups`.
STRICTNESS = {"explicit": 0, "soft": 1, "hard": 2}
MODES = tuple(STRICTNESS)
DEFAULT_MODE = "soft"


@dataclass(slots=True)
class Policy:
    default_mode: str = DEFAULT_MODE
    groups: dict[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # An unknown mode fails two ways from one value: KeyError via cwd, and
        # nothing at all via --group, which returns a Confinement carrying a
        # mode no branch below implements. `load_policy` normalizes, so this
        # only ever fires on a Policy built in code -- i.e. from user text.
        if self.default_mode not in MODES:
            raise ValueError(
                f"unknown mode {self.default_mode!r}; expected one of "
                f"{', '.join(MODES)}")

    def mode_for(self, group: str) -> tuple[str, str]:
        """(mode, source). `source` is "declared" or "default"."""
        declared = self.groups.get(group)
        if declared in MODES:
            return declared, "declared"
        return self.default_mode, "default"

    def to_json(self) -> dict:
        return {"version": 1, "default_mode": self.default_mode,
                "groups": {g: {"mode": m} for g, m in sorted(self.groups.items())}}


@dataclass(slots=True)
class Confinement:
    """What a group does to one question."""

    # Tri-state, and the two falsy states are opposites: None is unconfined,
    # set() is confined to nothing (`--group` naming a group nobody is in).
    eligible: set[str] | None = None   # hard, or an explicit --group
    demoted: set[str] | None = None    # soft
    names: tuple[str, ...] = ()        # the confining groups themselves
    group: str | None = None           # display label only; see `names`
    mode: str | None = None
    source: str | None = None
    strict: bool = False               # abstain when the top scope is outside


def load_policy() -> Policy:
    """The policy on disk, or the shipped default if it is absent or unusable.

    Every malformed shape lands on the default rather than raising: this is the
    one file a user is invited to author by hand, and a traceback out of
    `loci ask` is a worse answer than "no groups configured". That includes
    valid JSON of the wrong shape -- a top-level list, or `groups` written as
    the list of names the key invites -- not just unparseable text.
    """
    f = groups_file()
    if not f.is_file():
        return Policy()
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Policy()
    if not isinstance(raw, dict):
        return Policy()
    default = raw.get("default_mode")
    declared = raw.get("groups")
    groups: dict[str, str | None] = {}
    if isinstance(declared, dict):
        for name, cfg in declared.items():
            mode = cfg.get("mode") if isinstance(cfg, dict) else cfg
            groups[name] = mode if mode in MODES else None
    return Policy(default_mode=default if default in MODES else DEFAULT_MODE,
                  groups=groups)


def save_policy(policy: Policy) -> Path:
    ensure_home()
    f = groups_file()
    atomic_write(f, json.dumps(policy.to_json(), indent=2))
    return f


def members(group_names, scopes: list[Scope]) -> set[str]:
    """Scope ids belonging to any of `group_names`."""
    want = set(group_names)
    return {s.id for s in scopes if want & s.group_set()}


def confining_groups(policy: Policy, scope_groups) -> tuple[list[str], str | None, str | None]:
    """The groups tied at the strictest mode, and that mode."""
    scope_groups = list(scope_groups)
    if not scope_groups:
        return [], None, None
    resolved = {g: policy.mode_for(g) for g in scope_groups}
    top = max(STRICTNESS[mode] for mode, _ in resolved.values())
    winners = sorted(g for g, (mode, _) in resolved.items()
                     if STRICTNESS[mode] == top)
    mode, source = resolved[winners[0]]
    return winners, mode, source


def confinement(policy: Policy, scopes: list[Scope], *,
                cwd: str | Path | None = None,
                forced_group: str | None = None) -> Confinement:
    """Resolve the confining group and what it does.

    `--group` names a GROUP directly; cwd names a SCOPE whose groups are then
    read. Different inputs, same resolution.
    """
    from .scopes import scope_for_cwd

    if forced_group:
        mode, source = policy.mode_for(forced_group)
        return Confinement(eligible=members([forced_group], scopes),
                           names=(forced_group,), group=forced_group,
                           mode=mode, source=source, strict=(mode == "hard"))

    anchor = scope_for_cwd(scopes, cwd) if cwd is not None else None
    if anchor is None:
        return Confinement()

    winners, mode, source = confining_groups(policy, anchor.group_set())
    if not winners or mode is None:
        return Confinement()

    # `group` is for printing. Two groups tied at the strictest mode join into
    # one string that is not the name of any group, so anything that needs to
    # look a group up again reads `names`.
    names = tuple(winners)
    label = ", ".join(names)
    if mode == "hard":
        return Confinement(eligible=members(names, scopes), names=names,
                           group=label, mode=mode, source=source, strict=True)
    if mode == "soft":
        inside = members(names, scopes)
        return Confinement(demoted={s.id for s in scopes} - inside, names=names,
                           group=label, mode=mode, source=source)
    return Confinement(names=names, group=label, mode=mode, source=source)
