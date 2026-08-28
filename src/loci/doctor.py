"""Coverage diagnostics.

Every experiment behind loci landed on the same finding: coverage is the
binding constraint, not ranking. A scope with 2 episode chunks and no graph
cannot be routed to or answered from, however good the algorithm is.

Most memory tools hide that. Naming it is a feature -- an empty answer with a
reason is worth more than a confident answer from the wrong project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A README that is still the framework's generated placeholder tells the user
# nothing, and tells loci nothing either. Cheap to detect, and it explains an
# otherwise baffling "no evidence" better than any score could.
TEMPLATE_MARKERS = (
    "a new flutter project",
    "this project is a starting point",
    "create-react-app",
    "npx create-next-app",
    "getting started with create react app",
    "# TODO: describe",
)

THIN_CHUNKS = 20
THIN_NODES = 200


@dataclass(slots=True)
class ScopeHealth:
    scope_id: str
    name: str
    root: str
    nodes: int
    tokens: int
    chunks: int
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check(index: dict, store: dict, registry: list | None = None) -> list[ScopeHealth]:
    """Health per scope.

    Takes the REGISTRY, not just the index. A scope with nothing indexable is
    skipped at build time and so has no entry in the index -- which means the
    worst coverage gaps are exactly the ones a doctor reading only the index
    cannot see. Reporting on what was registered is the whole point.
    """
    out: list[ScopeHealth] = []
    indexed = set(index["scopes"])
    for sc in (registry or []):
        if sc.id in indexed:
            continue
        h = ScopeHealth(scope_id=sc.id, name=sc.name, root=str(sc.root),
                        nodes=0, tokens=0, chunks=0)
        h.problems.append("not indexed at all - no structure graph and no prose found")
        h.problems.append("run `loci graphs` to add symbols (free, no LLM)")
        out.append(h)

    for sid, meta in index["scopes"].items():
        raw = store.get("chunks", {}).get(sid) or []
        h = ScopeHealth(scope_id=sid, name=meta["name"], root=meta.get("root", ""),
                        nodes=meta.get("structure_nodes", meta.get("node_count", 0)),
                        tokens=meta.get("token_count", 0),
                        chunks=len(raw))
        if not meta.get("sources"):
            h.problems.append("no structure graph; routed on prose alone - "
                              "run `loci graphs` (free, no LLM)")
        elif h.nodes < THIN_NODES:
            h.problems.append(f"thin structure index ({h.nodes} nodes)")
        if h.chunks == 0:
            h.problems.append("no episode content - no README, docs or git history found")
        elif h.chunks < THIN_CHUNKS:
            h.problems.append(f"thin episode store ({h.chunks} chunks); questions here will "
                              f"often return no evidence")
        blob = " ".join(c["text"][:400] for c in raw[:6]).lower()
        for marker in TEMPLATE_MARKERS:
            if marker in blob:
                h.problems.append("README looks like an unmodified framework template - "
                                  "nothing describes what this project is for")
                break
        out.append(h)
    return sorted(out, key=lambda s: (s.ok, s.name.lower()))


def group_report(scopes: list, policy) -> list[str]:
    """Group coverage, as renderable lines.

    Five things worth naming: a declared mode with no members (almost always a
    typo in a group name), vendor scopes still competing in the routable set,
    vendor scopes the hard-group remedy cannot reach because they are ALSO in
    `me`, the confinement nobody chose, and scopes in no group at all.

    The third is the one no other surface reports. `discover` writes a
    containment label onto every sub-project of a monorepo, `DEFAULT_MODE` is
    soft, and `ask` reads both on every call -- so the first question asked from
    inside a re-scanned monorepo demotes every scope outside it, and nothing
    announces that. Silent registration is the failure this feature exists to
    close; a silent default is the same failure with a better label on it.
    """
    from .router import GROUP_PENALTY

    lines: list[str] = []
    silent: list[str] = []
    names = sorted({g for s in scopes for g in s.group_set()} | set(policy.groups))
    for g in names:
        mode, source = policy.mode_for(g)
        # IDs, not names. Only the id is uniquified -- two clones of one
        # upstream repo are (`utils`, "utils") and (`utils-2`, "utils") -- so a
        # name is neither a stable partition key nor something the user can
        # paste back into `loci group rm`. `loci scopes` prints the id first for
        # the same reason.
        owned = sorted(s.id for s in scopes if g in s.group_set())
        if not owned:
            lines.append(f"group {g!r} declares mode {mode} but has no members "
                         f"- check the name")
        elif g.startswith("vendor:"):
            # NOT `group set {g} --mode explicit`. A mode governs what a scope
            # in that group does to questions asked from INSIDE it; it is not a
            # way to exclude the group from everyone else's questions, and
            # `explicit` is the mode that confines least. What takes a vendor
            # out of the routable set is the asking scope's own group being
            # hard. Measured: with `vendor:x` explicit the vendor still ranks;
            # with `me` hard it is "excluded by group me" -- but only for a
            # scope that is NOT itself in `me`. `groups infer` never retracts
            # `me`, so a scope can carry both, and `confinement` resolves a hard
            # `me` to `members(["me"])`, which is a set this scope is IN. The
            # remedy is then inert on exactly the scopes the line is about, and
            # the line calls the user's own work "not yours". Split, because one
            # sentence cannot be true of both halves.
            #
            # Partitioned on the ID. Keyed on the name, one dual-labelled clone
            # took its identically-named sibling out of `outside` with it, so
            # the sibling -- a plain vendor scope -- silently lost the remedy
            # that does work on it.
            dual = sorted(s.id for s in scopes
                          if g in s.group_set() and "me" in s.group_set())
            outside = [n for n in owned if n not in set(dual)]
            if outside:
                lines.append(f"{g}: {', '.join(outside)} - not yours, and still "
                             f"in the routable set. `loci group set me --mode "
                             f"hard` (or whichever group your own work is in) "
                             f"keeps them out of every question asked from "
                             f"inside it.")
            if dual:
                # A real id in the command, not a `<scope>` placeholder: the
                # scopes this line is about are exactly the ones whose NAME may
                # be shared, and `resolve` answers an id before any name.
                lines.append(f"{g}: {', '.join(dual)} - also in `me`, so "
                             f"`loci group set me --mode hard` does NOT keep "
                             f"them out: a hard group admits its own members. "
                             f"That is right for your own work under a second "
                             f"org, and wrong for a repository a scan called "
                             f"yours before it could name your org - "
                             f"`loci group rm {dual[0]} me` settles which.")
        elif source == "declared":
            lines.append(f"{g}: mode {mode} (declared), {len(owned)} member(s)")
        elif mode != "explicit":
            silent.append(g)

    if silent:
        # Reported once, not per group: a scope in two groups is confined by
        # their union, so "every scope outside THIS group" would be false of
        # every line it appeared on.
        effect = (f"scores every scope outside that member's groups "
                  f"x{GROUP_PENALTY}" if policy.default_mode == "soft"
                  else "reaches nothing outside that member's groups")
        # "stops one confining", not "turns it off": `confining_groups` takes
        # the strictest mode among ALL of the anchor's groups, so a scope in
        # three soft groups is still confined after one of them goes explicit.
        lines.append(f"{len(silent)} group(s) confine on the default mode, which "
                     f"nobody chose: {', '.join(silent)} - a question asked from "
                     f"inside a member {effect}. "
                     f"`loci group set <group> --mode explicit` stops one "
                     f"confining; a scope is unconfined once every group it is "
                     f"in is explicit.")

    ungrouped = sorted(s.name for s in scopes if not s.group_set())
    if ungrouped:
        shown = ", ".join(ungrouped[:6])
        more = f" +{len(ungrouped) - 6} more" if len(ungrouped) > 6 else ""
        lines.append(f"{len(ungrouped)} scope(s) in no group: {shown}{more} "
                     f"- `loci groups infer` labels them")
    return lines


def render(healths: list[ScopeHealth], stale: list[str] | None = None) -> str:
    lines = [f"{'scope':<20} {'nodes':>8} {'tokens':>7} {'chunks':>7}  status",
             "-" * 74]
    for h in healths:
        status = "ok" if h.ok else f"{len(h.problems)} issue(s)"
        lines.append(f"{h.name:<20} {h.nodes:>8} {h.tokens:>7} {h.chunks:>7}  {status}")
        for p in h.problems:
            lines.append(f"{'':<20} {'':>8} {'':>7} {'':>7}    - {p}")
    bad = sum(1 for h in healths if not h.ok)
    lines.append("-" * 74)
    lines.append(f"{len(healths)} scopes, {bad} with coverage gaps")
    if stale is None:
        lines.append("no embeddings built - semantic search is off "
                     "(`loci embed` enables it)")
    elif stale:
        lines.append(f"STALE embeddings for {', '.join(stale)} - semantic search "
                     f"is silently off for those scopes; run `loci embed`")
    return "\n".join(lines)
