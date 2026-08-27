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
