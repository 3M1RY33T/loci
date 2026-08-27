"""Stage 2 and 3: query the routed scopes, merge the answer.

    question -> route -> for each selected scope, in parallel:
                           structure store (what calls what)
                           episode store   (what happened and why)
             -> merge

Scoped and cross-scope questions take the SAME path; only the size of the scope
set differs. That is the whole claim of the design: "how does auth work here"
and "have I solved this in any project" are one mechanism, not two.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_episode_backend, get_structure_backend
from .index import chunks_for, load_episodes, load_index
from .router import route
from .text import unique_tokens
from .types import EpisodeHit, RouteResult, Scope, StructureHit


@dataclass(slots=True)
class ScopeAnswer:
    scope_id: str
    name: str
    expanded: list[str] = field(default_factory=list)
    structure: list[StructureHit] = field(default_factory=list)
    episodes: list[EpisodeHit] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return {
            "scope_id": self.scope_id, "name": self.name,
            "expanded": self.expanded,
            "structure": [h.to_json() for h in self.structure],
            "episodes": [h.to_json() for h in self.episodes],
            "note": self.note,
        }


@dataclass(slots=True)
class Answer:
    question: str
    routing: RouteResult
    scopes: list[ScopeAnswer] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"question": self.question, "routing": self.routing.to_json(),
                "scopes": [s.to_json() for s in self.scopes]}


def expand_for_scope(question: str, index: dict, scope_id: str) -> list[str]:
    """Keep only the query tokens this scope's index actually contains.

    graphify's own skill solves wording mismatch by dumping a vocabulary file
    and having a model pick tokens out of it, under the rule "never invent a
    token". The scope index already IS that vocabulary, per scope, so the
    expansion is a dict lookup and the rule holds by construction.
    """
    postings = index["postings"]
    return [t for t in unique_tokens(question) if scope_id in postings.get(t, {})]


def ask(question: str, *, cwd: str | Path | None = None, budget: int = 2000,
        episodes_k: int = 3, dfs: bool = False, rerank: bool = False,
        with_structure: bool = True, with_episodes: bool = True,
        force_scopes: list[str] | None = None,
        index: dict | None = None, store: dict | None = None) -> Answer:
    index = index if index is not None else load_index()
    store = store if store is not None else (load_episodes() if with_episodes else {})

    if force_scopes:
        selected = force_scopes
        rt = RouteResult(question=question, query_tokens=unique_tokens(question),
                         ranked=selected, selected=selected, abstain=False,
                         top_score=0.0, top_matched=0,
                         detail={s: {"name": index["scopes"][s]["name"],
                                     "signals": {"forced": "cli"}} for s in selected})
    else:
        rt = route(question, index, cwd=cwd)
        selected = [] if rt.abstain else rt.selected

    sb = get_structure_backend(index.get("structure_backend", "graphify"))
    eb = get_episode_backend()
    per_scope_budget = max(250, budget // max(1, len(selected)))

    def work(sid: str) -> ScopeAnswer:
        meta = index["scopes"][sid]
        ans = ScopeAnswer(scope_id=sid, name=meta["name"],
                          expanded=expand_for_scope(question, index, sid))
        if with_structure and sb.available() and ans.expanded:
            sc = Scope(id=sid, name=meta["name"], root=Path(meta["root"]),
                       aliases=meta.get("aliases", []))
            ans.structure = sb.query(sc, " ".join(ans.expanded),
                                     budget=per_scope_budget, dfs=dfs)
        if with_episodes:
            chunks = chunks_for(store, sid)
            if chunks:
                ans.episodes = eb.search(question, chunks, sid, k=episodes_k,
                                         rerank=rerank, gate=not force_scopes)
        if not ans.structure and not ans.episodes and not ans.note:
            ans.note = "no evidence in this scope"
        return ans

    answers: list[ScopeAnswer] = []
    if selected:
        with ThreadPoolExecutor(max_workers=min(8, len(selected))) as pool:
            answers = list(pool.map(work, selected))
    return Answer(question=question, routing=rt, scopes=answers)


def render(answer: Answer, *, index: dict, chars: int = 400) -> str:
    out: list[str] = []
    rt = answer.routing
    names = {sid: m["name"] for sid, m in index["scopes"].items()}

    if rt.abstain:
        out.append("ABSTAINED - not specific enough to route.")
        out.append(f"  candidates: {', '.join(names.get(s, s) for s in rt.ranked)}")
        out.append("  re-run with --scope <name>, or from inside the project directory.")
        return "\n".join(out)

    how = "forced" if not rt.ranked or rt.top_matched == 0 else f"score={rt.top_score:.3f}"
    out.append(f"ROUTED -> {', '.join(a.name for a in answer.scopes)}   ({how})")

    for a in answer.scopes:
        out.append("")
        out.append("=" * 70)
        out.append(f"{a.name}   expanded: {a.expanded or '(none)'}")
        out.append("=" * 70)
        if a.note and not a.structure and not a.episodes:
            out.append(f"  -- {a.note}")
            continue
        for h in a.structure:
            out.append(f"\n--- structure [{h.source}: {h.nodes} nodes]")
            out.append(h.text if h.ok else f"  ERROR {h.text}")
        if a.episodes:
            out.append("\n--- episodes")
            for e in a.episodes:
                where = f"{e.chunk.kind}:{e.chunk.source}"
                if e.chunk.heading:
                    where += f"  > {e.chunk.heading}"
                out.append(f"\n  [{e.score:.4f}] {where}")
                body = " ".join(e.chunk.text.split())
                out.append(f"      {body[:chars]}{'...' if len(body) > chars else ''}")
        elif not a.structure:
            out.append("\n--- episodes: no content evidence")
    return "\n".join(out)
