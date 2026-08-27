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
from .groups import Policy, confinement, load_policy
from .index import chunks_for, load_episodes, load_index
from .router import route
from .scopes import load_scopes
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


# How many nearest symbol labels contribute tokens. Measured: 1 or 2 give 6/7
# on the structure probes, 3 or 4 give 5/7. Past two labels the extra tokens are
# generic -- "logic", "admin", "mutation" -- and they dilute a query that the
# lexical expansion had already aimed correctly.
SEMANTIC_SYMBOL_LABELS = 2


def semantic_symbols(question: str, scope_id: str,
                     k: int = SEMANTIC_SYMBOL_LABELS) -> list[str]:
    """Tokens of the symbol labels nearest this question, or [] when unavailable.

    graphify picks traversal seeds by lexical similarity to a label, so a
    question phrased in behaviour rather than in identifiers reaches nothing:
    "what parses the command line arguments?" expanded to no graph token at all
    and produced an empty traversal. Matching the question against embedded
    labels first, then handing graphify the tokens of the nearest ones, bridges
    that gap without changing how graphify traverses.
    """
    import json

    from .backends import episodes as ep
    from .index import SYMBOL_PREFIX
    from .paths import embeddings_file
    from .text import unique_tokens

    emb = ep._embeddings()
    key = f"{SYMBOL_PREFIX}{scope_id}"
    if not emb or key not in emb:
        return []
    names_file = embeddings_file().parent / f".symbols-{scope_id}.json"
    if not names_file.is_file():
        return []
    try:
        labels = json.loads(names_file.read_text(encoding="utf-8"))
        qv = ep._encode_query(question, str(emb["_model"][0]))
        if qv is None:
            return []
        import numpy as np
        order = np.argsort(-(emb[key] @ qv))[:k]
    except Exception:
        return []
    out: list[str] = []
    for i in order:
        if i < len(labels):
            out.extend(unique_tokens(labels[i]))
    return list(dict.fromkeys(out))


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
        group: str | None = None,
        policy: Policy | None = None, registry: list[Scope] | None = None,
        index: dict | None = None, store: dict | None = None) -> Answer:
    index = index if index is not None else load_index()
    store = store if store is not None else (load_episodes() if with_episodes else {})

    if force_scopes:
        # An explicit --scope outranks any group policy: the user has already
        # answered the question groups exist to answer.
        selected = force_scopes
        rt = RouteResult(question=question, query_tokens=unique_tokens(question),
                         ranked=selected, selected=selected, abstain=False,
                         top_score=0.0, top_matched=0,
                         detail={s: {"name": index["scopes"][s]["name"],
                                     "signals": {"forced": "cli"}} for s in selected})
    else:
        # Confinement is resolved from the REGISTRY, never from the index: the
        # index stores only id/name/root/aliases per scope, so the `Scope` that
        # `work` rebuilds from it below can never carry `groups`.
        #
        # `policy` and `registry` are injectable for the same reason `index` and
        # `store` are -- a caller already holding them should not re-read
        # $LOCI_HOME, and a test should not need one at all.
        policy = load_policy() if policy is None else policy
        if registry is None:
            # A malformed registry degrades to unconfined rather than raising.
            # `load_policy` already swallows every bad shape on the stated
            # ground that a traceback out of `loci ask` is a worse answer than
            # "no groups configured", and this is the same new code path: until
            # confinement arrived, `ask` ran off the index alone and never read
            # the registry at all. Measured, a hand-edited scopes.json raises
            # four different types -- JSONDecodeError, AttributeError, TypeError
            # and KeyError -- depending on which level is wrong, so the catch
            # is broad, like `_calibrated_floor`, which guards another
            # disk-loaded routing input the same way.
            #
            # The guard belongs HERE and not in `load_scopes`: `cmd_scan` and
            # `cmd_add` both do `save_scopes(upsert(load_scopes(), ...))`, so a
            # silent [] there would rewrite the registry with every scope but
            # the new one discarded.
            try:
                registry = load_scopes()
            except Exception:
                registry = []
        conf = confinement(policy, registry, cwd=cwd, forced_group=group)
        rt = route(question, index, cwd=cwd,
                   eligible=conf.eligible, demoted=conf.demoted,
                   strict_group=conf.strict, group=conf.group, mode=conf.mode)
        selected = [] if rt.abstain else rt.selected

    sb = get_structure_backend(index.get("structure_backend", "graphify"))
    eb = get_episode_backend()
    per_scope_budget = max(250, budget // max(1, len(selected)))

    def work(sid: str) -> ScopeAnswer:
        meta = index["scopes"][sid]
        ans = ScopeAnswer(scope_id=sid, name=meta["name"],
                          expanded=expand_for_scope(question, index, sid))
        if with_structure and sb.available():
            # Lexical expansion first, semantic symbols appended. Both, not
            # either: the lexical half anchors terms the user actually typed,
            # the semantic half reaches code they described instead of named.
            extra = [t for t in semantic_symbols(question, sid)
                     if t not in ans.expanded]
            ans.expanded = ans.expanded + extra
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
        # An abstention that does not name its cause is indistinguishable from a
        # bug, and a hard group turns answers into abstentions -- so this line is
        # the whole mitigation for hard mode reading as a regression.
        #
        # The cause REPLACES the old headline rather than trailing it: "not
        # specific enough to route" is false for `out_of_group`, where the
        # question was specific and merely aimed outside the group.
        if rt.group and not rt.ranked:
            # `eligible` filtered every scope out, so `route` weighed the
            # question against an empty candidate set and reported `no_evidence`
            # -- true of that set, false of the question, whose vocabulary was
            # never the problem. Naming the group is the only honest cause, and
            # `--group <typo>` is the likeliest way a user reaches this line.
            reason = f"no indexed project is in group {rt.group}"
        else:
            reason = {
                "out_of_group": f"best match was outside group {rt.group}",
                "deictic": "the question points at its subject without naming it",
                "no_evidence": "not enough of the question exists in any project",
            }.get(rt.abstain_reason or "", "not specific enough to route")
        out.append(f"ABSTAINED - {reason}.")
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
