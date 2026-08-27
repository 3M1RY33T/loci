#!/usr/bin/env python3
"""A retrieval metric for fitting the fusion weights, without a lexical bias.

`loci eval` scores routing only. The fusion weights, `MIN_GROUNDED` and
`LENGTH_SATURATION` govern which CHUNK comes back once a scope is chosen, and
nothing measures that.

The obvious known-item test -- build a query from a chunk's own words and check
the chunk is returned -- is unusable here. It rewards lexical overlap by
construction, so fitting fusion weights on it would drive `EMBED_WEIGHT` to zero
and the result would be an artefact of the metric rather than a property of the
corpus.

Instead the query is a chunk's own **heading**, and the heading is removed from
what is indexed. A heading is written by a human as a compressed description of
the section below it -- "Configure each platform you target", "Why this is not a
vibe-coded project" -- so it overlaps its body partly in wording and partly only
in meaning. That is the mix a real question has.

The bias is then measured rather than assumed: every query is also scored under
each ranker alone, and the report says how many items only one of them can find.
If lexical rankers alone solve everything, the metric is still lexical and the
weights fitted on it should not be trusted.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

# Headings that describe nothing in particular. A query of "Usage" cannot
# identify one section among many, and scoring it measures noise.
GENERIC = {"usage", "install", "installation", "license", "contributing",
           "overview", "introduction", "getting started", "api", "examples",
           "example", "notes", "todo", "changelog", "faq", "credits", "authors",
           "requirements", "setup", "options", "commands", "reference", "index"}
MIN_HEADING_TOKENS = 3
MIN_BODY_CHARS = 200
MAX_QUERIES_PER_SCOPE = 40


@dataclass
class Result:
    n: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    only_lexical: int = 0
    only_semantic: int = 0
    both: int = 0
    neither: int = 0
    detail: list = field(default_factory=list)


def _heading_leaf(heading: str) -> str:
    return heading.split(">")[-1].strip()


def build_queries(chunks) -> list[tuple[str, int]]:
    """(heading query, index of the chunk it belongs to)."""
    from loci.text import tokens

    out = []
    seen: set[str] = set()
    for i, c in enumerate(chunks):
        leaf = _heading_leaf(c.heading)
        if not leaf or len(c.text) < MIN_BODY_CHARS:
            continue
        norm = re.sub(r"[^a-z0-9 ]", " ", leaf.lower()).strip()
        if norm in GENERIC or norm in seen or len(tokens(leaf)) < MIN_HEADING_TOKENS:
            continue
        seen.add(norm)
        out.append((leaf, i))
    return out[:MAX_QUERIES_PER_SCOPE]


def _blind(chunks):
    """The same chunks with headings removed, so a query cannot match itself."""
    from loci.types import Chunk

    return [Chunk(kind=c.kind, source=c.source, heading="", text=c.text, ts=c.ts)
            for c in chunks]


def _encode_blind(blind, key: str) -> bool:
    """Encode heading-free chunks under `key`, so the semantic side is blind too.

    The shipped vectors were built over `heading + text`. Reusing them would let
    the semantic ranker match a query against its own heading and the whole
    design would collapse -- the lexical side would be blind and the semantic
    side would not, which is worse than no blinding at all.
    """
    import warnings

    from loci.backends import episodes as ep

    emb = ep._embeddings()
    if not emb or "_model" not in emb:
        return False
    try:
        warnings.filterwarnings("ignore")
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return False
    model = SentenceTransformer(str(emb["_model"][0]))
    vecs = model.encode([c.text for c in blind], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
    ep._EMB[key] = np.asarray(vecs, dtype="float32")
    return True


def prepare(store: dict, scope_ids: list[str]) -> list[dict]:
    """Build blind chunks, queries and vectors ONCE.

    Encoding is by far the dominant cost, and a naive sweep re-encoded every
    scope for every weight setting -- ten grid points did not finish in nine
    minutes. Everything that does not depend on the weights is hoisted here.
    """
    from loci.index import chunks_for

    prepared = []
    for sid in scope_ids:
        chunks = chunks_for(store, sid)
        queries = build_queries(chunks)
        if not queries:
            continue
        blind = _blind(chunks)
        key = f"blind::{sid}"
        prepared.append({"sid": sid, "blind": blind, "queries": queries,
                         "key": key, "semantic": _encode_blind(blind, key)})
    return prepared


def score(prepared: list[dict], *, weights=None, k: int = 10,
          variants: bool = False) -> Result:
    """Score one weight setting against pre-built corpora."""
    from loci.backends import episodes as ep
    from loci.backends.episodes import BuiltinEpisodeBackend

    backend = BuiltinEpisodeBackend()
    res = Result()
    saved = (ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT)
    if weights:
        ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT = weights
    try:
        for item in prepared:
            blind, key = item["blind"], item["key"]
            combos = [("fused", None)]
            if variants:
                combos.append(("lexical", (0.64, 0.36, 0.0)))
                if item["semantic"]:
                    combos.append(("semantic", (0.0, 0.0, 1.0)))
            for q, gold in item["queries"]:
                ranks = {}
                for label, w in combos:
                    prev = (ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT)
                    if w:
                        ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT = w
                    # Do NOT drop the fit here. BM25 and the TF-IDF matrix do
                    # not depend on the weights -- only the score combination
                    # does -- so refitting per query turned a sweep into an
                    # hours-long job that never finished.
                    hits = backend.search(q, blind, key, k=k, score_floor=-1.0,
                                          min_grounded=0, min_grounded_frac=0.0,
                                          semantic_floor=-1.0)
                    ranks[label] = next((i for i, h in enumerate(hits, 1)
                                         if h.chunk.text == blind[gold].text), None)
                    ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT = prev
                r = ranks["fused"]
                res.n += 1
                res.recall_at_1 += 1 if r == 1 else 0
                res.recall_at_5 += 1 if r and r <= 5 else 0
                res.mrr += (1.0 / r) if r else 0.0
                if variants:
                    lex = ranks.get("lexical") is not None and ranks["lexical"] <= 5
                    sem = ranks.get("semantic") is not None and ranks["semantic"] <= 5
                    if lex and sem:
                        res.both += 1
                    elif lex:
                        res.only_lexical += 1
                    elif sem:
                        res.only_semantic += 1
                    else:
                        res.neither += 1
                res.detail.append((q, r, ranks.get("lexical"), ranks.get("semantic")))
    finally:
        ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT = saved
    if res.n:
        res.recall_at_1 /= res.n
        res.recall_at_5 /= res.n
        res.mrr /= res.n
    return res


def evaluate(store: dict, scope_ids: list[str], *, weights=None, k: int = 10) -> Result:
    """Convenience wrapper: prepare then score once."""
    return score(prepare(store, scope_ids), weights=weights, k=k, variants=True)


def render(res: Result) -> str:
    if not res.n:
        return "no usable heading queries in this corpus"
    solvable = res.n - res.neither
    lines = [
        f"{res.n} heading queries over bodies that never contain them",
        f"  recall@1   {res.recall_at_1:.1%}",
        f"  recall@5   {res.recall_at_5:.1%}",
        f"  MRR        {res.mrr:.3f}",
        "",
        "Is this metric secretly lexical? Of the ones any ranker can solve:",
        f"  both rankers find it      {res.both:>4}  ({res.both / max(1, solvable):.0%})",
        f"  only lexical finds it     {res.only_lexical:>4}  "
        f"({res.only_lexical / max(1, solvable):.0%})",
        f"  only semantic finds it    {res.only_semantic:>4}  "
        f"({res.only_semantic / max(1, solvable):.0%})",
        f"  neither                   {res.neither:>4}",
    ]
    if res.only_semantic == 0 and res.n > 10:
        lines.append("\n! No query needs the semantic ranker. Weights fitted here would "
                     "be\n  an artefact of the metric; treat EMBED_WEIGHT as unfitted.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", help="LOCI_HOME of an already-built index")
    ap.add_argument("--weights", help="bm25,char,embed to score with")
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()
    if args.home:
        os.environ["LOCI_HOME"] = args.home

    from loci.index import load_episodes, load_index

    index, store = load_index(), load_episodes()
    weights = tuple(float(x) for x in args.weights.split(",")) if args.weights else None
    prepared = prepare(store, list(index["scopes"]))
    res = score(prepared, weights=weights, variants=True)
    print(render(res))
    if args.misses:
        print("\nunfound (fused rank, lexical, semantic):")
        for q, r, lx, sm in res.detail:
            if r is None or r > 5:
                print(f"  {str(r):>5} {str(lx):>5} {str(sm):>5}  {q[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
