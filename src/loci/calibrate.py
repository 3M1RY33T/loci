"""Fit the routing thresholds to the corpus in front of you.

Every threshold in `router.py` was swept against one person's ten repositories.
A user with Rust systems code, or a monorepo of near-identical services, or
documentation in a language other than English inherits numbers fitted to
somebody else's habits.

Calibration needs labelled questions, and labelling is exactly what a user will
not do. `loci.eval` solves that: its families have gold answers known by
construction -- a deictic question asked from inside a scope must route there,
the same question with no working directory must abstain, nonsense must abstain,
and a question built from a scope's own rarest words should reach it. That is a
labelled set nobody had to write.

**What the measurement says.** Four candidate statistics were compared on that
set, and the one already in use is the worst of them:

    matched token count   NOT separable -- inverted, in fact. Routable questions
                          matched a median of 2 tokens, unroutable ones a median
                          of 2 and a maximum of 4.
    max token evidence    separable: 4.31 vs 3.75
    TOTAL token evidence  separable, widest margin: 8.59 vs 7.12
    mean token evidence   NOT separable: 3.07 vs 3.45

So the count gate works on the corpus it was fitted to, and does so by accident.
Summed evidence is what actually distinguishes a question that identifies a
scope from one that cannot, and its threshold is corpus-dependent -- which is
precisely why it should be fitted per corpus rather than shipped as a constant.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .eval import (NONSENSE, SIGNATURE_TEMPLATES, TAXONOMY, contended_terms,
                   halve, signature_terms)
from .paths import atomic_write, home

CALIBRATION_VERSION = 4
# Weight on correctly refusing an unroutable question, relative to correctly
# routing one. Above 1.0 because the errors are not symmetric: a false route
# hands back a confident wrong project, a false abstention asks "which project
# did you mean".
ABSTAIN_WEIGHT = 1.25
MIN_SAMPLES = 6
# Routable samples are drawn across this range of term rarity. Using only a
# scope's rarest words produces a best-case band -- measured, rank 1-2 terms
# score from 8.59 while ordinary rank 7-10 terms score from 5.74, and real
# questions land at 6.5-8.8. A floor fitted to the best case sat at 7.49 and
# cut real questions in half: cross-scope routing fell from 100% to 33%.
TERM_RANKS = (0, 2, 4, 6, 8)


def calibration_file():
    return home() / "calibration.json"


@dataclass
class Calibration:
    evidence_floor: float
    semantic_floor: float               # fallback for a scope with no fit
    semantic_separation: float
    semantic_n: int
    semantic_by_scope: dict
    separation: float          # accuracy of the fitted threshold on its own samples
    n_route: int
    n_abstain: int
    route_min: float
    abstain_max: float
    trustworthy: bool

    def to_json(self) -> dict:
        return {"version": CALIBRATION_VERSION, **self.__dict__}


# Queries guaranteed to be unrelated to any software corpus. The floor they
# produce is this model's "no relationship" baseline, which is what the shipped
# 0.57 was silently assuming was the same everywhere.
UNRELATED = list(NONSENSE) + [
    "the emperor penguin incubates a single egg through the antarctic winter",
    "sourdough starter needs equal parts flour and water refreshed daily",
    "a sonnet has fourteen lines and a volta somewhere near the ninth",
    "monarch butterflies migrate several thousand kilometres each autumn",
    "the 1966 final went to extra time at wembley stadium",
]
SEMANTIC_MARGIN = 0.35     # where to sit between the two bands
MIN_SEMANTIC_SAMPLES = 8


def fit_semantic_floor(index: dict, store: dict) -> tuple[dict, float, int]:
    """Fit the cosine floor PER SCOPE, to this corpus and this embedding model.

    `SEMANTIC_FLOOR` was picked from one corpus with one model: real questions
    scored 0.61-0.80 and unrelated ones 0.46-0.52, so 0.57 sat in the gap. Cosine
    scale is a property of the model and the size of the gap is a property of the
    corpus, so a number from one pairing describes neither in general.

    Per scope, not pooled. The floor is compared against a MAXIMUM over a
    scope's chunks, and the expected maximum of an unrelated query grows with
    how many chunks you take it over -- a scope with 8,000 chunks throws up a
    spurious 0.60 that a scope with 119 never will. Measured, the bands separate
    cleanly inside each scope (0.78 vs 0.52 for one, 0.69 vs 0.55 for another)
    and collapse to nothing when pooled: separation -0.002 across 170 samples.

    Both bands are label-free. A section's own heading is related to its body by
    construction; the UNRELATED queries are related to no software corpus.
    """
    from .backends import episodes as ep
    from .index import chunks_for

    emb = ep._embeddings()
    if not emb or "_model" not in emb:
        return {}, 0.0, 0
    model_name = str(emb["_model"][0])

    floors: dict[str, float] = {}
    seps: list[float] = []
    total = 0
    for sid in index["scopes"]:
        vecs = emb.get(sid)
        chunks = chunks_for(store, sid)
        if vecs is None or len(chunks) != vecs.shape[0]:
            continue
        related: list[float] = []
        for c in chunks:
            leaf = c.heading.split(">")[-1].strip()
            if len(leaf.split()) >= 3 and len(c.text) >= 200:
                qv = ep._encode_query(leaf, model_name)
                if qv is not None:
                    related.append(float((vecs @ qv).max()))
            if len(related) >= 25:
                break
        unrelated = []
        for q in UNRELATED:
            qv = ep._encode_query(q, model_name)
            if qv is not None:
                unrelated.append(float((vecs @ qv).max()))
        if len(related) < MIN_SEMANTIC_SAMPLES or len(unrelated) < MIN_SEMANTIC_SAMPLES:
            continue
        related.sort()
        unrelated.sort()
        lo = related[len(related) // 10]          # 10th percentile of related
        hi = unrelated[-1 - len(unrelated) // 10]  # 90th percentile of unrelated
        sep = lo - hi
        floors[sid] = round((hi + SEMANTIC_MARGIN * sep) if sep > 0 else min(lo, hi), 3)
        seps.append(sep)
        total += len(related) + len(unrelated)

    mean_sep = round(sum(seps) / len(seps), 3) if seps else 0.0
    return floors, mean_sep, total


def _evidence_total(r, idf_max: float) -> float:
    if not r.ranked:
        return 0.0
    toks = [c for _, c in r.detail[r.ranked[0]]["top_tokens"]]
    return sum(toks) / idf_max if toks else 0.0


def collect_samples(index: dict) -> tuple[list[float], list[float]]:
    """Summed evidence for questions that should route, and that should not."""
    from .router import route

    idf_max = math.log(1 + len(index["scopes"])) or 1.0
    should_route: list[float] = []
    should_abstain: list[float] = []

    for sid in index["scopes"]:
        pool = signature_terms(index, sid, k=max(TERM_RANKS) + 2)
        for start in TERM_RANKS:
            terms = pool[start:start + 2]
            if len(terms) < 2:
                continue
            # The fit half only. `loci eval` reports on the other half, so a
            # perfect score straight after calibrating is not the fit reading
            # itself back.
            for tpl in (halve(SIGNATURE_TEMPLATES, "fit") or SIGNATURE_TEMPLATES[:1]):
                q = tpl.format(a=terms[0], b=terms[1])
                r = route(q, index)
                # Only count it when the evidence actually points at the right
                # scope; a signature question that lands elsewhere says the
                # corpus is ambiguous, not that this evidence level is routable.
                if r.ranked and r.ranked[0] == sid:
                    should_route.append(_evidence_total(r, idf_max))

    # Contended questions belong in the routable sample. They are structurally
    # lower-evidence than signature questions -- two words, one of them shared
    # by two scopes -- and fitting a floor without them puts it above the range
    # they occupy. Measured on a corpus of four real repositories: signature
    # questions scored from 7.76, contended ones around 6.0, and the fitted
    # floor of 8.69 refused every contended question in the set.
    for term, sids in contended_terms(index):
        r = route(f"how is {term} handled?", index)
        if r.ranked and r.ranked[0] in sids:
            should_route.append(_evidence_total(r, idf_max))

    for q in halve(TAXONOMY, "fit") + halve(NONSENSE, "fit"):
        should_abstain.append(_evidence_total(route(q, index), idf_max))

    return should_route, should_abstain


def fit(index: dict, store: dict | None = None) -> Calibration:
    if store is None:
        from .index import load_episodes
        store = load_episodes()
    sem_by_scope, sem_sep, sem_n = fit_semantic_floor(index, store)
    from .backends.episodes import SEMANTIC_FLOOR as _DEFAULT_SEM
    sem_floor = (round(sum(sem_by_scope.values()) / len(sem_by_scope), 3)
                 if sem_by_scope else _DEFAULT_SEM)
    route_s, abstain_s = collect_samples(index)
    if len(route_s) < MIN_SAMPLES or len(abstain_s) < MIN_SAMPLES:
        from .router import EVIDENCE_FLOOR
        return Calibration(EVIDENCE_FLOOR, sem_floor, sem_sep, sem_n,
                           sem_by_scope, 0.0, len(route_s), len(abstain_s),
                           0.0, 0.0, False)

    route_s.sort()
    abstain_s.sort()

    # Sweep, rather than picking an endpoint. The bands usually overlap on a
    # real corpus, and then no endpoint is defensible: the routable minimum
    # admits every unroutable question, and the unroutable maximum refuses half
    # the real ones. The threshold that classifies the most samples correctly is
    # the only choice that survives overlap.
    candidates = sorted({round(v, 3) for v in route_s + abstain_s})
    best_floor, best_score = candidates[0], -1.0
    for c in candidates:
        routed_ok = sum(1 for v in route_s if v >= c) / len(route_s)
        refused_ok = sum(1 for v in abstain_s if v < c) / len(abstain_s)
        score = routed_ok + ABSTAIN_WEIGHT * refused_ok
        # >= so ties resolve upward, toward abstaining
        if score >= best_score:
            best_score, best_floor = score, c

    route_min, abstain_max = route_s[0], abstain_s[-1]
    accuracy = ((sum(1 for v in route_s if v >= best_floor)
                 + sum(1 for v in abstain_s if v < best_floor))
                / (len(route_s) + len(abstain_s)))
    return Calibration(best_floor, sem_floor, sem_sep, sem_n, sem_by_scope,
                       round(accuracy, 3), len(route_s), len(abstain_s),
                       round(route_min, 3), round(abstain_max, 3),
                       accuracy >= 0.9)


def save(cal: Calibration) -> None:
    atomic_write(calibration_file(), json.dumps(cal.to_json(), indent=2))


def load() -> Calibration | None:
    f = calibration_file()
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("version") != CALIBRATION_VERSION:
        return None
    d.pop("version", None)
    try:
        return Calibration(**d)
    except TypeError:
        return None


def render(cal: Calibration) -> str:
    out = [f"evidence floor      {cal.evidence_floor}",
           f"semantic floor      {cal.semantic_floor}"
           + (f"   (mean per-scope, {len(cal.semantic_by_scope)} scopes fitted, "
              f"bands separate by {cal.semantic_separation:+.3f})"
              if cal.semantic_n else
              "   (default; no embeddings built)"),]
    out += [
           f"fitted from         {cal.n_route} routable + {cal.n_abstain} unroutable "
           f"questions, none hand-labelled",
           f"routable band       from {cal.route_min}",
           f"unroutable band     up to {cal.abstain_max}",
           f"classifies            {cal.separation:.1%} of its own samples correctly"]
    if cal.trustworthy:
        out.append("\nEvidence alone separates routable from unroutable questions on "
                   "this corpus.")
    else:
        out.append("\n! The bands OVERLAP -- some questions that should route score no "
                   "higher than\n  questions that should not, so no threshold separates "
                   "them cleanly. The fitted\n  value is the best available compromise; "
                   "expect more abstentions and lean on\n  cwd or --scope. This usually "
                   "means your projects share most of their\n  vocabulary.")
    return "\n".join(out)
