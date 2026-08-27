"""`loci eval` — measure routing on the user's own corpus, with no hand labels.

Every accuracy figure in this project's README was fitted and measured against
one person's ten repositories. That is a claim a user has to trust. This turns
it into something they can check on their own machine in a few seconds.

The trick is to only ask questions whose correct answer is known by
construction, so nobody has to label anything:

  deictic + cwd     "How do I run the tests?" asked from inside a scope. The
                    answer is that scope, definitionally. Measures whether the
                    strongest routing signal works at all.
  deictic, no cwd   the same questions with nothing to bind them to. The correct
                    answer is to abstain -- there is no way to know which
                    project "this" means.
  nonsense          questions no software corpus can answer. Correct answer is
                    abstain. Fixed list, independent of any corpus.
  signature         built from each scope's own most discriminative vocabulary,
                    excluding its name. Gold is that scope. This measures
                    whether the user's scopes are distinguishable FROM EACH
                    OTHER, which is the thing that actually varies between
                    corpora and the thing no amount of tuning on someone else's
                    repos can predict.

The signature family is derived from the index it is scoring, so it is a
discrimination test rather than a recall test, and it is reported as such. A low
score there means the user's projects genuinely look alike to a lexical router --
which is a fact worth knowing about the corpus, not a bug to tune away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .router import route
from .text import tokens as vtokens

# Deliberately deictic: every one points at its subject instead of naming it.
TAXONOMY = [
    "What does this project do and who is it for?",
    "How do I run the tests?",
    "How is this deployed and what runs it?",
    "What is the entry point and how does it start up?",
    "What external services or libraries does this depend on?",
    "What are the known risks, gaps or limitations here?",
    "How do I set up a development environment?",
    "What configuration or environment variables does it need?",
]

# No software corpus answers these. Any confident route is a false positive.
NONSENSE = [
    "What is the airspeed velocity of an unladen swallow?",
    "Describe the mating habits of the emperor penguin.",
    "What is the best recipe for sourdough starter?",
    "Who won the 1966 World Cup final?",
    "How do I fix this bug?",
    "What should I work on next?",
]

# Templates must contain no deictic marker. An earlier one read "where is {a}
# implemented and what does it do with {b}?" -- the "it" tripped the router's
# deixis rule, which correctly abstains on a question that points at a subject
# instead of naming it. The generator was scoring the router for behaving as
# designed, and it dragged the signature family from 90% to 45%. A benchmark
# that penalises correct behaviour is worse than no benchmark, so this is
# enforced by a test rather than trusted to review.
SIGNATURE_TEMPLATES = [
    "how is {a} handled together with {b}?",
    "what happens to {a} during {b} processing?",
    "which component connects {a} and {b}?",
]


@dataclass
class Family:
    name: str
    n: int = 0
    correct: int = 0
    abstained: int = 0
    set_total: int = 0
    detail: str = ""
    misses: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.correct / self.n if self.n else 0.0


def signature_terms(index: dict, scope_id: str, k: int = 2) -> list[str]:
    """A scope's most discriminative tokens, excluding anything naming it.

    Uses the router's own evidence measure, so the questions probe exactly the
    signal routing depends on rather than a proxy for it.
    """
    meta = index["scopes"][scope_id]
    postings = index["postings"]
    S = len(index["scopes"])
    idf_max = math.log(1 + S) or 1.0
    n_nodes = max(1, meta.get("node_count", 1))
    banned = {t for a in meta.get("aliases", []) for t in vtokens(a)}
    banned |= set(vtokens(meta["name"]))

    scored = []
    for t, post in postings.items():
        if scope_id not in post or t in banned or len(t) < 4:
            continue
        idf = math.log(1 + S / max(1, len(post)))
        ev = idf * (1 + math.log1p(post[scope_id] / n_nodes * 1000)) / idf_max
        scored.append((ev, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:k]]


def run(index: dict, *, signature_per_scope: int = 2) -> dict:
    scopes = index["scopes"]
    fams: dict[str, Family] = {}

    cwd_f = Family("deictic + cwd", detail="should route to the scope you are in")
    for sid, meta in scopes.items():
        for q in TAXONOMY:
            r = route(q, index, cwd=meta["root"])
            cwd_f.n += 1
            if not r.abstain and r.ranked[0] == sid:
                cwd_f.correct += 1
            else:
                cwd_f.abstained += r.abstain
                cwd_f.misses.append(f"{meta['name']}: {q}")
    fams["cwd"] = cwd_f

    nocwd_f = Family("deictic, no cwd", detail="should abstain; nothing says which project")
    for q in TAXONOMY:
        r = route(q, index)
        nocwd_f.n += 1
        nocwd_f.correct += r.abstain
        nocwd_f.abstained += r.abstain
        if not r.abstain:
            nocwd_f.misses.append(f"routed to {scopes[r.ranked[0]]['name']}: {q}")
    fams["nocwd"] = nocwd_f

    neg_f = Family("unanswerable", detail="should abstain; no corpus can answer these")
    for q in NONSENSE:
        r = route(q, index)
        neg_f.n += 1
        neg_f.correct += r.abstain
        neg_f.abstained += r.abstain
        if not r.abstain:
            neg_f.misses.append(f"routed to {scopes[r.ranked[0]]['name']}: {q}")
    fams["nonsense"] = neg_f

    sig_f = Family("signature", detail="are your scopes distinguishable from each other?")
    for sid, meta in scopes.items():
        terms = signature_terms(index, sid, k=2)
        if len(terms) < 2:
            continue
        for tpl in SIGNATURE_TEMPLATES[:signature_per_scope]:
            q = tpl.format(a=terms[0], b=terms[1])
            r = route(q, index)
            sig_f.n += 1
            sig_f.set_total += len(r.selected)
            if not r.abstain and r.ranked[0] == sid:
                sig_f.correct += 1
            else:
                got = "abstained" if r.abstain else scopes[r.ranked[0]]["name"]
                sig_f.abstained += r.abstain
                sig_f.misses.append(f"{meta['name']} -> {got}: {q}")
    fams["signature"] = sig_f

    return {"families": fams, "n_scopes": len(scopes),
            "chance": 1.0 / len(scopes) if scopes else 0.0}


def render(result: dict, *, show_misses: bool = False) -> str:
    fams: dict[str, Family] = result["families"]
    n = result["n_scopes"]
    out = [f"{n} scopes | random guessing would score {result['chance']:.1%}", ""]
    out.append(f"{'family':<20} {'n':>4} {'correct':>8} {'scopes':>7}  what it measures")
    out.append("-" * 84)
    for f in fams.values():
        avg = f"{f.set_total / f.n:.1f}" if f.set_total and f.n else "-"
        out.append(f"{f.name:<20} {f.n:>4} {f.rate:>7.1%} {avg:>7}  {f.detail}")
    out.append("")
    out.append("signature is an UPPER BOUND: it asks each scope about its own most "
               "distinctive\nwords, so a real question phrased in shared vocabulary "
               "will do worse than this.")
    out.append("")

    cwd, nocwd, neg, sig = (fams["cwd"], fams["nocwd"], fams["nonsense"], fams["signature"])
    verdicts = []
    if cwd.rate < 0.95:
        verdicts.append("! cwd routing is unreliable here -- that is the signal everything "
                        "else leans on. Check that your scope roots are correct.")
    if nocwd.rate < 0.75 or neg.rate < 0.6:
        verdicts.append("! the router answers questions it cannot know the answer to. "
                        "Expect confident wrong scopes; raise MIN_MATCHED.")
    if sig.n and sig.rate < 0.7:
        worst = sig.set_total / sig.n if sig.n else 0
        verdicts.append(f"! your scopes are hard to tell apart (avg {worst:.1f} returned). "
                        f"Projects that share vocabulary need cwd or an explicit --scope.")
    if not verdicts:
        verdicts.append("routing looks healthy on this corpus.")
    out += verdicts

    if show_misses:
        for f in fams.values():
            if f.misses:
                out.append(f"\n{f.name} misses ({len(f.misses)}):")
                out += [f"  {m}" for m in f.misses[:10]]
                if len(f.misses) > 10:
                    out.append(f"  ... and {len(f.misses) - 10} more")
    return "\n".join(out)
