#!/usr/bin/env python3
"""How much of the remaining routing error a clarifying question could recover.

This measures the SHORTLIST an abstention hands back, not routing accuracy.
`loci eval` already scores which scope wins; nothing here can move those
numbers, because the candidate list is read only once routing has given up.

Three questions, in the order they decide anything:

    A  is the shortlist worth reading?     |candidates| and whether gold is on it
    B  would a splitting question help?    where gold SITS on the shortlist
    C  is there error outside abstention?  wrong-but-confident routes

The B answer is the one that costs money to get wrong. A shortlist whose first
entry is already the gold scope needs no question asked about it -- the caller
takes the top and is right. A splitting question only earns its round trip on
the cases where gold is present and NOT first, so that bucket is the ceiling on
what B can buy, before any question is designed.

Run it against your own corpus:

    python evals/clarify.py            # report
    python evals/clarify.py --json     # the same numbers, machine-readable
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from loci.eval import NONSENSE, TAXONOMY   # noqa: E402
from loci.index import load_index          # noqa: E402
from loci.router import route              # noqa: E402
from loci.text import unique_tokens        # noqa: E402

# How far to ablate a routable question before giving up on making it abstain.
# Each step deletes the winning scope's highest-contributing token, which is the
# same question asked less precisely -- up to a point. Past three steps the
# remainder stops being a question anyone would type ("projects handle files"),
# and a shortlist filter tuned to recover THOSE is tuned to noise.
ABLATION_STEPS = 3
MIN_ABLATED_TOKENS = 3


def _negative(item: dict) -> bool:
    """Items whose correct answer is to abstain, so they have no gold scope."""
    return item["family"] == "negative" or item["id"].startswith("enum-neg")


def build_cases(index: dict) -> tuple[list[dict], list[dict]]:
    """(gold-bearing cases, should-abstain cases) from the hand-authored set.

    The hand-authored questions are the held-out set -- they were written
    against the corpus, not generated from its index, so they are the only ones
    that can say whether a shortlist is any good. There are not many of them,
    and the ones that ABSTAIN are fewer still, which is why the ablations below
    exist and why they are labelled separately in the report.
    """
    scopes = index["scopes"]
    hand = json.loads((HERE / "questions.json").read_text())["questions"]
    gold, negative = [], []

    for it in hand:
        cwd = scopes[it["cwd"]]["root"] if it.get("cwd") else None
        if _negative(it):
            negative.append({"id": it["id"], "q": it["question"], "cwd": cwd})
            continue
        base = {"id": it["id"], "q": it["question"], "cwd": cwd,
                "gold": set(it["gold"]), "ablated": 0}
        gold.append(base)
        r = route(it["question"], index, cwd=cwd)
        if r.abstain:
            continue
        # Weaken it until it stops routing, so the shortlist is measured on more
        # than the three real questions that abstain outright. An ablated item is
        # a WEAKER PHRASING of a question whose answer is known, not a new one.
        cur = it["question"]
        for step in range(1, ABLATION_STEPS + 1):
            rr = route(cur, index, cwd=cwd)
            if rr.abstain:
                gold.append({**base, "id": f"{it['id']}~{step - 1}",
                             "q": cur, "ablated": step - 1})
                break
            drop = {t for t, _ in rr.detail[rr.ranked[0]]["top_tokens"][:1]}
            keep = [t for t in unique_tokens(cur) if t not in drop]
            if len(keep) == len(unique_tokens(cur)) or len(keep) < MIN_ABLATED_TOKENS:
                break
            cur = " ".join(keep)

    # `loci eval`'s own two abstain-by-construction families, on top of the
    # hand-authored negatives. Eight questions is too few to say anything about
    # how wide a shortlist gets on a question that has no answer, and these are
    # already shipped, already labelled, and cost nothing to reuse.
    negative += [{"id": f"nonsense:{i}", "q": q, "cwd": None}
                 for i, q in enumerate(NONSENSE)]
    negative += [{"id": f"deictic:{i}", "q": q, "cwd": None}
                 for i, q in enumerate(TAXONOMY)]
    return gold, negative


def measure(index: dict) -> dict:
    gold_cases, neg_cases = build_cases(index)
    S = len(index["scopes"])
    buckets = {"routed_right": [], "routed_wrong_reachable": [],
               "routed_wrong_lost": [], "abstained_reachable": [],
               "abstained_lost": []}
    before_size, after_size, ranks = [], [], []

    for c in gold_cases:
        r = route(c["q"], index, cwd=c["cwd"])
        cand = r.candidates
        on_list = bool(c["gold"] & set(cand))
        if not r.abstain:
            if c["gold"] & set(r.selected):
                buckets["routed_right"].append(c["id"])
            else:
                buckets["routed_wrong_reachable" if on_list
                        else "routed_wrong_lost"].append(c["id"])
            continue
        before_size.append(len(r.ranked))
        after_size.append(len(cand))
        if on_list:
            buckets["abstained_reachable"].append(c["id"])
            ranks.append(next(i + 1 for i, s in enumerate(cand) if s in c["gold"]))
        else:
            buckets["abstained_lost"].append(c["id"])

    neg_before, neg_after, neg_empty = [], [], 0
    for c in neg_cases:
        r = route(c["q"], index, cwd=c["cwd"])
        if not r.abstain:
            continue
        neg_before.append(len(r.ranked))
        neg_after.append(len(r.candidates))
        neg_empty += not r.candidates

    n_ab = len(before_size)
    return {
        "n_scopes": S,
        "n_gold": len(gold_cases),
        "n_ablated": sum(1 for c in gold_cases if c["ablated"]),
        "n_abstained_gold": n_ab,
        "n_abstained_neg": len(neg_after),
        "size_gold": (st.mean(before_size or [0]), st.mean(after_size or [0])),
        "size_neg": (st.mean(neg_before or [0]), st.mean(neg_after or [0])),
        "recall": len(buckets["abstained_reachable"]) / n_ab if n_ab else 0.0,
        "gold_first": sum(r == 1 for r in ranks) / n_ab if n_ab else 0.0,
        "gold_top2": sum(r <= 2 for r in ranks) / n_ab if n_ab else 0.0,
        "needs_a_question": sum(r > 1 for r in ranks),
        "neg_empty": neg_empty / len(neg_after) if neg_after else 0.0,
        "buckets": {k: sorted(v) for k, v in buckets.items()},
    }


def render(m: dict) -> str:
    b = m["buckets"]
    real = m["n_gold"] - m["n_ablated"]
    out = [
        f"{m['n_scopes']} scopes | {real} hand-authored questions with a gold scope "
        f"+ {m['n_ablated']} ablated phrasings of them",
        f"{m['n_abstained_gold']} of those abstain | "
        f"{m['n_abstained_neg']} questions that SHOULD abstain",
        "",
        "A -- is the shortlist worth reading?",
        f"{'':<34}{'before':>9}{'after':>9}",
        f"  {'shortlist size, gold-bearing':<32}"
        f"{m['size_gold'][0]:>9.1f}{m['size_gold'][1]:>9.1f}",
        f"  {'shortlist size, should-abstain':<32}"
        f"{m['size_neg'][0]:>9.1f}{m['size_neg'][1]:>9.1f}",
        f"  {'gold is on it':<32}{'100.0%':>9}{m['recall']:>8.1%}",
        f"  {'empty on should-abstain':<32}{'0.0%':>9}{m['neg_empty']:>8.1%}",
        "",
        "  Before is `ranked`: every eligible scope, which is the whole registry.",
        "  It cannot lose the gold scope and cannot inform anyone either.",
        "",
        "B -- would a splitting question help?",
        f"  gold ranked first on the shortlist   {m['gold_first']:>6.1%}",
        f"  gold in the top two                  {m['gold_top2']:>6.1%}",
        f"  gold present but NOT first           {m['needs_a_question']:>4} "
        f"question(s)",
        "",
        "  Taking the first candidate is the baseline any generated question has",
        "  to beat, and it is only wrong on that last line. Build B when that",
        "  count is large enough to pay for a round trip; it is the whole ceiling.",
        "",
        "C -- is there error outside abstention?",
        f"  routed correctly                     {len(b['routed_right']):>4}",
        f"  routed WRONG, gold on the shortlist  {len(b['routed_wrong_reachable']):>4}"
        f"   <- only C reaches these",
        f"  routed WRONG, gold not on it         {len(b['routed_wrong_lost']):>4}",
        f"  abstained, gold on the shortlist     {len(b['abstained_reachable']):>4}",
        f"  abstained, gold not on it            {len(b['abstained_lost']):>4}"
        f"   <- coverage, not clarification",
    ]
    lost = b["abstained_lost"] + b["routed_wrong_lost"]
    if lost:
        out += ["", "  unreachable by any shortlist (run `loci doctor`):",
                "    " + ", ".join(lost)]
    return "\n".join(out)


def sweep(index: dict) -> str:
    """Reproduce the filter comparison recorded against CANDIDATE_SHARE.

    Every alternative here was a serious candidate for the shipped rule, and the
    table is the reason none of them is. It is computed from the postings rather
    than from `detail["claims"]`, so it measures the RULE and not the router's
    current implementation of it -- a sweep that reads the shipped filter's own
    output can only ever agree with itself.
    """
    postings, scopes = index["postings"], index["scopes"]
    S = len(scopes)
    gold_cases, neg_cases = build_cases(index)

    def matched(r, sid):
        return [t for t in r.query_tokens if sid in postings.get(t, {})]

    def under(r, cut):
        return [s for s in r.ranked
                if any(len(postings[t]) <= cut for t in matched(r, s))]

    rules = {
        "ranked (before)": lambda r: list(r.ranked),
        "holds any matched token": lambda r: [s for s in r.ranked if matched(r, s)],
        "held by <= S/2 (shipped)": lambda r: under(r, max(1, S // 2)),
        "held by <= S/3": lambda r: under(r, max(1, S // 3)),
        "held by <= 4 scopes": lambda r: under(r, 4),
        "held by <= 2 scopes": lambda r: under(r, 2),
    }

    gold_r = [(c, route(c["q"], index, cwd=c["cwd"])) for c in gold_cases]
    gold_r = [(c, r) for c, r in gold_r if r.abstain]
    neg_r = [r for r in (route(c["q"], index, cwd=c["cwd"]) for c in neg_cases)
             if r.abstain]

    out = [f"CANDIDATE_SHARE sweep | {len(gold_r)} gold-bearing abstentions, "
           f"{len(neg_r)} that should abstain", "",
           f"{'rule':<26}{'gold recall':>12}{'|cand| gold':>13}"
           f"{'|cand| neg':>12}{'neg empty':>11}", "-" * 74]
    for name, rule in rules.items():
        sets = [rule(r) for _, r in gold_r]
        neg = [rule(r) for r in neg_r]
        rec = sum(bool(c["gold"] & set(s)) for (c, _), s in zip(gold_r, sets))
        out.append(f"{name:<26}{rec / max(1, len(gold_r)):>11.1%}"
                   f"{st.mean([len(s) for s in sets] or [0]):>13.1f}"
                   f"{st.mean([len(s) for s in neg] or [0]):>12.1f}"
                   f"{sum(not s for s in neg) / max(1, len(neg)):>10.1%}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="compare shortlist filters instead of reporting A/B/C")
    args = ap.parse_args()
    index = load_index()
    if args.sweep:
        print(sweep(index))
        return 0
    m = measure(index)
    print(json.dumps(m, indent=2) if args.json else render(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
