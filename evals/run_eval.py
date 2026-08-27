#!/usr/bin/env python3
"""Run the eval set and report by family.

Reports routing and retrieval SEPARATELY, and never averages across
contamination levels. An item authored from indexed prose can only demonstrate
ranking; an item authored from code bodies -- which neither store indexes --
is the one that demonstrates recall. Blending them into a single headline number
is the specific dishonesty this harness exists to avoid.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from taxonomy import generate as gen_taxonomy  # noqa: E402

from loci.ask import ask  # noqa: E402
from loci.index import load_episodes, load_index  # noqa: E402
from loci.router import route  # noqa: E402


def load_items(index: dict, families: set[str] | None = None) -> list[dict]:
    hand = json.loads((HERE / "questions.json").read_text())["questions"]
    items = gen_taxonomy(sorted(index["scopes"])) + hand
    unknown = {g for it in items for g in it["gold"] if g not in index["scopes"]}
    if unknown:
        raise SystemExit(f"eval set references unknown scopes: {sorted(unknown)}")
    if families:
        items = [it for it in items if it["family"] in families]
    return items


def score_routing(items: list[dict], index: dict, *, use_cwd: bool) -> list[dict]:
    rows = []
    for it in items:
        cwd = None
        if use_cwd and it.get("cwd"):
            cwd = index["scopes"][it["cwd"]]["root"]
        r = route(it["question"], index, cwd=cwd)
        gold = set(it["gold"])
        top = r.ranked[0] if r.ranked else None
        sel = set(r.selected)

        if not gold:                       # negatives: abstaining IS the answer
            ok_top = ok_widen = r.abstain
            cov = 1.0 if r.abstain else 0.0
        else:
            ok_top = (not r.abstain) and top in gold
            ok_widen = (not r.abstain) and bool(gold & sel)
            cov = len(gold & sel) / len(gold)
        rows.append({**it, "top": top, "selected": r.selected, "abstain": r.abstain,
                     "ok_top": ok_top, "ok_widen": ok_widen, "gold_cov": cov,
                     "n_sel": len(r.selected), "matched": r.top_matched,
                     "evidence": max((d.get("evidence", 0) for d in r.detail.values()),
                                     default=0.0)})
    return rows


def score_retrieval(items: list[dict], index: dict, store: dict, *,
                    rerank: bool) -> list[dict]:
    rows = []
    for it in items:
        if not it["gold"]:
            continue
        gold = it["gold"][0]
        cwd = index["scopes"][it["cwd"]]["root"] if it.get("cwd") else None
        a = ask(it["question"], cwd=cwd, force_scopes=[gold], episodes_k=3,
                with_structure=False, rerank=rerank, index=index, store=store)
        hits = a.scopes[0].episodes if a.scopes else []
        wants = [w.lower() for w in (it.get("expect_contains") or [])]
        rank = None
        if wants:
            rank = next((i for i, h in enumerate(hits, 1)
                         if any(w in f"{h.chunk.heading} {h.chunk.text}".lower()
                                for w in wants)), None)
        rows.append({**it, "n_hits": len(hits), "rank": rank,
                     "top": hits[0].chunk.source if hits else None,
                     "top_score": hits[0].score if hits else 0.0})
    return rows


def score_structure(items: list[dict], index: dict) -> list[dict]:
    """Score the STRUCTURE store: does the traversal surface the answering symbol?

    Scope is forced to gold and episodes are off, so this isolates graphify. On a
    machine with no graphs every item scores zero by construction -- which is the
    point: it is the only part of this eval that can see what graphify adds.
    """
    from loci.ask import ask
    rows = []
    for it in items:
        gold = it["gold"][0]
        if not index["scopes"].get(gold, {}).get("sources"):
            rows.append({**it, "found": False, "reason": "no structure graph"})
            continue
        a = ask(it["question"], force_scopes=[gold], with_episodes=False,
                budget=1200, index=index, store={})
        txt = " ".join(h.text for h in a.scopes[0].structure) if a.scopes else ""
        want = it.get("expect_symbol", "")
        rows.append({**it, "found": bool(want and want in txt),
                     "reason": "" if txt else "empty traversal"})
    return rows


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def table(rows: list[dict], key: str, title: str) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = [f"\n{title}",
           f"{key:<14} {'n':>4} {'top-1':>7} {'hit@widen':>10} {'goldcov':>8} {'set':>5}",
           "-" * 52]
    for g in sorted(groups):
        sub = groups[g]
        n = len(sub)
        out.append(f"{g:<14} {n:>4} "
                   f"{_pct(sum(r['ok_top'] for r in sub)/n):>7} "
                   f"{_pct(sum(r['ok_widen'] for r in sub)/n):>10} "
                   f"{_pct(sum(r['gold_cov'] for r in sub)/n):>8} "
                   f"{sum(r['n_sel'] for r in sub)/n:>5.2f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", action="append")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--retrieval", action="store_true", help="also score retrieval")
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    index = load_index()
    items = load_items(index, set(args.family) if args.family else None)
    print(f"{len(index['scopes'])} scopes | {len(items)} items "
          f"({sum(1 for i in items if i['family'] == 'taxonomy')} generated, "
          f"{sum(1 for i in items if i['family'] != 'taxonomy')} hand-authored)")

    deictic = [i for i in items if i["family"] == "taxonomy"]
    structure = [i for i in items if i["family"] == "structure"]
    named = [i for i in items if i["family"] not in ("taxonomy", "structure")]

    with_cwd = score_routing(deictic, index, use_cwd=True) if deictic else []
    no_cwd = score_routing(deictic, index, use_cwd=False) if deictic else []
    other = score_routing(named, index, use_cwd=True) if named else []

    print("\n=== ROUTING ===")
    if with_cwd:
        n = len(with_cwd)
        print(f"\ntaxonomy (deictic, {n} items)")
        print(f"  with cwd     top-1 {_pct(sum(r['ok_top'] for r in with_cwd)/n)}   "
              f"abstained {_pct(sum(r['abstain'] for r in with_cwd)/n)}")
        print(f"  without cwd  top-1 {_pct(sum(r['ok_top'] for r in no_cwd)/n)}   "
              f"abstained {_pct(sum(r['abstain'] for r in no_cwd)/n)}"
              "   <- abstaining is CORRECT here")
    if other:
        print(table(other, "family", "hand-authored families"))
        print(table([r for r in other if r["gold"]], "contamination",
                    "hand-authored, by what the author had seen"))

    if args.misses:
        print("\nrouting misses:")
        names = {s: m["name"] for s, m in index["scopes"].items()}
        for r in other + with_cwd:
            if r["ok_top"]:
                continue
            got = "ABSTAIN" if r["abstain"] else names.get(r["top"], r["top"])
            want = ", ".join(names.get(g, g) for g in r["gold"]) or "(abstain)"
            print(f"  [{r['id']:<28}] want {want:<24} got {got:<22} "
                  f"matched={r['matched']} ev={r['evidence']:.2f}")

    if structure:
        srows = score_structure(structure, index)
        hits = sum(r["found"] for r in srows)
        print(f"\n=== STRUCTURE STORE (scope forced, episodes off) ===")
        print(f"  answering symbol surfaced   {hits}/{len(srows)}")
        for r in srows:
            mark = "OK" if r["found"] else "--"
            note = f"  ({r['reason']})" if r["reason"] else ""
            print(f"  {mark} [{r['id']:<20}] want {r['expect_symbol']}{note}")

    if args.retrieval:
        store = load_episodes()
        rr = score_retrieval(named, index, store, rerank=args.rerank)
        print(f"\n=== RETRIEVAL (scope forced to gold; rerank={args.rerank}) ===")
        checked = [r for r in rr if r.get("expect_contains")]
        clean = [r for r in checked if r["contamination"] == "none"]
        print(f"  evidence returned      {sum(1 for r in rr if r['n_hits'])}/{len(rr)}")
        if checked:
            print(f"  justifying file @1     {sum(1 for r in checked if r['rank'] == 1)}/{len(checked)}")
            print(f"  justifying file @3     {sum(1 for r in checked if r['rank'])}/{len(checked)}")
        if clean:
            print(f"  ...of which uncontaminated: "
                  f"@1 {sum(1 for r in clean if r['rank'] == 1)}/{len(clean)}  "
                  f"@3 {sum(1 for r in clean if r['rank'])}/{len(clean)}")
        for r in rr:
            mark = "  " if r["n_hits"] else "XX"
            print(f"  {mark} [{r['id']:<26}] {r['contamination']:<6} "
                  f"rank={str(r['rank'] or '-'):<3} top={str(r['top'])[:38]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
