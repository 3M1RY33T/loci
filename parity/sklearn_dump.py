"""Dump what sklearn's TfidfVectorizer actually chose, per scope.

Rust must reproduce the char_wb 3-5 gram vocabulary and the IDF vector, and one
part of sklearn's choice is not reproducible by construction:
`_limit_features` ranks by corpus term frequency with `(-tfs[mask]).argsort()`,
and numpy's default introsort is not stable, so terms tied at the 60,000th
position are ordered by sort internals.

That only bites a scope whose vocabulary EXCEEDS the cap. Measured on this
corpus, 12 of 15 never reach it -- for those `_limit_features` returns before
ranking anything and the port is exact. This writes the evidence down so the
Rust side is graded against a number rather than a hope.

Run BEFORE the .lex store replaces the joblib rankers:

    . parity/env.sh && python parity/sklearn_dump.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

MAX_FEATURES = 60000


_COUNT_VOCAB: dict[str, int] = {}


def _raw_counts(scope_id: str):
    """The count matrix sklearn ranks, refitted with max_features removed.

    Identical CountVectorizer parameters minus the cap, so the corpus term
    frequencies are the ones `_limit_features` would have sorted. Slow -- it
    re-extracts char 3-5 grams over the whole scope -- and run only for the
    scopes that actually reach the cap.
    """
    global _COUNT_VOCAB
    from sklearn.feature_extraction.text import CountVectorizer

    from loci.index import chunks_for, load_episodes

    store = load_episodes()
    chunks = chunks_for(store, scope_id)
    texts = [f"{c.heading} {c.text}" for c in chunks]
    cv = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                         lowercase=True)
    X = cv.fit_transform(texts)
    _COUNT_VOCAB = {t: int(i) for t, i in cv.vocabulary_.items()}
    return X


def dump(rankers: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(rankers.glob("*.joblib")):
        blob = joblib.load(f)
        vec, mat = blob["vec"], blob["mat"]
        if vec is None:
            print(f"  {f.stem}: no vectorizer, skipped")
            continue
        vocab = {t: int(i) for t, i in vec.vocabulary_.items()}
        capped = len(vocab) >= MAX_FEATURES
        tie_terms: list[str] = []
        cut = None
        if capped:
            # Terms sharing the frequency at the cut are the ones whose
            # retention an unstable sort decided. Everything above the cut is
            # retained regardless of order, everything below is dropped
            # regardless -- only the tie is ambiguous.
            #
            # Recomputed from RAW COUNTS, not from the stored matrix.
            # `_limit_features` runs inside fit_transform on the count matrix,
            # before any tf-idf weighting, so ranking the weighted matrix
            # identifies terms tied in tf-idf MASS -- a different set, and the
            # wrong oracle to grade a port against.
            counts = _raw_counts(f.stem)
            tfs = np.asarray(counts.sum(axis=0)).ravel()
            cut = float(np.sort(tfs)[::-1][MAX_FEATURES - 1])
            inv = {i: t for t, i in _COUNT_VOCAB.items()}
            tie_terms = sorted(inv[i] for i in np.where(tfs == cut)[0])
        rec = {
            "scope": f.stem,
            "n_docs": int(blob["n"]),
            "n_vocab": len(vocab),
            "capped": capped,
            "cut_frequency": cut if capped else None,
            "tie_terms": tie_terms,
            "idf": [float(x) for x in vec.idf_],
            "vocabulary": vocab,
        }
        (out / f"{f.stem}.json").write_text(json.dumps(rec), encoding="utf-8")
        summary.append((f.stem, int(blob["n"]), len(vocab), capped, len(tie_terms)))

    print(f"\n{'scope':<24}{'docs':>8}{'vocab':>9}{'capped':>9}{'ties':>7}")
    print("-" * 57)
    for name, n, v, capped, ties in summary:
        print(f"{name:<24}{n:>8}{v:>9}{str(capped):>9}{ties:>7}")
    print("-" * 57)
    total = sum(t for *_, t in summary)
    n_capped = sum(1 for *_, c, _ in summary if c)
    print(f"{n_capped} of {len(summary)} scopes reach the {MAX_FEATURES:,} cap")
    print(f"ambiguous columns across the whole corpus: {total}")
    print(f"as a fraction of one capped vocabulary: {total / MAX_FEATURES:.6%}")
    print()
    print("Rust must reproduce every vocabulary exactly where capped=False.")
    print("Where capped=True, a difference is acceptable ONLY inside tie_terms.")


if __name__ == "__main__":
    if not os.environ.get("LOCI_HOME"):
        sys.exit("LOCI_HOME is unset. Run `. parity/env.sh` first.")
    dump(Path(os.environ["LOCI_HOME"]) / "rankers",
         Path(__file__).resolve().parent / "corpus" / "sklearn")
