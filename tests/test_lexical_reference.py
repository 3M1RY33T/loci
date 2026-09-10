"""Comparisons against scikit-learn and rank_bm25, in a FILE of their own.

Not merely a marker. `pytest.importorskip` runs when the module is imported,
and pytest imports a module to collect it even when every test inside is
deselected -- so `-m "not lexical_reference"` still pulled scikit-learn and
scipy into the process, and the abort this separation exists to prevent went on
happening at about one run in eight.

onnxruntime and scipy/sklearn each carry a native threading runtime, and a
process holding both aborts at interpreter exit:

    libc++abi: terminating due to uncaught exception of type
    std::__1::system_error: recursive_mutex lock failed: Invalid argument

with SIGABRT, which is indistinguishable from a crash to anything reading the
exit code. Measured: ORT alone 0/8, neither 0/15, both 2/15.

Nothing under src/ imports scikit-learn any more, so no user install and no CI
job holds both. This is a property of the test suite alone, and the suite runs
in two passes:

    pytest tests --ignore=tests/test_lexical_reference.py
    pytest tests/test_lexical_reference.py
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from loci import _core

HOME = Path(os.environ["LOCI_HOME"]) if os.environ.get("LOCI_HOME") else None
needs_corpus = pytest.mark.skipif(
    HOME is None or not (HOME / "episodes.json").is_file(),
    reason="needs the frozen corpus; run `. parity/env.sh` first")

# -- lexical ---------------------------------------------------------------
# scikit-learn and rank_bm25 are DEV dependencies now: nothing under src/
# imports them any more. They stay as the reference these tests compare
# against, exactly as sentence-transformers does for the encoder.
sk = pytest.importorskip("sklearn", reason="reference for the char-gram matrix")
rb = pytest.importorskip("rank_bm25", reason="reference for BM25")

# Run these in their OWN process: `pytest -m lexical_reference`.
#
# onnxruntime and scipy/sklearn each bring a native threading runtime, and a
# process holding both aborted at interpreter exit -- SIGABRT, indistinguishable
# from a crash to anything reading the exit code -- roughly once in five runs.
# Releasing ORT sessions at exit cut it to about one in twenty-five; it did not
# remove it, and a probabilistic fix is not a fix for a gate.
#
# Separating them removes the collision by construction rather than by odds.
# Nothing under src/ imports sklearn any more, so no user process ever holds
# both; this is a property of the test suite alone.
pytestmark = pytest.mark.lexical_reference

LEX_SCOPES = ["zim-compress", "beacon", "urthreads", "loci"]


def _fit_both(scope):
    """(rust Lex, rank_bm25 model, sklearn vectorizer, sklearn matrix)."""
    import tempfile

    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer

    from loci.index import chunks_for, load_episodes
    from loci.text import tokens as vtokens

    chunks = chunks_for(load_episodes(), scope)
    texts = [f"{c.heading} {c.text}" for c in chunks]
    toks = [vtokens(t) for t in texts]

    with tempfile.NamedTemporaryFile(suffix=".lex", delete=False) as f:
        path = f.name
    _core.lex_fit_write(path, texts, toks)
    lex = _core.Lex.open(path, len(chunks))

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                          lowercase=True)
    mat = vec.fit_transform(texts)
    return lex, BM25Okapi(toks), vec, mat, path


@needs_corpus
@pytest.mark.parametrize("scope", LEX_SCOPES)
def test_bm25_is_bit_identical_to_rank_bm25(scope):
    import numpy as np

    from loci.text import tokens as vtokens

    lex, bm, _, _, path = _fit_both(scope)
    questions = [q.text for q in _reference_questions()]
    worst = 0.0
    for q in questions:
        want = np.asarray(bm.get_scores(vtokens(q)), dtype=float)
        got = np.asarray(lex.bm25_scores(vtokens(q)), dtype=float)
        assert got.shape == want.shape
        worst = max(worst, float(np.abs(want - got).max()) if want.size else 0.0)
    os.unlink(path)
    # Not "close": identical. The formula is deterministic and the port
    # reproduces the epsilon floor over raw idfs, so anything above 0 means a
    # real difference rather than float noise.
    assert worst == 0.0, f"{scope}: max BM25 difference {worst}"


@needs_corpus
@pytest.mark.parametrize("scope", LEX_SCOPES)
def test_char_tfidf_matches_sklearn_and_preserves_order(scope):
    import numpy as np
    from sklearn.metrics.pairwise import linear_kernel

    lex, _, vec, mat, path = _fit_both(scope)
    questions = [q.text for q in _reference_questions()]
    worst = 0.0
    for q in questions:
        want = linear_kernel(vec.transform([q]), mat).ravel()
        got = np.asarray(lex.char_scores(q), dtype=float)
        worst = max(worst, float(np.abs(want - got).max()))
        assert np.array_equal(np.argsort(-want)[:5], np.argsort(-got)[:5]), \
            f"{scope}: top-5 reordered for {q!r}"
    os.unlink(path)
    # f64 epsilon, not tolerance: ~1e-15 against a 1e-6 parity budget.
    assert worst < 1e-12, f"{scope}: max char-score difference {worst}"


@needs_corpus
def test_the_uncapped_vocabulary_matches_sklearn_term_for_term():
    """With max_features removed there is nothing arbitrary left: min_df=1
    keeps every n-gram, so the vocabulary is a pure function of the corpus and
    must match exactly -- including odysseus, which held 178,769 columns and
    used to be truncated to 60,000 by an unstable sort."""
    from sklearn.feature_extraction.text import CountVectorizer

    from loci.index import chunks_for, load_episodes
    from loci.text import tokens as vtokens

    for scope in ("zim-compress", "loci"):
        chunks = chunks_for(load_episodes(), scope)
        texts = [f"{c.heading} {c.text}" for c in chunks]
        cv = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                             lowercase=True)
        cv.fit(texts)
        rust = set()
        for t in texts:
            rust.update(_core.char_wb_ngrams(t.lower(), 3, 5))
        assert rust == set(cv.vocabulary_), (
            f"{scope}: {len(rust - set(cv.vocabulary_))} only-rust, "
            f"{len(set(cv.vocabulary_) - rust)} only-sklearn")


def _reference_questions():
    from parity.record import load_manifest
    return load_manifest(Path(__file__).parent.parent / "parity" / "corpus")["questions"]
