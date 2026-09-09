"""ONNX must reproduce sentence-transformers, or the calibration is invalid.

~/.loci/calibration.json holds semantic_floor 0.639 and eight per-scope floors,
fitted over 264 samples as COSINES PRODUCED BY THIS MODEL UNDER
SENTENCE-TRANSFORMERS. An encoder that drifts does not degrade retrieval a
little -- it silently invalidates all nine numbers, and every downstream
measurement taken after that point is against a moved baseline.
"""
from __future__ import annotations

import numpy as np
import pytest

from loci.backends.episodes import RERANK_MODEL
from loci.index import DEFAULT_EMBED_MODEL

# sentence-transformers is a DEV dependency and stays one. It is the reference
# these tests compare against -- dropping it would leave the ONNX path with
# nothing proving it still matches the encoder the floors were fitted to.
pytestmark = pytest.mark.reference

st = pytest.importorskip("sentence_transformers",
                         reason="parity test needs the thing being replaced")

TEXTS = [
    "Deployment: Cloudflare Workers and D1",
    "why was the admin session cookie dropped on localhost?",
    "def build_parser() -> argparse.ArgumentParser",
    "SameSite=None requires Secure, which localhost over http is not",
    "naïve café résumé",      # non-ASCII, to catch a tokenizer mismatch
    "x" * 4000,               # longer than the 512-token window; must truncate
    "",                       # empty string: a real chunk edge case
]
QUERY = "how does the session cookie get dropped"


def test_encoder_matches_sentence_transformers():
    from loci import embed

    ref = st.SentenceTransformer(DEFAULT_EMBED_MODEL)
    expected = ref.encode(TEXTS, normalize_embeddings=True, batch_size=64)
    actual = embed.encode(TEXTS, model_name=DEFAULT_EMBED_MODEL)

    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    assert np.abs(actual - expected).max() < 1e-4, "per-element drift"

    # The number that actually matters: cosine, which is what the floor is in.
    cos = (actual * expected).sum(axis=1)
    assert cos.min() > 1.0 - 1e-5, f"cosine drift, min={cos.min()}"


def test_query_prefix_is_applied_to_queries_only():
    from loci import embed

    ref = st.SentenceTransformer(DEFAULT_EMBED_MODEL)
    prefix = "Represent this sentence for searching relevant passages: "
    expected = ref.encode([prefix + QUERY], normalize_embeddings=True)[0]
    actual = embed.encode([QUERY], model_name=DEFAULT_EMBED_MODEL, is_query=True)[0]
    assert float(actual @ expected) > 1.0 - 1e-5

    unprefixed = embed.encode([QUERY], model_name=DEFAULT_EMBED_MODEL)[0]
    assert float(unprefixed @ actual) < 0.999, "prefix had no effect"


def test_output_is_l2_normalised():
    from loci import embed
    vecs = embed.encode(TEXTS, model_name=DEFAULT_EMBED_MODEL)
    # Every row but the empty string, which is allowed to be degenerate.
    norms = np.linalg.norm(vecs[:-1], axis=1)
    assert np.abs(norms - 1.0).max() < 1e-5


def test_empty_input_returns_an_empty_array():
    from loci import embed
    assert embed.encode([], model_name=DEFAULT_EMBED_MODEL).shape[0] == 0


def test_cross_encoder_matches_sentence_transformers():
    from loci import embed

    pairs = [(QUERY, t) for t in TEXTS[:4]]
    ref = st.CrossEncoder(RERANK_MODEL, max_length=512)
    expected = list(ref.predict(pairs, show_progress_bar=False))
    actual = embed.rerank_scores(pairs, model_name=RERANK_MODEL)

    assert len(actual) == len(expected)
    assert max(abs(float(a) - float(b)) for a, b in zip(actual, expected)) < 1e-3
    # Ordering is what rerank exists to produce; it must not merely be close.
    assert np.argsort(actual).tolist() == np.argsort(expected).tolist()


def test_no_progress_bar_reaches_stderr(capfd):
    from loci import embed
    embed.reset_sessions()
    embed.encode(TEXTS[:2], model_name=DEFAULT_EMBED_MODEL)
    err = capfd.readouterr().err
    assert "it/s" not in err and "%|" not in err
