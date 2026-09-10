"""Differential tests: Rust against the Python it replaces.

Unit tests prove the Rust does what its author meant. These prove it does what
the OLD implementation did, over the real corpus, which is the only question
strict parity asks.

The references in tests/reference/ are verbatim copies of 0.5.0, taken with
`git show main:src/loci/<mod>.py`. They are tracked because once the live
module becomes a shim there is nothing left in Python to compare against.
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


def _reference(name: str):
    ref = Path(__file__).parent / "reference" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def py():
    return _reference("text_0_5_0")


# -- tokenizer -------------------------------------------------------------
@needs_corpus
def test_tokenizer_matches_python_over_every_chunk(py):
    store = json.loads((HOME / "episodes.json").read_text(encoding="utf-8"))
    checked = 0
    for chunks in (store.get("chunks") or {}).values():
        for c in chunks:
            text = f"{c.get('heading', '')} {c['text']}"
            assert _core.tokens(text, True) == py.tokens(text), text[:160]
            checked += 1
    assert checked > 1000, f"only {checked} chunks compared"


@needs_corpus
def test_tokenizer_matches_python_on_every_index_term(py):
    index = json.loads((HOME / "scope_index.json").read_text(encoding="utf-8"))
    terms = list(index.get("postings", {}))
    assert len(terms) > 1000, f"only {len(terms)} terms in the index"
    for term in terms:
        assert _core.tokens(term, True) == py.tokens(term), term


@needs_corpus
def test_tokenizer_matches_python_on_headings_and_sources(py):
    """Headings and citable sources carry punctuation and paths that chunk
    bodies mostly do not -- `doc:README.md > A > B`, `git:8a39216bd5`."""
    store = json.loads((HOME / "episodes.json").read_text(encoding="utf-8"))
    for chunks in (store.get("chunks") or {}).values():
        for c in chunks:
            for field in ("heading", "source"):
                v = c.get(field) or ""
                assert _core.tokens(v, True) == py.tokens(v), f"{field}: {v!r}"


def test_rules_signature_is_unchanged(py):
    # index.fingerprint hashes this. A different value silently reindexes every
    # installation, because the signature is what makes a tokenizer change
    # invalidate the cache by construction.
    assert _core.rules_signature() == py.rules_signature()


@pytest.mark.parametrize("text", [
    # camelCase / acronym decomposition -- the cases the Python comments cite
    "GlassesBridge", "OAuth", "macOS", "OpenID", "GraphQL", "IOError",
    "HTTPServer", "XMLHttpRequest", "parseHTMLString",
    # alphanumeric terms that a digit-stripping tokenizer destroyed
    "base64", "sha256", "D1", "S3", "n8n", "2fa", "md5", "utf8", "g2", "f1",
    # hex-blob guard, and the real words that are spellable in hex
    "41768130f0d5a159ec5100160890b2315ebb4fcb", "e4bcfe0", "741bbb9",
    "decade", "facade", "deface", "deeded",
    # scripts
    "invoice設定", "naïve café résumé", "日本語のドキュメント", "한국어",
    # separators and shapes
    "run_agent_turn", "get_node", "use_cache", "worktree",
    "docs/guide.md", "a-b-c", "__init__", "CONSTANT_NAME",
    # degenerate
    "", "   ", "a", "of", "the", "x" * 40, "1234", "...", "\n\t",
])
def test_tokenizer_matches_python_on_known_edge_cases(py, text):
    assert _core.tokens(text, True) == py.tokens(text)
    assert _core.tokens(text, False) == py.tokens(text, drop_stopwords=False)


@pytest.mark.parametrize("text", ["OAuth", "GlassesBridge", "invoice設定", "base64"])
def test_unique_tokens_matches_python(py, text):
    assert _core.unique_tokens(text, True) == py.unique_tokens(text)


@pytest.mark.parametrize("chunk,expected", [
    ("41768130f0d5a159ec5100160890b2315ebb4fcb", True),
    ("e4bcfe0", True),
    ("decade", False),      # all hex, but no digit
    ("123456", False),      # all hex, but no letter
    ("2fa", False),         # under HEX_BLOB_MIN
    ("zzzzzz", False),
])
def test_is_hex_blob_matches_python(py, chunk, expected):
    assert _core.is_hex_blob(chunk) is expected
    assert _core.is_hex_blob(chunk) == py.is_hex_blob(chunk)


@pytest.mark.parametrize("text,expected", [
    ("日本語", True), ("한국어", True), ("漢字", True),
    ("hello", False), ("naïve", False), ("", False),
])
def test_is_unsegmented_matches_python(py, text, expected):
    assert _core.is_unsegmented(text) is expected
    assert _core.is_unsegmented(text) == py.is_unsegmented(text)


@pytest.mark.parametrize("text", ["naïve café résumé", "hello", "日本語", ""])
def test_strip_diacritics_matches_python(py, text):
    assert _core.strip_diacritics(text) == py.strip_diacritics(text)
