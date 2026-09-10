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
    # Named under `loci` so `from .defaults import SKIP_DIRS` in the verbatim
    # walk copy resolves. Setting __package__ alone works but disagrees with
    # __spec__.parent, which Python warns about.
    spec = importlib.util.spec_from_file_location(f"loci.{name}", ref)
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


# -- walk ------------------------------------------------------------------
@pytest.fixture(scope="module")
def pywalk():
    return _reference("walk_0_5_0")


def _scope_rows():
    rows = json.loads((HOME / "scopes.json").read_text(encoding="utf-8"))
    rows = rows.get("scopes", rows) if isinstance(rows, dict) else rows
    return list(rows.values()) if isinstance(rows, dict) else rows


@needs_corpus
def test_walk_matches_python_over_every_registered_scope(pywalk):
    """The walk decides what gets indexed. A pattern matching one file fewer is
    a silently smaller corpus, and the ROADMAP records that exact failure: a
    glob bug dropped 28 files, the routing number moved, and it was diagnosed
    as threshold noise before anyone diffed the inputs."""
    from loci.defaults import SKIP_DIRS

    skip = sorted(SKIP_DIRS)
    compared = 0
    for s in _scope_rows():
        root = Path(s["root"])
        if not root.is_dir():
            continue
        pats = list(s.get("episode_globs") or []) + list(s.get("code_globs") or [])
        got = _core.iter_files(str(root), pats, [], skip)
        want = [str(p) for p in pywalk.iter_files(root, pats)]
        assert got == want, (
            f"{s['id']}: rust {len(got)} files, python {len(want)}; "
            f"only-rust={sorted(set(got) - set(want))[:5]} "
            f"only-python={sorted(set(want) - set(got))[:5]}")
        compared += 1
    assert compared >= 10, f"only {compared} scopes walked"


def _tree(root: Path) -> None:
    """A tree with every shape the walk has to handle."""
    for rel in (
        "top.md",
        "docs/guide.md",
        "docs/deep/nested/ref.md",
        "src/app.py",
        "sub/README.md",              # a nested scope's territory
        "sub/deep/notes.md",
        "node_modules/pkg/index.md",  # SKIP_DIRS
        ".hidden/secret.md",          # dot-prefixed
        "target/build.md",            # SKIP_DIRS
    ):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")


PATTERNS = ["*.md", "docs/**/*.md", "**/*.py", "sub/**/*.md"]


def test_walk_prunes_the_same_subtrees_as_python(pywalk, tmp_path):
    """No registered scope currently has an excluded subtree, so the corpus
    cannot exercise pruning at all -- the comparison over real scopes passes
    vacuously for it. This builds a tree that does."""
    from loci.defaults import SKIP_DIRS

    _tree(tmp_path)
    skip = sorted(SKIP_DIRS)

    got = _core.iter_files(str(tmp_path), PATTERNS, [], skip)
    want = [str(p) for p in pywalk.iter_files(tmp_path, PATTERNS)]
    assert got == want

    names = {Path(p).name for p in got}
    assert "index.md" not in names, "node_modules was not pruned"
    assert "build.md" not in names, "target was not pruned"
    assert "secret.md" not in names, "a dot-directory was not pruned"
    assert {"top.md", "guide.md", "ref.md", "app.py"} <= names


def test_walk_honours_an_excluded_subtree(pywalk, tmp_path):
    from loci.defaults import SKIP_DIRS

    _tree(tmp_path)
    skip = sorted(SKIP_DIRS)
    excluded = [tmp_path / "sub"]

    got = _core.iter_files(str(tmp_path), PATTERNS, [str(e) for e in excluded], skip)
    want = [str(p) for p in pywalk.iter_files(tmp_path, PATTERNS, exclude=excluded)]
    assert got == want
    assert not any("/sub/" in p for p in got), "the excluded subtree was walked"
    assert any(p.endswith("docs/guide.md") for p in got), "pruning took too much"


def test_walk_returns_a_sorted_deduplicated_list(pywalk, tmp_path):
    # Two patterns match the same file; it must appear once, and the order is
    # sorted regardless of how the directories were traversed.
    from loci.defaults import SKIP_DIRS

    _tree(tmp_path)
    pats = ["docs/**/*.md", "**/*.md"]
    got = _core.iter_files(str(tmp_path), pats, [], sorted(SKIP_DIRS))
    assert got == sorted(got)
    assert len(got) == len(set(got))
    assert got == [str(p) for p in pywalk.iter_files(tmp_path, pats)]


@needs_corpus
def test_walk_honours_exclusions_over_the_real_corpus(pywalk):
    """Kept, but it reports when it has nothing to compare rather than
    passing silently."""
    from loci.defaults import SKIP_DIRS
    from loci.scopes import load_scopes, nested_roots

    skip = sorted(SKIP_DIRS)
    registry = load_scopes()
    compared = 0
    for sc in registry:
        excl = nested_roots(sc, registry)
        if not excl or not sc.root.is_dir():
            continue
        pats = list(sc.episode_globs or []) + list(sc.code_globs or [])
        got = _core.iter_files(str(sc.root), pats, [str(e) for e in excl], skip)
        want = [str(p) for p in pywalk.iter_files(sc.root, pats, exclude=excl)]
        assert got == want, f"{sc.id} with {len(excl)} exclusions"
        compared += 1
    if compared == 0:
        pytest.skip("no registered scope has a nested sub-scope to exclude")


@pytest.mark.parametrize("pattern,rel,expected", [
    ("*.md", "guide.md", True),
    ("*.md", "docs/guide.md", False),          # a single * must not cross /
    ("docs/**/*.md", "docs/guide.md", True),   # ** matches ZERO directories
    ("docs/**/*.md", "docs/a/b/guide.md", True),
    ("docs/**/*.md", "guide.md", False),
    ("**/*.py", "a/b/c.py", True),
    ("**/*.py", "c.py", True),
    ("?.py", "a.py", True),
    ("?.py", "ab.py", False),
    ("README*", "README.md", True),
    ("README*", "docs/README.md", False),
    ("*.md", "guide.markdown", False),         # anchored at the end
])
def test_glob_translation_has_pathlib_semantics_not_fnmatch(pywalk, pattern, rel, expected):
    # Both were real bugs. fnmatch lets * cross a separator, and has no notion
    # of ** at all, which silently skipped docs/guide.md across 28 files.
    assert _core.glob_matches(rel, pattern) is expected
    assert _core.glob_matches(rel, pattern) == pywalk._matches(rel, pattern)


def test_a_windows_separator_is_normalised(pywalk):
    assert _core.glob_matches("docs\\guide.md", "docs/*.md") is True
    assert _core.glob_matches("docs\\guide.md", "docs/*.md") == pywalk._matches(
        "docs\\guide.md", "docs/*.md")


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
