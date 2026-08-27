"""Tests for the parts where a regression would be silent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from loci.backends.episodes import (BuiltinEpisodeBackend, chunk_markdown,
                                      is_navigation, is_stub, pack)
from loci.router import route
from loci.scopes import make_scope, resolve, scope_for_cwd, slugify
from loci.text import tokens, unique_tokens
from loci.types import Chunk, Scope


# -- tokenizer -------------------------------------------------------------
def test_identifiers_decompose():
    assert tokens("GlassesBridge") == ["glasses", "bridge"]
    assert tokens("run_agent_turn()") == ["run", "agent", "turn"]
    assert tokens("src/worker-security.mjs") == ["src", "worker", "security", "mjs"]


def test_common_verbs_survive_stopwording():
    # These are identifier tokens, not prose noise; dropping them would delete
    # exactly what discriminates between scopes.
    for t in ("run", "get", "use", "work", "set"):
        assert t in tokens(f"how do I {t} this")


def test_unique_tokens_preserves_order():
    assert unique_tokens("step budget step landing") == ["step", "budget", "landing"]


# -- chunk hygiene ---------------------------------------------------------
def test_stub_and_navigation_detection():
    assert is_stub("- [[A]]\n- [[B]]")
    assert is_navigation("- [[A|a]] - one\n- [[B|b]] - two\n- [[C|c]] - three")
    # a single-line pointer is still a pointer
    assert is_navigation("- [[Risks/Known Risks|Risks]] - no tests, monolith")
    assert not is_navigation(
        "- **Language**: JavaScript\n- **Deployment**: Cloudflare Workers\n"
        "- **Database**: Cloudflare D1")


def test_pack_respects_target_size():
    body = "\n\n".join(f"Paragraph number {i} with some words in it." for i in range(40))
    pieces = pack(body)
    assert len(pieces) > 1
    assert all(len(p) <= 900 for p in pieces)


def test_chunk_markdown_carries_heading_path():
    md = "# Top\n\n## Middle\n\n" + ("Real sentence with enough words here. " * 4)
    chunks = chunk_markdown(md, "doc:x.md", "doc", "")
    assert chunks and chunks[0].heading == "Top > Middle"


# -- scopes ----------------------------------------------------------------
def test_slugify():
    assert slugify("Hlep Davay") == "hlep-davay"
    assert slugify("yigityildiz.dev") == "yigityildiz-dev"


def test_scope_for_cwd_prefers_deepest(tmp_path):
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    scopes = [Scope(id="o", name="outer", root=outer),
              Scope(id="i", name="inner", root=inner)]
    assert scope_for_cwd(scopes, inner).id == "i"
    assert scope_for_cwd(scopes, outer).id == "o"
    assert scope_for_cwd(scopes, tmp_path) is None


def test_resolve_by_alias():
    s = Scope(id="a", name="Alpha", root=Path("/tmp"), aliases=["alpha", "al"])
    assert resolve([s], "ALPHA") is s
    assert resolve([s], "nope") is None


# -- router ----------------------------------------------------------------
def _index(**scopes) -> dict:
    postings: dict[str, dict[str, int]] = {}
    meta = {}
    for sid, (name, root, toks, nodes) in scopes.items():
        meta[sid] = {"name": name, "root": root, "aliases": [name.lower()],
                     "node_count": nodes, "token_count": len(toks),
                     "chunk_count": 0, "sources": [], "updated_at": ""}
        for t, c in toks.items():
            postings.setdefault(t, {})[sid] = c
    return {"version": 1, "scopes": meta, "postings": postings}


def test_thresholds_stay_at_their_fitted_values():
    """Fitted across two corpora (symbol-indexed and prose-only). See router.py."""
    import loci.router as R
    assert (R.SIZE_PRIOR, R.WIDEN_RATIO, R.MIN_MATCHED) == (0.15, 0.85, 4)


def test_router_abstains_on_vague_questions():
    idx = _index(a=("Alpha", "/a", {"widget": 5, "gizmo": 3}, 100))
    r = route("what should I do next?", idx)
    assert r.abstain


def test_router_routes_on_specific_vocabulary():
    idx = _index(
        a=("Alpha", "/a", {"widget": 5, "gizmo": 3, "sprocket": 2}, 100),
        b=("Beta", "/b", {"flange": 4, "grommet": 2, "bolt": 1}, 100),
    )
    r = route("how does the widget gizmo sprocket work", idx)
    assert not r.abstain and r.selected[0] == "a"


def test_cwd_overrides_a_vague_question(tmp_path):
    root = tmp_path / "beta"
    root.mkdir()
    idx = _index(
        a=("Alpha", "/nowhere/alpha", {"widget": 500}, 10000),
        b=("Beta", str(root), {"flange": 1}, 10),
    )
    # Without cwd this is unroutable; with cwd it is decided.
    assert route("how do I run the tests?", idx).abstain
    r = route("how do I run the tests?", idx, cwd=root)
    assert not r.abstain and r.selected[0] == "b"


def test_size_prior_stops_the_biggest_scope_winning_everything():
    # A scope 100x larger contains ordinary words by accident; without size
    # normalization it wins every query that touches one.
    idx = _index(
        big=("Big", "/big", {"common": 900, "misc": 800}, 10000),
        small=("Small", "/small", {"common": 8, "flange": 5, "grommet": 4}, 50),
    )
    r = route("flange grommet common", idx)
    assert r.selected[0] == "small"


def test_one_decisive_token_beats_the_count_gate():
    # Two matched tokens is below MIN_MATCHED, but a token exclusive to one
    # scope and prominent within it is stronger evidence than three generic
    # ones -- a raw count cannot tell those apart.
    idx = _index(
        a=("Alpha", "/a", {"wrangler": 75, "database": 14}, 732),
        b=("Beta", "/b", {"database": 2180, "common": 40}, 119599),
        c=("Gamma", "/c", {"common": 5}, 400),
    )
    r = route("which projects use wrangler and a database?", idx)
    assert not r.abstain, "a decisive exclusive token should defeat abstention"
    assert r.selected[0] == "a"


def test_decisive_tier_does_not_admit_vague_questions():
    idx = _index(
        a=("Alpha", "/a", {"widget": 5, "thing": 3}, 100),
        b=("Beta", "/b", {"thing": 4, "stuff": 2}, 100),
    )
    assert route("what should I do next?", idx).abstain


def test_deictic_questions_abstain_without_a_working_directory(tmp_path):
    root = tmp_path / "beta"
    root.mkdir()
    idx = _index(
        a=("Alpha", "/nowhere", {"entry": 40, "point": 30, "start": 25}, 500),
        b=("Beta", str(root), {"entry": 5}, 100),
    )
    # Plenty of matched tokens, but nothing says WHICH project.
    assert route("What is the entry point of this project?", idx).abstain
    # cwd names the subject, so the same question is answerable.
    r = route("What is the entry point of this project?", idx, cwd=root)
    assert not r.abstain and r.selected[0] == "b"


def test_naming_the_project_defeats_deixis():
    idx = _index(a=("Alpha", "/a", {"entry": 40}, 500),
                 b=("Beta", "/b", {"entry": 5}, 100))
    r = route("What is the entry point of this project in Alpha?", idx)
    assert not r.abstain and r.selected[0] == "a"


def test_is_deictic_detects_pointing_phrases():
    from loci.router import is_deictic
    assert is_deictic("what does this project do?")
    assert is_deictic("what are the risks here?")
    assert is_deictic("how is the app deployed?")
    assert not is_deictic("how does urthreads verify an admin session cookie?")
    assert not is_deictic("which projects use wrangler?")


# -- packaging identity ----------------------------------------------------
def test_install_hints_name_the_distribution_not_the_import():
    """`pip install loci` installs somebody else's 2018 outlier-detection lib.

    The import name and the distribution name differ, so every user-facing
    install string has to say `loci-mem`. A global rename silently rewrote all
    five of them to the wrong one, which is exactly why this is a test.
    """
    import re
    root = Path(__file__).resolve().parent.parent
    files = [root / "README.md", *(root / "src" / "loci").rglob("*.py")]
    bad = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b(?:pip|pipx)\s+install\s+'?loci(?:\[|\s|'|$)", line):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, "install hints must say loci-mem:\n" + "\n".join(bad)


def test_distribution_name_matches_the_install_hints():
    # tomllib is 3.11+; the package supports 3.10, so the TEST has to too.
    tomllib = pytest.importorskip("tomllib", reason="3.11+; library itself is 3.10-safe")
    root = Path(__file__).resolve().parent.parent
    meta = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert meta["name"] == "loci-mem"
    assert list(meta["scripts"]) == ["loci"]


# -- docstring collector ---------------------------------------------------
PY_SAMPLE = '''
"""Module level explanation that is long enough to be worth keeping around."""

# Windows: force the loader to COPY model files instead of symlinking, because
# on a network share Windows refuses to follow them and the model fails to load.
X = 1


def trivial():
    """Return x."""


class Widget:
    def configure(self):
        """Force stable MIME types across platforms, because some setups
        inherit stale registry mappings and serve modules with the wrong
        content type, which makes the page load but not respond."""
'''


def test_python_extractor_finds_docstrings_and_comment_blocks():
    from loci.backends.docstrings import extract_python
    got = dict(extract_python(PY_SAMPLE, "sample.py"))
    assert "sample.py" in got                       # module docstring
    assert any("comment" in h for h in got)         # comment run
    assert "Widget.configure()" in got              # qualified symbol path
    assert "trivial()" not in got                   # "Return x." explains nothing


def test_extractor_skips_boilerplate():
    from loci.backends.docstrings import _keep
    assert not _keep("Copyright (c) 2026 Somebody. All rights reserved. Licensed "
                     "under the MIT license, see LICENSE for details.")
    assert not _keep("Return the value.")
    assert _keep("Force stable MIME types because stale registry mappings make "
                 "the page load but never respond to interaction.")


def test_clike_extractor_attributes_to_following_declaration():
    from loci.backends.docstrings import extract_clike
    src = ("/** Builds the admin session cookie, choosing SameSite by origin so "
           "loopback HTTP never emits None without Secure. */\n"
           "export function buildAdminSessionCookie(origin) {}\n")
    got = dict(extract_clike(src, "worker.mjs"))
    assert "buildAdminSessionCookie()" in got


def test_docstrings_are_held_back_from_routing_by_default():
    # Symbol vocabulary is bounded by the code; docstring prose is not. Mixing
    # the units makes scopes incomparable and breaks the size prior.
    from loci.index import ROUTING_KINDS
    assert "docstring" not in ROUTING_KINDS
    assert {"doc", "note", "commit", "session"} <= ROUTING_KINDS


def test_docstring_routing_fallback_stays_disabled():
    """Measured worse in both regimes -- see the note in index.py.

    Warm it costs 14 points of top-1; cold, applied uniformly, it halves
    accuracy. Docstrings are high-volume generic English, which is the worst
    possible routing vocabulary however little else a scope has.
    """
    from loci.index import DOCSTRING_FALLBACK_VOCAB
    assert DOCSTRING_FALLBACK_VOCAB == 0


# -- redaction -------------------------------------------------------------
SECRETS = [
    ('aws_key = "AKIAIOSFODNN7EXAMPLE"', "aws-key-id"),
    ("token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789", "github-token"),
    ("postgres://admin:hunter2hunter2@db.internal:5432/app", "connection-string"),
    ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----",
     "private-key"),
    ('client_secret = "aVeryLongOpaqueValue1234567890"', "secret-assignment"),
]
SAFE = [
    "The admin session cookie needs SameSite and Secure set together.",
    "def get_api_key(env): return env.get('API_KEY')",
    "password: required",
    "Deployment uses Cloudflare Workers and a D1 database.",
]


@pytest.mark.parametrize("text,label", SECRETS)
def test_redaction_catches_credentials(text, label):
    from loci.redact import redact
    out, found = redact(text)
    assert label in found
    assert "REDACTED" in out


@pytest.mark.parametrize("text", SAFE)
def test_redaction_leaves_ordinary_prose_alone(text):
    from loci.redact import redact
    out, found = redact(text)
    assert found == {} and out == text


def test_sensitive_files_are_never_collected(tmp_path):
    from loci.redact import is_sensitive_file
    for name in (".env", ".dev.vars", "secrets.yaml", "id_rsa", "server.pem"):
        assert is_sensitive_file(tmp_path / name), name
    for name in ("README.md", "app.py", "config.json", "index.ts"):
        assert not is_sensitive_file(tmp_path / name), name


def test_every_collection_path_redacts(tmp_path):
    """Chunks are built through one constructor so no path can skip redaction."""
    from loci.backends import episodes as ep
    leak = "ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
    # markdown body
    md = f"# Title\n\nWe rotated the token {leak} after the incident was found.\n"
    chunks = ep.chunk_markdown(md, "doc:x.md", "doc", "")
    assert chunks and leak not in chunks[0].text
    # a heading carrying one
    chunks = ep.chunk_markdown(f"# key {leak}\n\n" + ("word " * 20), "doc:y.md", "doc", "")
    assert chunks and leak not in chunks[0].heading


def test_redaction_preserves_the_key_name_so_the_chunk_still_routes():
    from loci.redact import redact
    out, _ = redact('client_secret = "aVeryLongOpaqueValue1234567890"')
    assert "client_secret" in out and "aVeryLongOpaqueValue" not in out


# -- episode gate ----------------------------------------------------------
def _chunks(*texts) -> list[Chunk]:
    return [Chunk(kind="doc", source=f"doc:{i}.md", heading="H", text=t)
            for i, t in enumerate(texts)]


def test_episode_search_returns_nothing_for_ungrounded_nonsense():
    from loci.backends import episodes as ep
    ep.reset_caches()
    chunks = _chunks(
        "Deployment uses Cloudflare Workers and a D1 SQLite database for storage.",
        "The admin dashboard supports moderation, analytics and logs.",
    )
    assert ep.BuiltinEpisodeBackend().search(
        "airspeed velocity of an unladen swallow", chunks, "t") == []


def test_episode_search_finds_grounded_content():
    from loci.backends import episodes as ep
    ep.reset_caches()
    chunks = _chunks(
        "Deployment uses Cloudflare Workers and a D1 SQLite database for storage.",
        "The admin dashboard supports moderation, analytics and logs.",
    )
    hits = ep.BuiltinEpisodeBackend().search("cloudflare d1 database", chunks, "t2")
    assert hits and "Cloudflare" in hits[0].chunk.text


def test_collect_is_empty_for_an_empty_scope(tmp_path):
    s = make_scope(tmp_path)
    assert BuiltinEpisodeBackend().collect(s) == []
