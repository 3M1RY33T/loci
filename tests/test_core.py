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


@pytest.mark.parametrize("text,must_contain", [
    # A mixed-script identifier must not lose its non-Latin half. This one
    # tokenized to ['invoice'] before the fix, and the part that identified the
    # project silently vanished.
    ("invoice\u8a2d\u5b9a", ["invoice", "\u8a2d\u5b9a"]),
    ("\u8a2d\u5b9a\u51e6\u7406", ["\u8a2d\u5b9a\u51e6\u7406"]),
    ("\u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f", ["\u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f"]),
    ("na\u00efve_handler", ["naive", "handler"]),
    ("invoice_record", ["invoice", "record"]),
])
def test_tokenizer_keeps_every_script(text, must_contain):
    """Containment, not equality: unsegmented scripts also emit bigrams.

    What matters is that no script is silently dropped and that the whole run
    survives for exact matching.
    """
    got = tokens(text)
    for expected in must_contain:
        assert expected in got, f"{expected!r} missing from {got!r}"


def test_unsegmented_scripts_also_emit_bigrams():
    """A space-free script yields one token per phrase without segmentation.

    Measured on a real Chinese repository: only 35% of CJK tokens were short
    enough to be searchable and the longest was a 30-character sentence. Bigrams
    approximate word boundaries without a segmenter; the whole run is kept too,
    so an exact phrase still matches exactly.
    """
    got = tokens("\u8a2d\u5b9a\u51e6\u7406\u7ba1\u7406")
    assert "\u8a2d\u5b9a\u51e6\u7406\u7ba1\u7406" in got      # whole run kept
    assert "\u8a2d\u5b9a" in got and "\u51e6\u7406" in got       # bigrams present
    # Scripts that DO use spaces are already segmented and must not be exploded.
    assert tokens("\u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f") == ["\u043f\u0435\u0440\u0435\u043c\u0435\u043d\u043d\u0430\u044f"]
    assert tokens("handler") == ["handler"]


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


def test_absent_groups_is_not_the_same_as_empty_groups():
    """A registry written before groups existed must not read as "the user
    deliberately ungrouped this" -- that would make `groups infer` a no-op on
    every pre-upgrade install. Same trap as code_globs; see types.py.
    """
    absent = Scope.from_json({"id": "a", "name": "A", "root": "/a"})
    empty = Scope.from_json({"id": "a", "name": "A", "root": "/a", "groups": []})

    assert absent.groups is None
    assert empty.groups == []
    assert "groups" not in absent.to_json()
    assert empty.to_json()["groups"] == []


def test_group_set_is_empty_for_both_absent_and_empty():
    assert Scope(id="a", name="A", root=Path("/a")).group_set() == set()
    assert Scope(id="a", name="A", root=Path("/a"), groups=[]).group_set() == set()
    assert Scope(id="a", name="A", root=Path("/a"),
                 groups=["me", "client:acme"]).group_set() == {"me", "client:acme"}


def test_upsert_preserves_user_edits_across_a_rescan(tmp_path):
    """`upsert` replaces by id wholesale and `make_scope` regenerates defaults,
    so a re-scan silently discarded custom aliases. Groups would inherit that.
    """
    from loci.scopes import upsert

    root = tmp_path / "proj"
    root.mkdir()

    first = make_scope(root, name="proj")
    first.aliases = ["custom-alias"]
    first.groups = ["me", "client:acme"]
    registry = upsert([], first)

    registry = upsert(registry, make_scope(root, name="proj"))

    kept = registry[0]
    assert kept.aliases == ["custom-alias"], "re-scan discarded a custom alias"
    assert kept.groups == ["me", "client:acme"], "re-scan discarded groups"


def test_upsert_preserves_a_deliberate_ungrouping(tmp_path):
    from loci.scopes import upsert

    root = tmp_path / "proj"
    root.mkdir()
    first = make_scope(root, name="proj")
    first.groups = []                      # deliberate, not "unset"
    registry = upsert([], first)
    registry = upsert(registry, make_scope(root, name="proj"))
    assert registry[0].groups == []


def test_upsert_does_not_mutate_the_scope_it_is_given(tmp_path):
    """Preservation must produce a new scope, not edit the caller's.

    `loci add` prints the scope it built, so rewriting that object in place
    reported values the user had never asked for.
    """
    from loci.scopes import upsert

    root = tmp_path / "proj"
    root.mkdir()
    old = make_scope(root, name="proj")
    old.aliases = ["kept"]
    old.groups = ["me"]

    fresh = make_scope(root, name="proj")
    before = (list(fresh.aliases), fresh.groups)

    registry = upsert([old], fresh)

    assert registry[0].aliases == ["kept"]
    assert registry[0].groups == ["me"]
    assert (fresh.aliases, fresh.groups) == before, "upsert edited the caller's scope"


def test_a_rescan_delivers_new_default_globs_and_keeps_custom_ones(tmp_path, monkeypatch):
    """Preserving a glob list wholesale would freeze the defaults forever.

    `make_scope` always fills episode_globs and code_globs, so a stored list is
    always truthy -- carrying it forward unconditionally closes the only path by
    which a glob added to the defaults reaches an already registered scope, and
    there is no CLI surface for code_globs at all. Comparing the stored list
    against the current default does not rescue it either: once the defaults
    change, an untouched stored list differs from them too, which is
    indistinguishable from a user edit. So what is preserved is the difference --
    the globs the current defaults do not provide -- re-applied on top of the
    regenerated list. This is the trap types.py:38-41 describes for the absent
    key, reached from the other side.
    """
    import loci.scopes as S
    from loci.scopes import upsert

    root = tmp_path / "proj"
    root.mkdir()
    registered_before = make_scope(root, name="proj")
    customized = make_scope(
        root, name="proj",
        episode_globs=list(S.DEFAULT_EPISODE_GLOBS) + ["notes/**.md"])

    # a glob shipped in the defaults after both scopes were registered
    monkeypatch.setattr(S, "DEFAULT_CODE_GLOBS",
                        list(S.DEFAULT_CODE_GLOBS) + ["*.zig"])

    kept = upsert([registered_before], make_scope(root, name="proj"))[0]
    assert "*.zig" in kept.code_globs, \
        "a glob added to the defaults never reaches an already registered scope"

    kept = upsert([customized], make_scope(root, name="proj"))[0]
    assert "*.zig" in kept.code_globs, "a custom episode glob froze code_globs too"
    assert "notes/**.md" in kept.episode_globs, "re-scan discarded a custom glob"
    assert all(g in kept.episode_globs for g in S.DEFAULT_EPISODE_GLOBS), \
        "keeping the custom glob dropped the defaults"


def test_add_with_an_explicit_alias_beats_preservation(tmp_path, monkeypatch, capsys):
    """Preservation exists so a re-scan cannot clobber user edits, but a flag on
    `loci add` IS the user naming the value -- it has to win, or re-adding a
    registered path silently keeps the old aliases and prints the new ones.
    """
    from argparse import Namespace

    import loci.paths as P
    from loci.cli import cmd_add, cmd_scan
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    root = tmp_path / "proj"
    (root / ".git").mkdir(parents=True)     # so `scan` discovers it

    cmd_add(Namespace(path=str(root), name="proj", alias=["first"], glob=None))
    assert load_scopes()[0].aliases == ["first"]

    cmd_add(Namespace(path=str(root), name="proj", alias=["second"], glob=None))
    assert load_scopes()[0].aliases == ["second"], "explicit --alias was discarded"

    # No flag: now preservation applies, and the report must name what was
    # actually registered rather than the defaults `make_scope` regenerated.
    capsys.readouterr()
    cmd_add(Namespace(path=str(root), name="proj", alias=None, glob=None))
    out = capsys.readouterr().out
    assert load_scopes()[0].aliases == ["second"]
    assert "aliases: second" in out, f"reported an alias it did not register: {out!r}"

    cmd_scan(Namespace(roots=[str(root)], depth=1))
    assert load_scopes()[0].aliases == ["second"], "re-scan discarded a custom alias"


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


def test_thresholds_are_read_at_call_time_not_bound_at_import():
    """A default argument is evaluated once, when the function is defined.

    Binding thresholds in the signature meant `router.MIN_MATCHED = 6` had no
    effect on callers using the default, which silently turned several sweeps
    into no-ops and reported constants that matter as inert.
    """
    import loci.router as R

    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))
    q = "how does the widget gizmo sprocket work"
    # `evidence_floor` is passed explicitly: it resolves through the calibration
    # file by design, so a module attribute is not its source of truth.
    kw = dict(evidence_floor=10 ** 6)
    saved = (R.MIN_MATCHED, R.CONCENTRATED_EVIDENCE)
    try:
        R.CONCENTRATED_EVIDENCE = 10 ** 6
        R.MIN_MATCHED = 2
        assert not route(q, idx, **kw).abstain
        R.MIN_MATCHED = 99          # the only change; must flip the outcome
        assert route(q, idx, **kw).abstain, \
            "module-level threshold did not reach route()"
    finally:
        R.MIN_MATCHED, R.CONCENTRATED_EVIDENCE = saved


def test_thresholds_stay_at_their_fitted_values():
    """Fitted across two corpora (symbol-indexed and prose-only). See router.py."""
    import loci.router as R
    assert (R.SIZE_PRIOR, R.WIDEN_RATIO, R.MIN_MATCHED) == (0.15, 0.85, 4)


def test_fusion_weights_stay_at_their_fitted_values():
    """Fitted on a retrieval metric over two real corpora. See episodes.py.

    Both peaked at the same point and both beat the previous hand-set values;
    the band is flat for EMBED_WEIGHT in roughly [0.4, 0.7].
    """
    from loci.backends import episodes as ep
    assert (ep.BM25_WEIGHT, ep.CHAR_WEIGHT, ep.EMBED_WEIGHT) == (0.20, 0.20, 0.60)
    assert abs(ep.BM25_WEIGHT + ep.CHAR_WEIGHT + ep.EMBED_WEIGHT - 1.0) < 1e-9


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
    # ones -- a raw count cannot tell those apart. Handled by the concentrated
    # tier since DECISIVE_EVIDENCE was deleted as a duplicate of it.
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


# -- router: groups --------------------------------------------------------
def test_scope_idf_is_computed_over_the_whole_corpus_not_the_eligible_set():
    """Filtering candidates must not change how discriminative a token looks.

    If `eligible` shrank S in `scope_idf = log(1 + S/scope_df)`, surviving
    scopes' evidence would inflate and every calibrated threshold would move
    silently. Selection is filtered; scoring is not.
    """
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30}, 500),
                 b=("Beta", "/b", {"widget": 25, "flange": 10}, 400),
                 c=("Gamma", "/c", {"widget": 15}, 300))
    q = "how does the widget gizmo work"

    full = route(q, idx)
    narrowed = route(q, idx, eligible={"a"})

    assert narrowed.detail["a"]["evidence_total"] == full.detail["a"]["evidence_total"]
    assert narrowed.detail["a"]["score"] == full.detail["a"]["score"]


def test_group_parameters_left_unset_change_nothing():
    """The upgrade guarantee: inert until groups exist.

    Passing the parameters at their inert values must engage the machinery and
    still change nothing. An earlier version compared a call against the same
    call with explicit `None`s, which asserted nothing at all. This version
    fails if `demoted=set()` is read by truthiness-inverted logic, if
    `strict_group` fires without `eligible`, or if the penalty leaks into the
    un-demoted path.
    """
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))
    q = "how does the widget gizmo work"
    assert route(q, idx).to_json() == route(
        q, idx, demoted=set(), group_penalty=0.01, strict_group=True).to_json()


def test_an_unknown_group_confines_to_nothing_not_to_everything():
    """`eligible` is tri-state: None is unconfined, set() is confined to nothing.

    A truthiness check here would route an unknown --group across the corpus.
    Reachable in production: groups.py builds
    `Confinement(eligible=members([forced_group], scopes))`, and `members`
    returns set() when nobody is in the named group.
    """
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))
    q = "how does the widget gizmo sprocket work"

    r = route(q, idx, eligible=set())
    assert r.ranked == [] and r.selected == []
    assert r.abstain
    assert r.abstain_reason == "no_evidence"

    # The answer really is outside the (empty) group, and `a` clears the
    # concentrated tier, so `hard` attributes it rather than falling through.
    strict = route(q, idx, eligible=set(), strict_group=True)
    assert strict.ranked == [] and strict.selected == []
    assert strict.abstain
    assert strict.abstain_reason == "out_of_group"


def test_hard_group_does_not_blame_the_group_for_an_unroutable_question():
    """`out_of_group` must mean "the answer is elsewhere", not "nothing matched".

    `top_all` is argmax(scores) with no evidence gate of its own, so on a
    question that matches no vocabulary every scope sits near 0 and the winner
    is whoever took the 0.15 recency tiebreak. Attributing an abstention to the
    group because a scope scoring 0.15 sits outside it made `out_of_group`
    absorb both other reasons for every unroutable question under `hard`.
    """
    idx = _index(a=("Alpha", "/a", {"widget": 40}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))
    # `b` wins the recency tiebreak at 0.15 and is outside the group.
    assert route("xyzzy plugh frotz", idx).ranked[0] == "b"

    hard = dict(eligible={"a"}, strict_group=True)
    assert route("xyzzy plugh frotz", idx, **hard).abstain_reason == "no_evidence"
    assert route("how does this project start up?", idx,
                 **hard).abstain_reason == "deictic"


def test_hard_group_abstains_rather_than_returning_the_runner_up():
    """Returning the best in-group scope when the answer is elsewhere is the
    merged-index failure wearing a different hat. Design rule 2.
    """
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))
    q = "how does the widget gizmo sprocket work"

    open_result = route(q, idx)
    assert not open_result.abstain and open_result.ranked[0] == "a"

    confined = route(q, idx, eligible={"b"}, strict_group=True)
    assert confined.abstain
    assert confined.abstain_reason == "out_of_group"


def test_hard_group_abstains_when_the_answer_is_named_outside_it(tmp_path):
    """An alias or cwd hit on an out-of-group scope IS the answer being
    elsewhere, even though it contributes nothing to evidence_total or matched.

    Gating the out-of-group branch on lexical evidence alone read a scope
    winning purely on ALIAS_BOOST or CWD_BOOST as zero evidence, skipped the
    branch, and let the in-group runner-up's own evidence carry `enough` --
    returning a confident answer from the wrong project, which is the single
    failure hard mode exists to prevent. Reachable as `loci ask --group backend`
    asked from a frontend checkout, or any question naming a project outside
    the group.
    """
    a = tmp_path / "a"
    a.mkdir()
    # Alpha has NO vocabulary in common with the question; only the signal.
    idx = _index(a=("Alpha", str(a), {"unrelated": 5}, 500),
                 b=("Beta", "/b", {"widget": 40, "gizmo": 30, "sprocket": 20}, 400))
    q = "how does the widget gizmo sprocket work"
    hard = dict(eligible={"b"}, strict_group=True)

    by_alias = route("alpha " + q, idx, **hard)
    assert by_alias.detail["a"]["evidence_total"] == 0.0, "fixture lost its point"
    assert by_alias.abstain
    assert by_alias.abstain_reason == "out_of_group"

    by_cwd = route(q, idx, cwd=str(a), **hard)
    assert by_cwd.abstain
    assert by_cwd.abstain_reason == "out_of_group"


def test_explicit_narrowing_does_not_abstain():
    """--group narrows because you asked. Only `hard` refuses."""
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                 b=("Beta", "/b", {"widget": 25, "flange": 10}, 400))
    q = "how does the widget gizmo sprocket work"

    r = route(q, idx, eligible={"b"}, strict_group=False)
    assert not r.abstain
    assert r.selected == ["b"]


def test_soft_penalty_reorders_but_never_overrides_a_cwd_signal(tmp_path):
    """The penalty scales the evidence base BEFORE the boosts are added, so a
    demoted scope holding cwd never scores below CWD_BOOST -- 4.0 here, at
    every penalty down to 0. Measured, unboosted evidence occupies 0.1-1.5, so
    nothing on a real corpus reaches that floor.

    This is a bound, not an absolute: recency is added to both sides after the
    penalty, so the sweep below inverts once Beta's base reaches
    CWD_BOOST + RECENCY_BOOST * (recency(a) - recency(b)) -- 3.85 here, since
    Beta holds the freshest rank and Alpha the stalest. See the counterexample
    recorded in router.py.
    """
    a = tmp_path / "a"
    a.mkdir()
    idx = _index(a=("Alpha", str(a), {"widget": 5}, 500),
                 b=("Beta", "/b", {"widget": 40, "gizmo": 30, "sprocket": 20}, 400))
    q = "how does the widget gizmo sprocket work"

    for p in (1.0, 0.5, 0.1, 0.0):
        r = route(q, idx, demoted={"a"}, group_penalty=p, cwd=str(a))
        assert r.ranked[0] == "a", f"penalty {p} overrode a cwd signal"
        assert r.detail["a"]["score"] >= 4.0, "the penalty ate the boost itself"


def test_soft_penalty_does_reorder_without_a_boost():
    # Beta is the far larger scope, so the same raw counts are much less
    # prominent in it: Alpha leads 2.1876 to 0.7403 (Beta's 0.5903 plus the
    # 0.15 recency tiebreak) before any penalty. That gap is deliberately wide
    # enough on both sides to survive the default GROUP_PENALTY and to fall to
    # a tenth of it -- see the call-time test below, which depends on it.
    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30}, 500),
                 b=("Beta", "/b", {"widget": 30, "gizmo": 22}, 50000))
    q = "how does the widget gizmo work"

    assert route(q, idx).ranked[0] == "a"
    assert route(q, idx, demoted={"a"}, group_penalty=0.1).ranked[0] == "b"


def test_group_penalty_is_read_at_call_time_not_bound_at_import():
    """Same trap as MIN_MATCHED; see router.py.

    The fixture is tuned so the DEFAULT penalty leaves Alpha on top (1.0938 vs
    0.7403). A signature-bound default would therefore keep Alpha first here,
    which is what makes the flip below evidence that the module attribute
    reached `route`.
    """
    import loci.router as R

    idx = _index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30}, 500),
                 b=("Beta", "/b", {"widget": 30, "gizmo": 22}, 50000))
    q = "how does the widget gizmo work"
    saved = R.GROUP_PENALTY
    try:
        R.GROUP_PENALTY = 0.01
        assert route(q, idx, demoted={"a"}).ranked[0] == "b", \
            "module-level GROUP_PENALTY did not reach route()"
    finally:
        R.GROUP_PENALTY = saved


def test_abstention_always_carries_a_reason():
    """Today `route --explain` cannot say WHY it abstained."""
    idx = _index(a=("Alpha", "/a", {"widget": 40}, 500),
                 b=("Beta", "/b", {"flange": 25}, 400))

    assert route("how does this project start up?", idx).abstain_reason == "deictic"
    assert route("xyzzy plugh frotz", idx).abstain_reason == "no_evidence"
    assert route("how does the widget work", idx).abstain_reason in (None, "no_evidence")


def test_group_penalty_stays_at_its_documented_value():
    """Hand-set, never fitted. If this moves, the sweep has to move with it."""
    import loci.router as R
    assert R.GROUP_PENALTY == 0.5


def test_semantic_symbol_seeding_stays_narrow():
    """Two nearest labels, not four.

    graphify seeds a traversal by lexical similarity to a node label, which
    cannot connect a question phrased in behaviour to code named for its
    implementation. Matching embedded labels first bridges that -- structure
    probes went 4/7 to 6/7 -- but only while the bridge stays narrow: at three
    or more labels the added tokens are generic and dilute queries the lexical
    expansion had already aimed correctly, dropping it back to 5/7.
    """
    from loci.ask import SEMANTIC_SYMBOL_LABELS
    assert SEMANTIC_SYMBOL_LABELS <= 2


def test_semantic_symbols_degrade_silently_without_embeddings(tmp_path, monkeypatch):
    """No vectors must mean no seeding, not an exception."""
    import loci.paths as P
    from loci.ask import semantic_symbols
    from loci.backends import episodes as ep

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    ep.reset_caches()
    assert semantic_symbols("anything at all", "nosuchscope") == []


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


# -- calibration -----------------------------------------------------------
def test_gates_are_an_or_not_a_replacement():
    """Evidence complements the count gate; it does not replace it.

    Replacing the count gate with a fitted evidence floor scored better on the
    calibration set and worse on held-out questions -- cross-scope routing fell
    from 100% to 33%. Any one gate passing is enough.
    """
    # Two matched tokens, both exclusive and prominent: below the count gate,
    # well above the evidence floor. A small-vocabulary scope looks like this.
    idx = _index(tiny=("Tiny", "/t", {"mimetype": 30, "voxel": 25}, 80),
                 huge=("Huge", "/h", {"handler": 900, "config": 800}, 9000))
    r = route("how is mimetype handled together with voxel?", idx,
              min_matched=4, evidence_floor=1.0)
    assert not r.abstain and r.selected[0] == "tiny"
    # With EVERY gate raised out of reach it must abstain rather than guess.
    # There are three now; a test that enumerates them has to be updated when
    # one is added, which is the point of enumerating them.
    r = route("how is mimetype handled together with voxel?", idx,
              min_matched=99, evidence_floor=10**6, concentrated_evidence=10**6)
    assert r.abstain


def test_concentrated_tier_returns_every_owner_of_a_shared_term():
    """A term two projects share must come back with both of them.

    Returning one owner is the failure this tier fixes, not a partial success.
    Across five corpora these questions scored 0-17% before it existed, and on
    real repositories they abstained outright rather than ranking badly.
    """
    idx = _index(alpha=("Alpha", "/a", {"wrangler": 60, "alpha_only": 40}, 500),
                 beta=("Beta", "/b", {"wrangler": 55, "beta_only": 30}, 480),
                 gamma=("Gamma", "/g", {"unrelated": 90}, 500))
    r = route("how is wrangler handled?", idx)
    assert not r.abstain
    assert {"alpha", "beta"} <= set(r.selected)
    assert "gamma" not in r.selected


def test_calibration_sweeps_rather_than_taking_a_band_endpoint():
    """Bands overlap on real corpora, and then endpoints are indefensible.

    The routable minimum admits every unroutable question; the unroutable
    maximum refuses half the real ones. Only the best-classifying threshold
    survives overlap.
    """
    from loci.calibrate import ABSTAIN_WEIGHT, TERM_RANKS
    assert ABSTAIN_WEIGHT > 1.0          # refusing wrongly is cheaper than routing wrongly
    assert len(TERM_RANKS) > 1           # sample across rarity, not just the rarest


def test_calibration_round_trips(tmp_path, monkeypatch):
    import loci.paths as P
    from loci.calibrate import Calibration, load, save

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    cal = Calibration(7.17, 0.58, 0.09, 40, {"alpha": 0.61}, 0.84, 149, 14,
                      5.228, 7.123, False)
    save(cal)
    back = load()
    assert back is not None and back.evidence_floor == 7.17 and not back.trustworthy
    assert back.semantic_floor == 0.58 and back.semantic_n == 40
    assert back.semantic_by_scope == {"alpha": 0.61}


# -- self-service eval -----------------------------------------------------
def test_eval_questions_are_never_deictic():
    """A generated question must not be one the router is designed to refuse.

    The signature templates once included "what does it do with {b}" -- the
    pronoun tripped the deixis rule, the router correctly abstained, and the
    benchmark recorded it as a failure. That halved the reported score and would
    have told every user their projects were indistinguishable.
    """
    from loci.eval import SIGNATURE_TEMPLATES, TAXONOMY
    from loci.router import is_deictic

    for tpl in SIGNATURE_TEMPLATES:
        q = tpl.format(a="alpha", b="beta")
        assert not is_deictic(q), f"signature template is deictic: {tpl!r}"


def test_taxonomy_questions_are_unanswerable_without_a_working_directory():
    """The no-cwd family scores the router on REFUSING these, so each one must
    genuinely be unroutable — not merely deictic. Two of them ("how do I run the
    tests?") carry no pronoun at all and are refused on evidence instead, which
    is why this asserts the behaviour rather than the grammar.
    """
    from loci.eval import TAXONOMY
    from loci.router import route

    idx = _index(a=("Alpha", "/a", {"widget": 40, "test": 30, "config": 20}, 500),
                 b=("Beta", "/b", {"flange": 30, "deploy": 12, "entry": 9}, 400))
    for q in TAXONOMY:
        assert route(q, idx).abstain, f"taxonomy question is routable without cwd: {q!r}"


def test_signature_terms_do_not_exclude_non_latin(tmp_path):
    """A length floor tuned for English hides whole writing systems.

    CJK tokens are commonly two characters, so a flat `len < 4` filter excluded
    every one of them and the benchmark reported a CJK corpus as unroutable
    without ever having considered its vocabulary.
    """
    from loci.eval import signature_terms
    idx = _index(a=("Alpha", str(tmp_path), {"\u8a2d\u5b9a": 40, "\u51e6\u7406": 30,
                                             "handler": 5}, 200))
    terms = signature_terms(idx, "a", k=3)
    assert "\u8a2d\u5b9a" in terms and "\u51e6\u7406" in terms


def test_verdict_never_says_healthy_while_a_family_is_at_zero(tmp_path):
    """A summary line that contradicts the table above it is worse than none.

    `contended` sat at 0% across three separate corpora while the verdict read
    "routing looks healthy on this corpus."
    """
    from loci.eval import Family, render

    fams = {
        "cwd": Family("deictic + cwd", n=8, correct=8),
        "nocwd": Family("deictic, no cwd", n=8, correct=8),
        "nonsense": Family("unanswerable", n=6, correct=6),
        "signature": Family("signature", n=4, correct=4),
        "contended": Family("contended", n=12, correct=0),
    }
    out = render({"families": fams, "n_scopes": 4, "chance": 0.25})
    assert "healthy" not in out
    assert "share" in out.lower()


def test_semantic_floor_is_per_scope_not_pooled(tmp_path, monkeypatch):
    """The floor is compared against a MAXIMUM over a scope's chunks.

    The expected maximum of an unrelated query grows with how many chunks you
    take it over, so a scope with 8,000 chunks throws up a spurious 0.60 that a
    scope with 119 never will. Pooling the two collapsed the separation from
    +0.109 to -0.002 and produced a floor that fit neither.
    """
    import loci.paths as P
    from loci.backends import episodes as ep
    from loci.calibrate import Calibration, save

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    save(Calibration(7.0, 0.60, 0.11, 200, {"big": 0.68, "small": 0.58},
                     0.9, 100, 20, 8.0, 7.0, True))
    assert ep._calibrated_semantic_floor("big") == 0.68
    assert ep._calibrated_semantic_floor("small") == 0.58
    # a scope with no fit of its own falls back to the corpus mean
    assert ep._calibrated_semantic_floor("unknown") == 0.60


def test_semantic_floor_falls_back_without_calibration(tmp_path, monkeypatch):
    import loci.paths as P
    from loci.backends import episodes as ep

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    assert ep._calibrated_semantic_floor("anything") == ep.SEMANTIC_FLOOR


def test_fit_and_report_halves_never_overlap():
    """Calibration fits on one half and evaluation reports on the other.

    Without the split, a perfect `loci eval` straight after `loci calibrate` is
    partly the fitted threshold reading its own training data back.
    """
    from loci.eval import NONSENSE, SIGNATURE_TEMPLATES, TAXONOMY, halve

    for items in (TAXONOMY, NONSENSE, SIGNATURE_TEMPLATES):
        fit, report = halve(items, "fit"), halve(items, "report")
        assert not (set(fit) & set(report)), items
        assert set(fit) | set(report) == set(items)
        assert fit and report


def test_eval_needs_no_hand_labels(tmp_path):
    """Every family's gold answer is known by construction."""
    from loci.eval import run
    idx = _index(a=("Alpha", str(tmp_path / "a"), {"widget": 40, "gizmo": 20}, 500),
                 b=("Beta", str(tmp_path / "b"), {"flange": 30, "grommet": 15}, 400))
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    res = run(idx)
    fams = res["families"]
    from loci.eval import NONSENSE, TAXONOMY, halve
    assert fams["cwd"].n == 2 * len(halve(TAXONOMY, "report"))
    assert fams["nonsense"].n == len(halve(NONSENSE, "report"))
    assert fams["nocwd"].correct == fams["nocwd"].n   # all must abstain
    assert res["chance"] == 0.5


# -- glob matching ---------------------------------------------------------
@pytest.mark.parametrize("rel,pattern,expected", [
    # `**` matches ZERO or more directories, as pathlib does. This exact case
    # silently dropped 28 documentation files before it was fixed.
    ("docs/guide.md", "docs/**/*.md", True),
    ("docs/a/b/guide.md", "docs/**/*.md", True),
    ("guide.md", "docs/**/*.md", False),
    # a single `*` must NOT cross a separator, as pathlib does and fnmatch does not
    ("README.md", "*.md", True),
    ("docs/nested.md", "*.md", False),
    ("README.md", "README*", True),
    ("src/a/b.py", "**/*.py", True),
    ("main.py", "**/*.py", True),
    ("src/a/b.ts", "**/*.py", False),
    # Windows hands back backslashes
    ("docs\\guide.md", "docs/**/*.md", True),
    ("src\\a\\b.py", "**/*.py", True),
])
def test_glob_matching_follows_pathlib_semantics(rel, pattern, expected):
    from loci.walk import _matches
    assert _matches(rel, pattern) is expected


# -- durability ------------------------------------------------------------
def test_atomic_write_is_never_observed_half_written(tmp_path):
    """write_text truncates then fills; a reader can catch it mid-flight.

    Measured on an 8MB payload: 11 torn reads in a few seconds of concurrent
    access with write_text, 0 with atomic_write.
    """
    import json
    import threading

    from loci.paths import atomic_write

    target = tmp_path / "big.json"
    payload = json.dumps({"k": {str(i): "x" * 200 for i in range(8000)}})
    torn, done = [0], [False]

    def writer():
        for _ in range(10):
            atomic_write(target, payload)
        done[0] = True

    def reader():
        while not done[0]:
            try:
                if target.exists():
                    json.loads(target.read_text())
            except json.JSONDecodeError:
                torn[0] += 1
            except (FileNotFoundError, UnicodeDecodeError):
                pass

    w, r = threading.Thread(target=writer), threading.Thread(target=reader)
    w.start(), r.start(), w.join(), r.join()
    assert torn[0] == 0


def test_build_lock_refuses_a_second_writer(tmp_path, monkeypatch):
    import loci.paths as P

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    with P.BuildLock():
        with pytest.raises(SystemExit):
            with P.BuildLock():
                pass
    # released on exit
    with P.BuildLock():
        pass


def test_build_lock_breaks_a_stale_lock(tmp_path, monkeypatch):
    import loci.paths as P

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    (tmp_path / ".index.lock").write_text("999999 0")  # dead pid, ancient
    with P.BuildLock():
        pass
    assert not (tmp_path / ".index.lock").exists()


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


# -- setup -----------------------------------------------------------------
def _fake_repo(root: Path, name: str, prose: str) -> Path:
    d = root / name
    (d / ".git").mkdir(parents=True)
    (d / "README.md").write_text(f"# {name}\n\n{prose}\n", encoding="utf-8")
    return d


def test_setup_never_prompts_when_it_cannot_be_answered(tmp_path, monkeypatch, capsys):
    """A wizard that blocks on input is a wizard that hangs CI and agents.

    `loci setup` is the command an agent or a container image is most likely to
    run unattended, and a prompt there does not fail -- it waits forever, which
    is the worst available failure. Every question must take its default
    instead, so `input` is made to raise: reaching it at all is the bug.
    """
    import builtins

    import loci.paths as P
    from loci.setup import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("setup prompted with no terminal"))
    corpus = tmp_path / "corpus"
    # Long enough to clear MIN_CONTENT_WORDS: a shorter README is discarded as
    # a stub and the scope then has nothing indexable at all.
    _fake_repo(corpus, "alpha", "Signs the admin session cookie with a rotating "
                                "HMAC key and refuses SameSite None on localhost.")
    _fake_repo(corpus, "beta", "Compresses ZIM archives with zstd and streams "
                               "them to object storage from a manifest table.")

    rc = run([corpus], assume_yes=True, graphs=False, embed=False, calibrate=False)

    assert rc == 0
    from loci.index import load_index
    assert set(load_index()["scopes"]) == {"alpha", "beta"}


def test_setup_names_every_step_it_did_not_run(tmp_path, monkeypatch, capsys):
    """Silence about a skipped step reads as "done".

    Someone who never saw the embeddings question will believe semantic search
    is on. Each skip has to name itself and the command that fixes it later.
    """
    import loci.paths as P
    from loci.setup import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    # Long enough to clear MIN_CONTENT_WORDS: a shorter README is discarded as
    # a stub and the scope then has nothing indexable at all.
    _fake_repo(corpus, "alpha", "Signs the admin session cookie with a rotating "
                                "HMAC key and refuses SameSite None on localhost.")
    _fake_repo(corpus, "beta", "Compresses ZIM archives with zstd and streams "
                               "them to object storage from a manifest table.")

    run([corpus], assume_yes=True, graphs=False, embed=False, calibrate=False)
    out = capsys.readouterr().out

    assert "skipped:" in out
    for cmd in ("loci embed", "loci calibrate", "loci graphs"):
        assert cmd in out, f"{cmd!r} missing from the skip report"


def test_setup_reads_a_project_path_containing_spaces(monkeypatch):
    """`~/My Projects` is ordinary on macOS and Windows.

    Splitting the answer on whitespace turns one real directory into two that
    do not exist, and the only symptom is "0 git repositories found".
    """
    import builtins

    from loci.setup import _prompt_paths

    monkeypatch.setattr(builtins, "input", lambda *a: '"/tmp/My Projects" /tmp/code')
    got = _prompt_paths("where?", [Path("/tmp/fallback")], interactive=True)
    assert got == [Path("/tmp/My Projects"), Path("/tmp/code")]


# -- provenance ------------------------------------------------------------
@pytest.fixture
def hermetic_git(tmp_path, monkeypatch):
    """Neutralize the ambient git environment for the provenance tests.

    These are the first tests in the suite to run `git` for real, and two
    processes do it: the `_repo` helper here, and `loci.provenance` itself.
    Both inherit this environment, so both hazards close in one place.

    Ambient config would break the helper: `commit.gpgsign = true` with no
    usable key, a global `core.hooksPath`, or an `init.templateDir` carrying
    hooks all fail the `--allow-empty` commit, which errors every test in this
    section rather than one.

    The ceiling is for the module. git walks upward looking for `.git`, so a
    TMPDIR inside a checkout makes the not-a-repo case adopt an enclosing
    repository's origin -- measured, before this fixture existed:
    `remote_org(<plain dir under a checkout>) == 'enclosing-org'`, and
    `classify` then returned `vendor:enclosing-org` for a directory that is
    not a repository at all. That walk is done by the git `provenance` spawns,
    which is why these have to be on the process environment and not merely on
    the helper's subprocess.

    Needs git >= 2.32; older git ignores the variables and we are no worse off.
    """
    import os

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def _repo(base, name, *, origin=None, email="me@example.com"):
    """A real git repository with one commit, for provenance tests.

    Callers take the `hermetic_git` fixture: the commit below depends on it.
    """
    import subprocess

    r = base / name
    r.mkdir(parents=True)

    def run(*args):
        subprocess.run(["git", "-C", str(r), *args],
                       capture_output=True, check=True)

    run("init", "-q")
    run("config", "user.email", email)
    run("config", "user.name", "Test User")
    run("commit", "--allow-empty", "-qm", "init")
    if origin:
        run("remote", "add", "origin", origin)
    return r


def test_remote_org_parses_every_url_form(tmp_path, hermetic_git):
    """Including `ssh://` with a port, which a self-hosted GitLab or Gitea
    remote carries. Reading the port as the org produced a `vendor:2222`
    group -- a wrong answer, not a missing one.
    """
    from loci.provenance import remote_org

    ssh = _repo(tmp_path, "a", origin="git@github.com:3M1RY33T/Delroy.git")
    https = _repo(tmp_path, "b", origin="https://github.com/3M1RY33T/brewery.git")
    bare = _repo(tmp_path, "c")
    port = _repo(tmp_path, "d",
                 origin="ssh://git@git.example.com:2222/acme/thing.git")

    assert remote_org(ssh) == "3m1ry33t"
    assert remote_org(https) == "3m1ry33t"
    assert remote_org(bare) is None
    assert remote_org(port) == "acme"


def test_identity_is_the_modal_remote_org(tmp_path, hermetic_git):
    """No configuration: the org that owns most of your disk is you.

    Measured on the development corpus, `3M1RY33T` wins 7-to-1.
    """
    from loci.provenance import infer_identity

    roots = [_repo(tmp_path, f"mine{i}", origin=f"git@github.com:ACME/mine{i}.git")
             for i in range(4)]
    roots.append(_repo(tmp_path, "theirs",
                       origin="https://github.com/stranger/theirs.git"))

    identity = infer_identity(roots)
    assert identity.confident
    assert identity.org == "acme"


def test_identity_refuses_to_guess_without_a_plurality(tmp_path, hermetic_git):
    """A wrong identity inverts every classification, so this failure is loud."""
    from loci.provenance import infer_identity

    roots = [_repo(tmp_path, "a", origin="git@github.com:one/a.git"),
             _repo(tmp_path, "b", origin="git@github.com:two/b.git"),
             _repo(tmp_path, "c", origin="git@github.com:three/c.git")]

    identity = infer_identity(roots)
    assert not identity.confident
    assert identity.org is None


def test_classification_covers_every_rule(tmp_path, hermetic_git):
    """The five rules in the spec, including the two no-remote cases that
    decide `beacon` (foreign author) and `hlep_davay` (your author) correctly.
    """
    from loci.provenance import Identity, classify

    identity = Identity(org="acme", email="me@example.com", confident=True)

    mine = _repo(tmp_path, "mine", origin="git@github.com:ACME/mine.git")
    theirs = _repo(tmp_path, "theirs", origin="https://github.com/stranger/x.git")
    no_remote_mine = _repo(tmp_path, "nrm", email="me@example.com")
    no_remote_theirs = _repo(tmp_path, "nrt", email="demo@beacon.dev")
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()

    assert classify(mine, identity) == "me"
    assert classify(theirs, identity) == "vendor:stranger"
    assert classify(no_remote_mine, identity) == "me"
    assert classify(no_remote_theirs, identity) == "vendor:unknown"
    assert classify(not_a_repo, identity) == "me"


def test_classification_never_accuses_without_a_confident_identity(tmp_path,
                                                                   hermetic_git):
    """With no identity, everything is `me`. Labelling a user's own work as a
    vendor's is worse than labelling nothing.

    Both branches have to obey the guard, and the author-email one did not.
    `infer_identity` fills `email` from the ambient git config even when it
    refuses to name an org, so `Identity(org=None, email=<ambient>,
    confident=False)` is the state the production path actually returns -- and
    the second case below labelled it `vendor:unknown`. A bare
    `Identity(confident=False)` hides that, because `email` defaults to None
    and the branch fell out on the wrong condition.
    """
    from loci.provenance import Identity, classify

    unknown = Identity(confident=False)
    theirs = _repo(tmp_path, "theirs", origin="https://github.com/stranger/x.git")
    assert classify(theirs, unknown) == "me"

    no_org = Identity(org=None, email="me@example.com", confident=False)
    foreign = _repo(tmp_path, "foreign", email="demo@beacon.dev")
    assert classify(foreign, no_org) == "me"


# -- groups ----------------------------------------------------------------
def _scope(sid, root, groups=None):
    return Scope(id=sid, name=sid, root=Path(root), groups=groups)


def test_mode_falls_back_to_the_default_and_says_so():
    """The one objection to per-group-with-fallback is "two places to look".
    The answer is that resolution reports its own source.
    """
    from loci.groups import Policy

    p = Policy(default_mode="soft", groups={"client:delroy": "hard", "me": None})
    assert p.mode_for("client:delroy") == ("hard", "declared")
    assert p.mode_for("me") == ("soft", "default")
    assert p.mode_for("never-declared") == ("soft", "default")


def test_hard_confines_and_marks_itself_strict(tmp_path):
    from loci.groups import Policy, confinement

    scopes = [_scope("glasses", tmp_path / "mono" / "glasses",
                     ["me", "client:delroy", "delroy"]),
              _scope("loci", tmp_path / "loci", ["me"])]
    (tmp_path / "mono" / "glasses").mkdir(parents=True)
    (tmp_path / "loci").mkdir(parents=True)

    conf = confinement(Policy(default_mode="soft", groups={"client:delroy": "hard"}),
                       scopes, cwd=tmp_path / "mono" / "glasses")

    assert conf.mode == "hard"
    assert conf.strict is True
    assert conf.eligible == {"glasses"}
    assert conf.demoted is None


def test_soft_demotes_and_is_never_strict(tmp_path):
    from loci.groups import Policy, confinement

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    scopes = [_scope("a", tmp_path / "a", ["me"]),
              _scope("b", tmp_path / "b", ["vendor:stranger"])]

    conf = confinement(Policy(default_mode="soft"), scopes, cwd=tmp_path / "a")

    assert conf.mode == "soft"
    assert conf.strict is False
    assert conf.eligible is None
    assert conf.demoted == {"b"}


def test_strictest_group_wins_and_ties_take_the_union(tmp_path):
    from loci.groups import Policy, confinement

    for n in ("anchor", "x", "y", "z"):
        (tmp_path / n).mkdir()
    scopes = [_scope("anchor", tmp_path / "anchor", ["one", "two", "loose"]),
              _scope("x", tmp_path / "x", ["one"]),
              _scope("y", tmp_path / "y", ["two"]),
              _scope("z", tmp_path / "z", ["loose"])]

    policy = Policy(default_mode="soft",
                    groups={"one": "hard", "two": "hard", "loose": "explicit"})
    conf = confinement(policy, scopes, cwd=tmp_path / "anchor")

    assert conf.mode == "hard"
    assert conf.eligible == {"anchor", "x", "y"}, "tied strictest groups must union"


def test_no_groups_anywhere_is_completely_inert(tmp_path):
    """The upgrade path: an install with no policy and no memberships must take
    the pre-change code path exactly.
    """
    from loci.groups import Policy, confinement

    (tmp_path / "a").mkdir()
    conf = confinement(Policy(), [_scope("a", tmp_path / "a")], cwd=tmp_path / "a")
    assert conf.eligible is None and conf.demoted is None and conf.mode is None


def test_forced_group_narrows_even_under_explicit(tmp_path):
    """--group is the whole point of `explicit`: nothing happens until asked."""
    from loci.groups import Policy, confinement

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    scopes = [_scope("a", tmp_path / "a", ["me"]),
              _scope("b", tmp_path / "b", ["vendor:x"])]

    policy = Policy(default_mode="explicit")
    assert confinement(policy, scopes, cwd=tmp_path / "a").eligible is None

    conf = confinement(policy, scopes, forced_group="me")
    assert conf.eligible == {"a"}
    assert conf.strict is False, "explicit narrows, but does not abstain"


def test_policy_round_trips(tmp_path, monkeypatch):
    from loci import groups as G

    monkeypatch.setenv("LOCI_HOME", str(tmp_path))
    G.save_policy(G.Policy(default_mode="hard", groups={"client:delroy": "hard"}))
    back = G.load_policy()
    assert back.default_mode == "hard"
    assert back.groups == {"client:delroy": "hard"}


def test_missing_policy_file_is_the_shipped_default(tmp_path, monkeypatch):
    from loci import groups as G

    monkeypatch.setenv("LOCI_HOME", str(tmp_path / "empty"))
    p = G.load_policy()
    assert p.default_mode == G.DEFAULT_MODE and p.groups == {}


@pytest.mark.parametrize("body", ['{"groups": ', '[]', '{"groups": ["me"]}'])
def test_a_hand_edited_policy_file_degrades_instead_of_crashing(tmp_path,
                                                                monkeypatch, body):
    """groups.json is the one file a user is invited to author by hand, so every
    shape it can arrive in has to land on the default rather than a traceback.

    Truncated JSON already fell back; valid JSON of the wrong shape did not --
    a top-level list, or `groups` written as the plain list of names the key
    invites, both reached `.get`/`.items()` on a list and raised AttributeError
    out of every command that loads the policy.
    """
    from loci import groups as G

    monkeypatch.setenv("LOCI_HOME", str(tmp_path))
    G.groups_file().write_text(body, encoding="utf-8")

    p = G.load_policy()
    assert p.default_mode == G.DEFAULT_MODE and p.groups == {}


def test_confinement_names_the_groups_it_confined_to(tmp_path):
    """`group` is display text -- ", ".join(winners) -- so a caller feeding it
    back to `members` is right in every single-group case and silently gets
    `set()` the first time two groups tie, which is the union case groups exist
    to support. `names` is the accessor that survives it, on every branch.
    """
    from loci.groups import Policy, confinement, members

    for n in ("anchor", "x", "y"):
        (tmp_path / n).mkdir()
    scopes = [_scope("anchor", tmp_path / "anchor", ["one", "two"]),
              _scope("x", tmp_path / "x", ["one"]),
              _scope("y", tmp_path / "y", ["two"])]
    anchor = tmp_path / "anchor"

    hard = confinement(Policy(default_mode="soft",
                              groups={"one": "hard", "two": "hard"}), scopes,
                       cwd=anchor)
    assert hard.names == ("one", "two")
    assert members(hard.names, scopes) == hard.eligible
    assert members([hard.group], scopes) == set(), "the label is not an identifier"

    assert confinement(Policy(default_mode="soft"), scopes, cwd=anchor).names \
        == ("one", "two")
    assert confinement(Policy(default_mode="explicit"), scopes, cwd=anchor).names \
        == ("one", "two")
    assert confinement(Policy(), scopes, forced_group="one").names == ("one",)
    assert confinement(Policy(), [_scope("bare", anchor)], cwd=anchor).names == ()


def test_policy_refuses_a_mode_that_does_not_exist():
    """One bad value failed two different ways, and the quiet one is worse: via
    cwd it raised `KeyError: 'strict'` from inside `confining_groups`, but via
    `--group` it raised nothing and returned a fully-formed Confinement carrying
    a mode no branch below implements. The CLI builds a Policy out of user text,
    so the check belongs at construction.
    """
    from loci.groups import Policy

    with pytest.raises(ValueError):
        Policy(default_mode="strict")


def test_an_unknown_forced_group_confines_to_nothing(tmp_path):
    """`eligible` is tri-state and the two falsy states mean opposite things:
    `None` is unconfined, `set()` is confined to nothing. A router writing
    `if conf.eligible:` would answer `--group typo` from every scope on disk.
    """
    from loci.groups import Policy, confinement

    (tmp_path / "a").mkdir()
    scopes = [_scope("a", tmp_path / "a", ["me"])]

    assert confinement(Policy(), scopes, forced_group="typo").eligible == set()
