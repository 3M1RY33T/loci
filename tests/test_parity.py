"""Tests for the parity harness itself.

The harness grades every later phase of the Rust port, so a bug here is
invisible and expensive: a comparator that passes everything reports a clean
port that isn't one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parity.questions import Question, distinctive_terms, generate

INDEX = {
    "version": 3,
    "scopes": {
        "alpha": {"name": "Alpha", "root": "/tmp/alpha", "aliases": ["alpha"],
                  "chunk_count": 100, "token_count": 500},
        "beta": {"name": "Beta", "root": "/tmp/beta", "aliases": ["beta"],
                 "chunk_count": 80, "token_count": 400},
    },
    "postings": {
        "shared":   {"alpha": 50, "beta": 40},
        "widget":   {"alpha": 12},
        "gadget":   {"alpha": 12},
        "sprocket": {"alpha": 3},
        "flange":   {"beta": 9},
    },
}


def test_distinctive_terms_excludes_shared_tokens():
    # A term two scopes own cannot make a question with one right answer.
    assert "shared" not in distinctive_terms(INDEX, "alpha", k=10)


def test_distinctive_terms_orders_by_df_then_lexically():
    # Lexical tie-break, or the corpus changes between two runs on one machine.
    assert distinctive_terms(INDEX, "alpha", k=3) == ["gadget", "widget", "sprocket"]


def test_generation_is_deterministic():
    assert [q.id for q in generate(INDEX)] == [q.id for q in generate(INDEX)]


def test_question_ids_are_unique():
    ids = [q.id for q in generate(INDEX)]
    assert len(ids) == len(set(ids))


def test_every_scope_gets_a_scoped_question():
    scoped = {q.id.split(":")[1] for q in generate(INDEX) if q.family == "scoped"}
    assert scoped == {"alpha", "beta"}


def test_deictic_and_no_evidence_families_carry_no_cwd():
    # cwd answers "which project"; a question testing abstention must not be
    # handed the answer for free.
    for q in generate(INDEX):
        if q.family in ("deictic", "no_evidence"):
            assert q.no_cwd is True and q.cwd is None


def test_enumerative_questions_pair_terms_from_different_scopes():
    # Both terms from one scope routes to that one scope, which is ordinary
    # routing wearing set mode's name. The family exists to select two or more.
    enum = [q for q in generate(INDEX) if q.family == "enumerative"]
    assert enum, "no enumerative questions generated"
    for q in enum:
        a, b = q.id.split(":")[1].split("+")
        assert a != b
        ta = set(distinctive_terms(INDEX, a, k=10_000))
        tb = set(distinctive_terms(INDEX, b, k=10_000))
        used = set(q.text.replace("which of my projects use ", "").split(" or "))
        assert len(used & ta) == 1 and len(used & tb) == 1


def test_enumerative_scopes_are_the_vocabulary_richest():
    # Alphabetical order picked the three thinnest scopes for the family that
    # needs the strongest signal.
    from parity.questions import _by_vocabulary
    assert _by_vocabulary(INDEX)[0] == "alpha"   # 3 sole-owned terms vs beta's 1


def test_recording_filename_is_filesystem_safe():
    # Question ids carry a colon; a colon is legal on POSIX and not on Windows,
    # and the CI matrix already runs all three platforms.
    from parity.record import filename_for
    assert filename_for("scoped:3m1ry33t-github-io") == "scoped__3m1ry33t-github-io.json"
    assert ":" not in filename_for("cross:delroy")
    assert "+" not in filename_for("enumerative:delroy+odysseus")


def test_recording_round_trips(tmp_path):
    from parity.record import load_manifest, write_manifest
    qs = generate(INDEX)
    write_manifest(tmp_path, qs, loci_version="0.5.0", index_version=3,
                   now="2026-09-09T12:00:00+00:00")
    got = load_manifest(tmp_path)
    assert [q.to_json() for q in got["questions"]] == [q.to_json() for q in qs]
    assert got["loci_version"] == "0.5.0"
    assert got["now"] == "2026-09-09T12:00:00+00:00"


# -- comparator ------------------------------------------------------------
BASE = {
    "route": {"returncode": 0, "json": {
        "question": "q", "query_tokens": ["a", "b"],
        "ranked": ["alpha", "beta"], "selected": ["alpha"],
        "abstain": False, "top_score": 6.289, "top_matched": 3,
        "mode": None, "enumerative": False, "abstain_reason": None,
        "candidates": [], "detail": {"alpha": {"score": 6.289}},
    }},
    "ask": {"returncode": 0, "json": {"scopes": [
        {"scope_id": "alpha", "episodes": [
            {"chunk": {"source": "doc:README.md", "heading": "A"}, "score": 0.71},
            {"chunk": {"source": "doc:B.md", "heading": "B"}, "score": 0.44},
        ]}
    ]}},
}


def _mutate(path, value):
    import copy
    d = copy.deepcopy(BASE)
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return d


def test_identical_recordings_compare_clean():
    from parity.check import compare
    assert compare(BASE, BASE) == []


def test_score_drift_below_tolerance_is_accepted():
    from parity.check import compare
    assert compare(BASE, _mutate(["route", "json", "top_score"], 6.2890000004)) == []


def test_score_drift_above_tolerance_is_reported():
    from parity.check import compare
    diffs = compare(BASE, _mutate(["route", "json", "top_score"], 6.290))
    assert len(diffs) == 1 and diffs[0].kind == "numeric"


def test_a_reordered_scope_set_is_never_tolerated():
    from parity.check import compare
    diffs = compare(BASE, _mutate(["route", "json", "ranked"], ["beta", "alpha"]))
    assert diffs and all(d.kind == "exact" for d in diffs)


def test_a_flipped_verdict_is_reported():
    from parity.check import compare
    diffs = compare(BASE, _mutate(["route", "json", "abstain"], True))
    assert any(d.kind == "exact" and d.path.endswith("abstain") for d in diffs)


def test_reordered_episode_hits_are_reported_even_at_equal_scores():
    # Order IS the answer. Two hits that swap places are a ranking regression
    # the tolerance must not absorb.
    import copy
    from parity.check import compare
    d = copy.deepcopy(BASE)
    d["ask"]["json"]["scopes"][0]["episodes"].reverse()
    assert any(dd.kind == "exact" for dd in compare(BASE, d))


def test_a_changed_exit_code_is_reported():
    from parity.check import compare
    assert compare(BASE, _mutate(["route", "returncode"], 1)) != []


def test_a_dropped_field_is_reported_not_ignored():
    import copy
    from parity.check import compare
    d = copy.deepcopy(BASE)
    del d["route"]["json"]["abstain_reason"]
    assert any(dd.kind == "missing" for dd in compare(BASE, d))


def test_null_becoming_a_number_is_reported():
    # abstain_reason: None -> 0.0 must never pass as "close enough to null".
    from parity.check import compare
    diffs = compare(BASE, _mutate(["route", "json", "abstain_reason"], 0.0))
    assert diffs and diffs[0].kind == "numeric"


def test_a_shortened_list_is_reported():
    import copy
    from parity.check import compare
    d = copy.deepcopy(BASE)
    d["ask"]["json"]["scopes"][0]["episodes"].pop()
    assert any(dd.kind == "exact" for dd in compare(BASE, d))


def test_progress_bar_noise_is_ignored_even_when_indexed():
    # The real path is `ask.stderr_tail[0]`, not `ask.stderr_tail`. An endswith
    # check that forgets the list index compares a torch progress bar's
    # throughput, which differs every run -- it scored the corpus 6/39 against
    # the very build that recorded it.
    from parity.check import _ignored
    assert _ignored("ask.stderr_tail[0]")
    assert _ignored("route.stderr_tail")
    assert _ignored("ask.stdout_len")
    assert not _ignored("ask.json.selected[0]")
    assert not _ignored("ask.json.scopes[0].episodes[1].score")


def test_edge_runs_are_compared_as_a_set_but_node_order_is_not():
    from parity.check import compare
    a = "NODE x\nNODE y\nEDGE b\nEDGE a\n"
    permuted_edges = "NODE x\nNODE y\nEDGE a\nEDGE b\n"
    swapped_nodes = "NODE y\nNODE x\nEDGE b\nEDGE a\n"
    assert compare(a, permuted_edges) == []          # graphify's ordering: noise
    assert compare(a, swapped_nodes) != []           # BFS order: the answer


def test_a_genuinely_different_edge_is_still_reported():
    # Sorting the run must not turn a MISSING edge into a pass.
    from parity.check import compare
    assert compare("EDGE a\nEDGE b\n", "EDGE a\nEDGE c\n") != []
    assert compare("EDGE a\nEDGE b\n", "EDGE a\n") != []


# -- the decaying oracle ---------------------------------------------------
def test_scoring_clock_is_pinnable(monkeypatch):
    # Episode scores carry a recency term, so a recorded answer decays against
    # the wall clock. Measured: the corpus scored 31/39 against the very build
    # that recorded it, forty minutes later, purely from that decay.
    from datetime import datetime, timezone
    from loci.backends.episodes import _now

    monkeypatch.setenv("LOCI_NOW", "2026-09-09T12:00:00+00:00")
    assert _now() == datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)


def test_a_naive_pin_is_read_as_utc(monkeypatch):
    from datetime import timezone
    from loci.backends.episodes import _now
    monkeypatch.setenv("LOCI_NOW", "2026-09-09T12:00:00")
    assert _now().tzinfo == timezone.utc


def test_a_malformed_pin_does_not_freeze_the_clock(monkeypatch):
    # Silently freezing time on a typo would make every later score wrong in a
    # way nothing reports.
    from datetime import datetime, timezone
    from loci.backends.episodes import _now
    monkeypatch.setenv("LOCI_NOW", "not-a-timestamp")
    assert abs((_now() - datetime.now(timezone.utc)).total_seconds()) < 5


def test_an_unset_pin_leaves_behaviour_unchanged(monkeypatch):
    from datetime import datetime, timezone
    from loci.backends.episodes import _now
    monkeypatch.delenv("LOCI_NOW", raising=False)
    assert abs((_now() - datetime.now(timezone.utc)).total_seconds()) < 5


def test_recency_actually_decays_with_the_pinned_clock():
    # The mechanism the pin exists to control, asserted directly.
    from datetime import datetime, timezone
    from loci.backends.episodes import _recency
    ts = "2026-08-09T12:00:00+00:00"
    near = _recency(ts, datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc))
    far = _recency(ts, datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc))
    assert near > far > 0.0
