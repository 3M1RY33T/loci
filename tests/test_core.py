"""Tests for the parts where a regression would be silent."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from loci.backends.episodes import (BuiltinEpisodeBackend, chunk_markdown,
                                      is_navigation, is_stub, pack)
from loci.index import INDEX_VERSION
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


def test_alphanumeric_terms_survive_the_tokenizer():
    """`D1` is a database, not a stray `d`.

    The digit-stripping pattern deleted 258 distinct alphanumeric terms across
    1,246 occurrences in a 53-file sample -- including `d1`, which this
    project's own README used as a routing example.
    """
    assert "d1" in tokens("which projects use wrangler and D1?")
    assert "s3" in tokens("uploaded to the S3 bucket")
    assert "n8n" in tokens("an n8n workflow")
    for term in ("utf8", "md5", "sha1", "2fa", "ipv6", "x86", "gpt4"):
        assert term in tokens(f"see the {term} handler"), term


def test_alphanumeric_output_is_a_superset_of_the_letters_only_form():
    """`base64` must yield BOTH forms, so no query that matched can stop matching."""
    got = tokens("base64 sha256 python3 warc2zim")
    for whole in ("base64", "sha256", "python3", "warc2zim"):
        assert whole in got, whole
    for part in ("base", "sha", "python", "warc", "zim"):
        assert part in got, part


def test_a_scope_name_carrying_digits_can_still_be_an_alias():
    """ALIAS_BOOST is the strongest signal in the router and it was unreachable.

    A scope named `3M1RY33T` tokenized its only alias to `[]`, so `_alias_hit`
    could never fire and naming the project outright abstained.
    """
    assert tokens("3M1RY33T") == ["3m1ry33t"]
    assert unique_tokens("what is 3M1RY33T for?") == ["3m1ry33t"]


def test_commit_hashes_do_not_become_vocabulary():
    """Unique by construction, so scope-IDF would rate every one maximally."""
    for h in ("e4bcfe0", "741bbb9", "ecef6264",
              "41768130f0d5a159ec5100160890b2315ebb4fcb"):
        assert tokens(h) == [], h
    # Bare numbers are not vocabulary either, now that digits survive.
    assert tokens("released in 2024 after 1000 commits") == [
        "released", "after", "commits"]   # `after` is not in STOPWORDS


def test_the_hex_guard_keeps_words_that_are_spellable_in_hex():
    """Dropping the digit requirement would delete real English."""
    for word in ("decade", "facade", "deface", "deeded", "deadbeef"):
        assert word in tokens(word), word
    # ...and short alphanumerics are never tested against it at all.
    assert "2fa" in tokens("2fa")


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


def test_resolve_answers_an_id_before_any_other_scopes_name():
    """Only the id is uniquified. `make_scope` takes `name` from the directory
    and `_unique_id` suffixes the id, so two clones of one upstream repo
    register as (`utils`, "utils") and (`utils-2`, "utils").

    Resolved by one three-way `or` per scope, `resolve("utils")` answered
    whichever came first in registry order -- filesystem order -- so in half of
    all scans the scope whose id is literally `utils` was unreachable by any
    string, and `loci group rm utils me` edited its sibling and reported
    success. BOTH orders are checked here: one of them passes under the old
    code, which is exactly why a single-order test would have missed this.

    It also pins the two-PASS structure rather than the ordering of one
    comparison: checking `s.id` before `s.name` inside a single loop is the same
    expression, and still lets an earlier scope's name beat a later scope's id.
    """
    a = Scope(id="utils", name="utils", root=Path("/x/utils"), aliases=["utils"])
    b = Scope(id="utils-2", name="utils", root=Path("/y/utils"), aliases=["utils"])

    for order in ([a, b], [b, a]):
        assert resolve(order, "utils") is a
        assert resolve(order, "utils-2") is b
    # A name still resolves when no id claims it -- this is a precedence rule,
    # not the removal of name lookup.
    assert resolve([b], "utils") is b


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

    cmd_scan(Namespace(roots=[str(root)], depth=1, split=False))
    assert load_scopes()[0].aliases == ["second"], "re-scan discarded a custom alias"


# -- scopes: monorepo split ------------------------------------------------
def test_subscopes_finds_depth_one_markers_only(tmp_path):
    """Depth-1 is what keeps client/static/package.json from minting a scope
    called `static`, and benchmarks/*/pyproject.toml from minting two harnesses.
    """
    from loci.scopes import subscopes

    repo = tmp_path / "mono"
    (repo / ".git").mkdir(parents=True)
    for name in ("glasses", "extension"):
        (repo / name).mkdir()
        (repo / name / "package.json").write_text("{}", encoding="utf-8")

    deep = repo / "client" / "static"
    deep.mkdir(parents=True)
    (deep / "package.json").write_text("{}", encoding="utf-8")

    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "package.json").write_text("{}", encoding="utf-8")

    assert {p.name for p in subscopes(repo, markers=True)} == {"glasses", "extension"}


def test_declaration_file_adds_what_markers_cannot_see(tmp_path):
    """`client/` is a flat Python package with no marker, and it is the biggest
    contributor in the real monorepo. Markers alone miss exactly that.
    """
    from loci.scopes import subscopes

    repo = tmp_path / "mono"
    (repo / ".git").mkdir(parents=True)
    (repo / "client").mkdir()
    (repo / ".loci.json").write_text(
        json.dumps({"scopes": [{"path": "client"}]}), encoding="utf-8")

    assert {p.name for p in subscopes(repo)} == {"client"}


def test_declaration_cannot_escape_the_repository(tmp_path):
    from loci.scopes import subscopes

    repo = tmp_path / "mono"
    (repo / ".git").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    (repo / ".loci.json").write_text(
        json.dumps({"scopes": [{"path": "../elsewhere"}]}), encoding="utf-8")

    assert subscopes(repo) == []


def test_malformed_declaration_is_ignored_not_fatal(tmp_path):
    """A scan that aborts on a stray comma is worse than one that misses a
    declaration -- and "malformed" is not only unparseable text. Valid JSON of
    the wrong shape reaches `.get` on a list or a string, which raises
    AttributeError, not the ValueError that json.loads raises.
    """
    from loci.scopes import subscopes

    repo = tmp_path / "mono"
    (repo / ".git").mkdir(parents=True)
    (repo / "glasses").mkdir()
    (repo / "glasses" / "package.json").write_text("{}", encoding="utf-8")

    for payload in ("{not json at all",          # unparseable
                    '["client"]',                # parses, but is not an object
                    '{"scopes": "client"}',      # `scopes` is not a list
                    '{"scopes": [["client"]]}',  # an entry is not an object
                    '{"scopes": [null]}'):       # an entry is nothing at all
        (repo / ".loci.json").write_text(payload, encoding="utf-8")
        assert {p.name for p in subscopes(repo, markers=True)} == {"glasses"}, \
            f"the marker-found sub-scope was lost to {payload!r}"


def test_discover_registers_the_parent_and_its_subscopes(tmp_path):
    from loci.scopes import discover

    repo = tmp_path / "Delroy"
    (repo / ".git").mkdir(parents=True)
    (repo / "glasses").mkdir()
    (repo / "glasses" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "plain" / ".git").mkdir(parents=True)   # no sub-projects

    found = {s.id: s for s in discover([tmp_path], max_depth=2, markers=True)}
    assert "delroy" in found
    assert "delroy-glasses" in found
    assert found["delroy-glasses"].root == (repo / "glasses").resolve()
    assert "delroy" in found["delroy-glasses"].group_set(), \
        "a sub-scope must carry its parent as a containment label"
    # A containment group that excludes the container is the wrong shape: the
    # monorepo root still owns everything no sub-project claimed, and under a
    # hard group policy it would be the one scope ruled out.
    assert "delroy" in found["delroy"].group_set(), \
        "the container is not a member of its own containment group"
    # Tri-state: an ordinary repo was never inferred, which is not the same as
    # deliberately ungrouped, and `[]` here would make `groups infer` skip it.
    assert found["plain"].groups is None


def test_subscope_ids_never_collide(tmp_path):
    """Two repos can each hold a `client/`, and a top-level repo can be named
    the same as a sub-scope's generated id.
    """
    from loci.scopes import discover

    for parent in ("One", "Two"):
        p = tmp_path / parent
        (p / ".git").mkdir(parents=True)
        (p / "client").mkdir()
        (p / "client" / "package.json").write_text("{}", encoding="utf-8")
    # The head-on collision: a repo whose own id is what `One/client` generates.
    # Without it the two sub-scopes are `one-client` and `two-client` and every
    # assertion below passes with no uniquing at all.
    (tmp_path / "one-client" / ".git").mkdir(parents=True)

    ids = [s.id for s in discover([tmp_path], max_depth=2, markers=True)]
    assert len(ids) == len(set(ids)), f"duplicate scope ids: {ids}"
    assert {"one-client", "two-client"} <= set(ids)
    # The real repository wins its own id; the generated one takes the suffix.
    # Filesystem order decided this before, and `upsert` matches on id while
    # `root` is not preserved -- so the loser's root would be written into the
    # winner's registry entry, under the winner's aliases and groups.
    by_id = {s.id: s for s in discover([tmp_path], max_depth=2, markers=True)}
    assert by_id["one-client"].root == (tmp_path / "one-client").resolve()
    assert by_id["one-client-2"].root == (tmp_path / "One" / "client").resolve()


def test_a_rescan_cannot_erase_a_subscopes_containment_label(tmp_path, monkeypatch):
    """Containment is structural, so a user's own grouping must add to it rather
    than replace it. `upsert` preserved a stored `groups` wholesale, which meant
    the first `group set` on a sub-scope silently dropped its parent label the
    next time `scan` ran -- and nothing would ever put it back.

    `me` rides along on the first scan because `scan` now labels what it
    registers with the provenance it inferred from that scope's own git. It
    does NOT come back after the user's edit wipes it: the re-scan finds nothing
    fresh, so nothing is re-classified, and provenance is never inherited from a
    container. The sets are pinned by equality: the point is that nothing is
    LOST, and a subset check would not notice a loss.
    """
    from argparse import Namespace
    from dataclasses import replace

    import loci.paths as P
    from loci.cli import cmd_scan
    from loci.scopes import load_scopes, save_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    repo = tmp_path / "Delroy"
    (repo / ".git").mkdir(parents=True)
    (repo / "glasses").mkdir()
    (repo / "glasses" / "package.json").write_text("{}", encoding="utf-8")

    cmd_scan(Namespace(roots=[str(repo)], depth=1, split=True))
    assert {s.id for s in load_scopes()} == {"delroy", "delroy-glasses"}
    assert next(s for s in load_scopes() if s.id == "delroy-glasses").group_set() \
        == {"delroy", "me"}

    # the user groups the sub-project by hand, naming nothing about containment
    save_scopes([replace(s, groups=["client:acme"]) if s.id == "delroy-glasses"
                 else s for s in load_scopes()])
    cmd_scan(Namespace(roots=[str(repo)], depth=1, split=True))

    sub = next(s for s in load_scopes() if s.id == "delroy-glasses")
    assert sub.group_set() == {"delroy", "client:acme"}, \
        "a re-scan dropped a label the user's edit never mentioned"


def test_a_monorepo_is_one_scope_until_the_split_is_asked_for(
        tmp_path, monkeypatch, capsys):
    """Marker splitting is OFF by default, because a new scope's aliases include
    its bare directory name and `ALIAS_BOOST` (6.0) beats `CWD_BOOST` (4.0):
    measured on the real corpus, splitting a repo holding `glasses/` sent 8/8
    hand-written questions about a DIFFERENT project to `Delroy/glasses`, 7 of
    which had routed correctly before. `--split` opts back in, and both
    directions are pinned here -- a default flipped to True fails the first
    block, a flag parsed and dropped fails the second.

    Through the CLI rather than `discover`, because the flag has to survive the
    parser and `cmd_scan` as well as reach `subscopes`.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    repo = tmp_path / "corpus" / "Delroy"
    (repo / ".git").mkdir(parents=True)
    for name in ("glasses", "extension"):
        (repo / name).mkdir()
        (repo / name / "package.json").write_text("{}", encoding="utf-8")

    assert main(["scan", str(tmp_path / "corpus")]) == 0
    assert {s.id for s in load_scopes()} == {"delroy"}, \
        "workspace markers split a monorepo with nobody asking"

    assert main(["scan", "--split", str(tmp_path / "corpus")]) == 0
    assert {s.id for s in load_scopes()} == {"delroy", "delroy-glasses",
                                            "delroy-extension"}
    # The machinery is intact, not merely reachable: the containment labels the
    # exclusion threading and `--group` both read are still written.
    by_id = {s.id: s for s in load_scopes()}
    assert by_id["delroy-glasses"].group_set() >= {"delroy"}
    assert by_id["delroy"].group_set() >= {"delroy"}

    # And `--no-split` is not merely accepted -- it is the default's spelling.
    assert main(["scan", "--no-split", str(tmp_path / "corpus")]) == 0
    assert "delroy-glasses" in {s.id for s in load_scopes()}, \
        "a scope already registered was dropped by a later plain scan"


def test_a_declaration_splits_whether_or_not_the_flag_is_passed(tmp_path):
    """Writing `.loci.json` is the user asserting "these are separate projects"
    about one repository they know; a workspace marker is loci guessing it about
    every repository in the corpus. Only the guess is held back, so a repo
    carrying both a declaration and a marker splits on the declaration alone
    with the flag off, and on both with it on.

    The marker directory is what makes this discriminating: honouring
    declarations by simply leaving `subscopes` alone would put `glasses` in the
    first set too.
    """
    from loci.scopes import discover, subscopes

    repo = tmp_path / "mono"
    (repo / ".git").mkdir(parents=True)
    (repo / "client").mkdir()
    (repo / "glasses").mkdir()
    (repo / "glasses" / "package.json").write_text("{}", encoding="utf-8")
    (repo / ".loci.json").write_text(
        json.dumps({"scopes": [{"path": "client"}]}), encoding="utf-8")

    assert {p.name for p in subscopes(repo)} == {"client"}
    assert {p.name for p in subscopes(repo, markers=True)} == {"client", "glasses"}

    assert {s.id for s in discover([tmp_path], max_depth=2)} == {"mono", "mono-client"}
    assert {s.id for s in discover([tmp_path], max_depth=2, markers=True)} \
        == {"mono", "mono-client", "mono-glasses"}


def test_the_split_flag_is_parsed_and_passed_on_by_both_commands(tmp_path,
                                                                 monkeypatch):
    """`setup` reaches `discover` through `setup.run`, so nothing above proves
    `cmd_setup` forwards the flag rather than parsing it and dropping it -- the
    failure mode is a `loci setup --split` that silently does not split.
    """
    import loci.setup as S
    from loci.cli import build_parser, main

    p = build_parser()
    assert p.parse_args(["scan", "x"]).split is False
    assert p.parse_args(["scan", "--split", "x"]).split is True
    assert p.parse_args(["scan", "--no-split", "x"]).split is False
    assert p.parse_args(["setup", "x"]).split is False
    assert p.parse_args(["setup", "--split", "x"]).split is True

    seen: list = []
    monkeypatch.setattr(S, "run", lambda *a, **kw: seen.append(kw.get("split")) or 0)
    assert main(["setup", "--split", str(tmp_path)]) == 0
    assert main(["setup", str(tmp_path)]) == 0
    assert seen == [True, False]


# -- scopes: exclusion -----------------------------------------------------
def test_nested_roots_finds_subscopes_of_one_scope(tmp_path):
    from loci.scopes import nested_roots

    parent = Scope(id="mono", name="mono", root=tmp_path / "mono")
    child = Scope(id="mono-a", name="mono/a", root=tmp_path / "mono" / "a")
    sibling = Scope(id="other", name="other", root=tmp_path / "other")
    # A sibling whose name merely STARTS WITH the parent's. Containment is by
    # path component, not by string prefix: `str(root).startswith(...)` swallows
    # `mono-old` whole, and the scope silently stops being collected at all.
    prefix = Scope(id="mono-old", name="mono-old", root=tmp_path / "mono-old")

    everyone = [parent, child, sibling, prefix]
    assert nested_roots(parent, everyone) == [tmp_path / "mono" / "a"]
    assert nested_roots(child, everyone) == []
    assert nested_roots(prefix, everyone) == []


def test_iter_files_does_not_descend_into_an_excluded_root(tmp_path, monkeypatch):
    """Pruned, not filtered. Asserting only on the returned list cannot tell the
    two apart, and the difference is the entire reason this module exists:
    descending a sub-scope's node_modules to discard the result afterwards is
    the 32.9s it was written to avoid. So the walk itself is watched.
    """
    import os

    import loci.walk as walk_mod
    from loci.walk import iter_files

    root = tmp_path / "mono"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.md").write_text("a", encoding="utf-8")
    (root / "glasses" / "docs").mkdir(parents=True)
    (root / "glasses" / "docs" / "b.md").write_text("b", encoding="utf-8")

    everything = iter_files(root, ["**/*.md"])
    assert len(everything) == 2

    visited: list[Path] = []
    real_walk = os.walk

    def spy(top, **kw):
        # The yielded `dirnames` list is passed on as the same object, so the
        # pruning `iter_files` performs still reaches the real walk.
        for dirpath, dirnames, filenames in real_walk(top, **kw):
            visited.append(Path(dirpath))
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(walk_mod.os, "walk", spy)
    pruned = iter_files(root, ["**/*.md"], exclude=[root / "glasses"])

    assert [p.name for p in pruned] == ["a.md"]
    # Without this the assertion below passes vacuously whenever the spy fails
    # to take effect, which is the one way it could stop testing anything.
    assert root in visited, "the spy never observed the walk"
    excluded = root / "glasses"
    assert not [p for p in visited if p == excluded or excluded in p.parents], \
        f"descended into an excluded subtree: {visited}"


def test_a_parent_and_its_subscope_share_no_chunk(tmp_path):
    """The same file collected twice inflates two vocabularies with identical
    tokens and breaks the size prior for both.

    Code, not prose. The default episode globs are root-relative (`README*`,
    `*.md`), so a sub-scope's markdown never reaches its parent's walk at all
    and asserting on it would pass with no exclusion whatsoever. `**/*.py` is
    what actually crosses the boundary, and docstrings are the duplication this
    exists to stop -- hence the control assertion at the end.
    """
    from loci.backends.episodes import BuiltinEpisodeBackend

    root = tmp_path / "mono"
    (root / "glasses").mkdir(parents=True)
    (root / "app.py").write_text(
        '"""Parent prose, long enough to clear the twelve word minimum that the '
        'docstring collector applies to every candidate."""\n', encoding="utf-8")
    (root / "glasses" / "lens.py").write_text(
        '"""Child prose, long enough to clear the twelve word minimum that the '
        'docstring collector applies to every candidate."""\n', encoding="utf-8")

    eb = BuiltinEpisodeBackend()
    parent = make_scope(root, name="mono")
    child = make_scope(root / "glasses", name="mono/glasses")

    parent_chunks = eb.collect(parent, exclude=[root / "glasses"])
    child_chunks = eb.collect(child)

    assert any("Parent prose" in c.text for c in parent_chunks)
    assert not any("Child prose" in c.text for c in parent_chunks)
    assert any("Child prose" in c.text for c in child_chunks)
    assert any("Child prose" in c.text for c in eb.collect(parent)), \
        "control: unexcluded, the parent must reach the sub-scope's code"


def test_parent_fingerprint_is_stable_when_only_a_subscope_changes(tmp_path):
    """Otherwise the parent re-parses thousands of files every time a sub-scope
    is touched, and `loci index` stops being idempotent in practice.
    """
    import time

    from loci.backends import get_structure_backend
    from loci.index import fingerprint

    root = tmp_path / "mono"
    (root / "glasses").mkdir(parents=True)
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "glasses" / "lens.py").write_text("y = 1\n", encoding="utf-8")

    sb = get_structure_backend()
    parent = make_scope(root, name="mono")
    excl = [root / "glasses"]

    before = fingerprint(parent, sb, exclude=excl)
    unfiltered_before = fingerprint(parent, sb)
    time.sleep(1.1)                      # mtime has one-second resolution
    (root / "glasses" / "lens.py").write_text("y = 2\n", encoding="utf-8")

    assert fingerprint(parent, sb, exclude=excl) == before
    assert fingerprint(parent, sb) != unfiltered_before, \
        "control: unexcluded, the sub-scope's file must move the fingerprint"


def test_a_parent_graph_does_not_lend_its_subscope_symbols(tmp_path):
    """Where the routing corruption actually lands: a monorepo's graph covers
    the whole tree, so without a read-time filter the parent's vocabulary, node
    count and symbol labels all carry its sub-scopes' code -- and the size prior
    that is meant to make scopes comparable is computed from the inflated count.

    Filtered at read time, not re-extracted: the nodes already carry
    `source_file`, and it may be relative to the root or absolute.

    `glasses-old/` is the string-prefix trap. Containment is by path component,
    so a directory whose NAME merely starts with an excluded one keeps its
    nodes; `source_file.startswith(str(excluded))` would silently delete them.
    """
    from loci.backends.graphify import GraphifyBackend

    root = tmp_path / "mono"
    (root / "graphify-out").mkdir(parents=True)
    (root / "graphify-out" / "graph.json").write_text(json.dumps({"nodes": [
        {"label": "parentWidget", "source_file": "src/app.py"},
        {"label": "childLens", "source_file": "glasses/src/lens.ts"},
        {"label": "childAbsolute", "source_file": str(root / "glasses" / "b.ts")},
        {"label": "legacyWidget", "source_file": "glasses-old/src/legacy.ts"},
    ]}), encoding="utf-8")

    sb = GraphifyBackend()
    scope = make_scope(root, name="mono")
    excl = [root / "glasses"]

    assert sb.labels(scope) == ["parentWidget", "childLens", "childAbsolute",
                                "legacyWidget"]
    assert sb.labels(scope, exclude=excl) == ["parentWidget", "legacyWidget"]
    assert sb.node_count(scope) == 4
    assert sb.node_count(scope, exclude=excl) == 2
    assert "lens" in sb.vocabulary(scope), "control: unexcluded, the token is there"
    assert "lens" not in sb.vocabulary(scope, exclude=excl)
    assert "legacy" in sb.vocabulary(scope, exclude=excl), \
        "a sibling whose name starts with the excluded one was swallowed"


def _monorepo(tmp_path):
    """A parent scope holding one sub-scope: prose and code in each, and a
    single graph at the parent covering the whole tree -- which is what
    graphify actually produces for a monorepo. Returns (parent, child).
    """
    root = tmp_path / "mono"
    (root / "glasses").mkdir(parents=True)
    (root / "README.md").write_text(
        "# mono\n\n## Settlement\n\nReconciles invoices against bank statements "
        "nightly and files any discrepancy as a dunning notice for review.\n",
        encoding="utf-8")
    (root / "app.py").write_text(
        '"""Parent prose, long enough to clear the twelve word minimum that the '
        'docstring collector applies to every candidate."""\n', encoding="utf-8")
    (root / "glasses" / "README.md").write_text(
        "# glasses\n\n## Windowing\n\nApplies radiograph windowing to DICOM voxel "
        "data before handing it to the viewer, so contrast stays stable.\n",
        encoding="utf-8")
    (root / "glasses" / "lens.py").write_text(
        '"""Child prose, long enough to clear the twelve word minimum that the '
        'docstring collector applies to every candidate."""\n', encoding="utf-8")
    (root / "graphify-out").mkdir()
    (root / "graphify-out" / "graph.json").write_text(json.dumps({"nodes": [
        {"label": "parentWidget", "source_file": "app.py"},
        {"label": "childLens", "source_file": "glasses/lens.py"},
    ]}), encoding="utf-8")
    return (make_scope(root, name="mono"),
            make_scope(root / "glasses", name="mono/glasses"))


def test_build_gives_a_subscopes_content_to_the_subscope_alone(tmp_path, monkeypatch):
    """`build` is the seam this exclusion exists for, and the one place a
    missing `exclude=` is invisible: every reader below it defaults to
    collecting everything, so dropping the argument restores the duplication
    silently instead of raising. All four call sites are asserted here.
    """
    import loci.paths as P
    from loci.backends.graphify import GraphifyBackend
    from loci.index import build, load_episodes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    # The readers below never shell out; only `available` gates them, and a
    # machine without the CLI would otherwise skip the whole structure path.
    monkeypatch.setattr(GraphifyBackend, "available", lambda self: True)
    parent, child = _monorepo(tmp_path)

    idx = build([parent, child], verbose=False)
    store = load_episodes()

    def sources(sid):
        return {c["source"] for c in store["chunks"][sid]
                if not c["source"].startswith("git:")}

    # vocabulary, and the node count the size prior is computed from
    assert parent.id in idx["postings"].get("widget", {}), \
        "control: the parent's own graph was never read"
    assert parent.id not in idx["postings"].get("lens", {})
    assert idx["scopes"][parent.id]["structure_nodes"] == 1

    # collect
    assert "code:glasses/lens.py" not in sources(parent.id), \
        "the parent collected a file its sub-scope already owns"
    assert sources(parent.id) == {"code:app.py", "doc:README.md"}
    assert "code:lens.py" in sources(child.id), \
        "control: the sub-scope must own that file itself"

    # fingerprint. A NEW file rather than an edited one, because the signature
    # stats mtime at one-second resolution and a path is hashed immediately.
    (child.root / "extra.py").write_text("x = 1\n", encoding="utf-8")
    again = build([parent, child], verbose=False)
    assert again["scopes"][parent.id]["fingerprint"] \
        == idx["scopes"][parent.id]["fingerprint"]
    assert again["scopes"][child.id]["fingerprint"] \
        != idx["scopes"][child.id]["fingerprint"], \
        "control: a new file must move the fingerprint of the scope owning it"


def test_symbol_seeding_never_encodes_a_subscopes_labels(tmp_path, monkeypatch):
    """The graph's second reader. `build_embeddings` seeds semantic search from
    symbol labels, so without the same exclusion a parent answers questions
    about its sub-project out of its own vectors -- the failure `build` was just
    fixed for, reintroduced one function later.

    The encoder is stubbed: sentence-transformers is an optional extra, and what
    matters is WHICH labels were selected, never what a model made of them.
    """
    import sys
    import types

    import loci.paths as P
    from loci.backends.graphify import GraphifyBackend
    from loci.index import build, build_embeddings
    from loci.scopes import save_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(GraphifyBackend, "available", lambda self: True)
    parent, child = _monorepo(tmp_path)
    build([parent, child], verbose=False)
    save_scopes([parent, child])

    class _Encoder:
        def __init__(self, *a, **kw):
            pass

        def encode(self, texts, **kw):
            return [[float(len(t)), 0.0] for t in texts]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _Encoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    build_embeddings(verbose=False)

    seeded = json.loads((P.embeddings_file().parent / f".symbols-{parent.id}.json")
                        .read_text(encoding="utf-8"))
    assert seeded == ["parentWidget"], \
        "the parent seeded semantic search from its sub-scope's symbols"


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
    # The live constant, never a literal: a fixture pinned to a stale version
    # fails the load gate instead of testing what it was written to test.
    return {"version": INDEX_VERSION, "scopes": meta, "postings": postings}


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


def test_the_concentrated_tier_routes_when_its_owner_is_not_the_top_scope():
    """The in-group evidence gate asks whether ANY eligible scope holds a
    concentrated token, not whether the TOP one does, and the difference is the
    whole point of the tier: a question about something two projects share
    splits its evidence between them, so the owner of the shared term routinely
    ranks below a scope that merely uses the common word a lot.

    Here `shared` is everywhere and prominent in Alpha, which therefore wins;
    `wrangler` lives only in Beta. The floor and the count gate are pushed out
    of reach so concentration is the only thing that can route this at all.
    Narrowing the gate to `top in concentrated` -- which reads tidier, and which
    the rest of the suite does not catch -- abstains with `no_evidence` and
    drops Beta, which is the failure the tier exists to fix.
    """
    idx = _index(
        a=("Alpha", "/a", {"shared": 900}, 10),
        b=("Beta", "/b", {"wrangler": 75, "shared": 2}, 2000),
        c=("Gamma", "/c", {"shared": 3}, 400),
        d=("Delta", "/d", {"shared": 3}, 400),
    )
    r = route("shared wrangler", idx, evidence_floor=10 ** 6, min_matched=99)

    assert r.ranked[0] == "a", "fixture lost its point: the owner must be the runner-up"
    assert not r.abstain, f"abstained {r.abstain_reason} on a contended question"
    assert "b" in r.selected, "the owner of the concentrated token was dropped"


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


def test_only_the_deepest_containing_scope_takes_the_cwd_boost(tmp_path):
    """CWD_BOOST went to EVERY scope whose root contains cwd. On a flat corpus
    that is the same thing; on a split monorepo it is not -- parent and child
    both took the identical 4.0, so cwd stopped discriminating between them at
    all and the parent won on vocabulary, which it always does: it is the larger
    index. Measured from inside `Delroy/glasses`, a deictic question scored
    Delroy 5.17 against Delroy/glasses 4.91 and answered from the parent.

    The evidence here is deliberately in the band measured on the real corpus
    (0.1-1.5): the point is that a 4.0 held by both sides decides nothing, not
    that vocabulary is weak.
    """
    import loci.router as R

    parent = tmp_path / "mono"
    child = parent / "glasses"
    child.mkdir(parents=True)
    idx = _index(
        mono=("Delroy", str(parent),
              {"deployed": 40, "widget": 30, "sprocket": 20}, 50000),
        glasses=("Delroy/glasses", str(child), {"lens": 3}, 40),
    )
    r = route("how is this deployed?", idx, cwd=child)

    assert r.detail["glasses"]["signals"].get("cwd") == str(child)
    assert "cwd" not in r.detail["mono"]["signals"], \
        "the container took the boost too, so cwd separated nothing"
    assert not r.abstain and r.selected == ["glasses"]
    assert r.detail["mono"]["score"] < R.CWD_BOOST, \
        "the parent is only ahead because it was boosted, not on evidence"

    # The rule is `scope_for_cwd`'s, which resolves cwd to the deepest scope and
    # documents that it does. Router and registry must not disagree about which
    # scope you are standing in.
    registry = [Scope(id="mono", name="Delroy", root=parent),
                Scope(id="glasses", name="Delroy/glasses", root=child)]
    assert scope_for_cwd(registry, child).id == "glasses"


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


def test_out_of_group_asks_whether_THAT_scope_is_concentrated_not_whether_any_is():
    """The mirror of the in-group gate, and deliberately not the same question.

    `out_of_group` judges ONE scope -- the corpus-wide winner -- so it asks
    whether that scope holds a concentrated token. `bool(concentrated_owners)`
    there fires on a token the scope being judged does not hold, and since the
    concentrated tier exists for terms two projects SHARE, the token is often
    held by the in-group scope that is about to answer. A hard group would then
    refuse the question on the strength of its own evidence.

    Alpha wins the corpus on a common word alone; Beta, inside the group, holds
    `wrangler`. The floor and count gate are out of reach, so Beta routes on
    concentration and nothing about Alpha says the answer is elsewhere. Four
    scopes, not two: `shared` has to exceed CONCENTRATED_SCOPES or Alpha is a
    concentrated owner in its own right and the two readings agree by accident.
    """
    idx = _index(
        a=("Alpha", "/a", {"shared": 900}, 10),
        b=("Beta", "/b", {"wrangler": 75, "shared": 2}, 2000),
        c=("Gamma", "/c", {"shared": 3}, 400),
        d=("Delta", "/d", {"shared": 3}, 400),
    )
    r = route("shared wrangler", idx, eligible={"b"}, strict_group=True,
              evidence_floor=10 ** 6, min_matched=99)

    assert r.detail["a"]["signals"] == {}, "fixture lost its point"
    assert not r.abstain, f"a hard group refused on its own evidence: {r.abstain_reason}"
    assert r.selected == ["b"]


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


# -- concurrency -----------------------------------------------------------
def test_the_embedding_model_is_built_exactly_once_under_threads():
    """`ask` fans out across scopes, so every lazy init is reachable from
    several threads at once.

    Unguarded, `if _MODEL is None: _MODEL = SentenceTransformer(...)` is a
    check-then-act race, and two threads constructing one concurrently crashed
    the process -- SIGSEGV, SIGABRT and an indefinite hang on different runs of
    the same question.
    """
    import sys
    import threading
    import types

    from loci.backends import episodes as ep

    built = []
    slow = threading.Event()

    class FakeModel:
        def __init__(self, name):
            built.append(name)
            slow.wait(0.5)          # widen the window the race needs
        def encode(self, texts, **kw):
            return [[0.0, 1.0] for _ in texts]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel
    fake.CrossEncoder = object
    prev_mod = sys.modules.get("sentence_transformers")
    prev_model = ep._MODEL
    sys.modules["sentence_transformers"] = fake
    ep._MODEL = None
    try:
        errors = []
        def call():
            try:
                ep._encode_query("hello", "bge-small")
            except Exception as exc:      # pragma: no cover - failure path
                errors.append(exc)
        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        slow.set()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "lazy model init deadlocked"
        assert not errors, errors
        assert len(built) == 1, f"model constructed {len(built)} times, not once"
    finally:
        ep._MODEL = prev_model
        if prev_mod is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = prev_mod


def test_ask_warms_the_model_before_fanning_out_over_scopes(monkeypatch):
    """Native init must happen on ONE thread, before the pool starts.

    The lock above makes concurrent construction safe; this makes it
    unnecessary. Measured, the crash only ever happened on a cold process with
    more than one scope selected.
    """
    from loci import ask as ask_mod
    from loci.backends import episodes as ep

    calls = []
    monkeypatch.setattr(ep, "warm_up",
                        lambda store=None, scope_ids=None: calls.append(list(scope_ids or [])))
    idx = _index(a=("A", "/a", {"alpha": 30}, 100),
                 b=("B", "/b", {"beta": 20}, 100))
    store = {"chunks": {}}
    ask_mod.ask("alpha and beta", index=idx, store=store,
                force_scopes=["a", "b"], with_structure=False)
    assert calls == [["a", "b"]], "ask did not warm before the fan-out"

    calls.clear()
    ask_mod.ask("alpha", index=idx, store=store, force_scopes=["a"],
                with_structure=False)
    assert calls == [], "a single scope has nothing to race with"


# -- enumerative set mode --------------------------------------------------
def test_is_enumerative_detects_the_closed_class():
    from loci.router import is_enumerative
    for q in ("which of my projects use Cloudflare workers?",
              "which projects deploy to Fly?",
              "what repos have a Dockerfile?",
              "do any of my projects use Postgres?",
              "where else is this pattern used?",
              "across all my projects, what handles auth?",
              "have I ever written a ZIM parser?"):
        assert is_enumerative(q), q
    # Singular questions, and questions that merely mention a project, are NOT
    # enumerative. Set mode drops the widen ratio, so a false positive here
    # widens every ordinary question.
    for q in ("how does this project start up?",
              "which function parses the arguments?",
              "why was the session cookie dropped?",
              "what does the project depend on?"):
        assert not is_enumerative(q), q


def test_an_enumerative_question_returns_every_owner_not_the_top_one():
    """The ratio cutoff measures distance from the top scope, which for a set
    question is merely whichever owner says the term most often."""
    idx = _index(
        loud=("Loud", "/l", {"cloudflare": 40, "workers": 60, "deploy": 20}, 200),
        quiet=("Quiet", "/q", {"cloudflare": 5, "workers": 2, "deploy": 3}, 700),
        other=("Other", "/o", {"handler": 900, "config": 800}, 9000))
    q = "which of my projects deploy to cloudflare workers?"
    # Widening from the top scope keeps only the loud owner...
    narrow = route(q, idx, evidence_floor=1.0, set_floor_ratio=0.0001,
                   max_set_scopes=1)
    assert narrow.selected == ["loud"]
    # ...while set mode returns both, because each clears the floor on its own.
    r = route(q, idx, evidence_floor=1.0)
    assert r.enumerative and not r.abstain
    assert set(r.selected) == {"loud", "quiet"}
    assert "other" not in r.selected


def test_the_enumerative_frame_is_not_treated_as_vocabulary():
    """`projects` is scaffolding, not evidence -- the deixis insight one level
    down. In the development corpus it was held by exactly two scopes, so it
    scored as discriminative FOR them and dragged both into every answer."""
    from loci.router import ENUM_FRAME
    idx = _index(
        meta=("Meta", "/m", {"projects": 90, "repos": 40}, 300),
        real=("Real", "/r", {"wrangler": 50, "cloudflare": 30}, 300))
    r = route("which of my projects use wrangler?", idx, evidence_floor=1.0)
    assert "projects" not in r.query_tokens
    assert r.selected == ["real"], "the frame noun elected its own holder"
    assert {"projects", "repos", "codebase"} <= ENUM_FRAME


def test_enumeration_outranks_deixis():
    """"where else does this appear" points AND asks across the corpus.

    Deixis abstains because a pointing question gives no way to pick ONE scope.
    An enumerative question is not picking one, so the rule does not apply."""
    idx = _index(a=("A", "/a", {"wrangler": 50}, 300),
                 b=("B", "/b", {"wrangler": 40}, 300))
    r = route("where else is this wrangler config used?", idx, evidence_floor=1.0)
    assert not r.abstain and r.abstain_reason is None
    assert set(r.selected) == {"a", "b"}


def test_enumerative_mode_still_abstains_without_evidence():
    """Set mode widens what is returned, never whether to answer at all."""
    idx = _index(a=("A", "/a", {"wrangler": 50}, 300),
                 b=("B", "/b", {"handler": 900}, 9000))
    r = route("which of my projects use Kubernetes ingress controllers?", idx)
    assert r.abstain and r.abstain_reason == "no_evidence"


def test_set_mode_thresholds_are_read_at_call_time():
    import loci.router as R
    idx = _index(a=("A", "/a", {"wrangler": 50, "cloudflare": 30}, 300),
                 b=("B", "/b", {"wrangler": 40, "cloudflare": 20}, 300))
    q = "which projects use wrangler and cloudflare?"
    before = R.MAX_SET_SCOPES
    try:
        R.MAX_SET_SCOPES = 1
        assert len(route(q, idx, evidence_floor=1.0).selected) == 1
    finally:
        R.MAX_SET_SCOPES = before
    assert len(route(q, idx, evidence_floor=1.0).selected) == 2


def test_corroboration_weight_ships_inert():
    """1.0 is a plain sum -- the scoring the router had before the knob existed.

    It is deliberately not enabled: the one thing it improves is a single
    negative-family item on a single corpus, and the synthetic bed cannot
    arbitrate because `size_skew` varies file volume rather than vocabulary
    breadth. Changing this default without first reproducing the failure on a
    generated corpus is fitting to one item.
    """
    from loci.router import CORROBORATION_WEIGHT
    assert CORROBORATION_WEIGHT == 1.0
    # ...and at 1.0 the blend must be arithmetically identical to the sum.
    idx = _index(a=("A", "/a", {"alpha": 30, "beta": 20, "gamma": 10}, 200),
                 b=("B", "/b", {"alpha": 5, "delta": 40}, 400))
    q = "how do alpha beta and gamma relate to delta?"
    assert (route(q, idx, evidence_floor=1.0).detail
            == route(q, idx, evidence_floor=1.0,
                     corroboration_weight=1.0).detail)


def test_set_mode_constants_stay_at_their_swept_values():
    """0.8 is the top of a flat region, not a sharp optimum.

    Confusable gold-coverage is flat at 83.3% from 0.6 to 0.8 and drops to
    66.7% at 0.9; precision peaks at 0.8 and decays below 0.7. Moving this
    without re-running the sweep in evals/RESULTS.md is how a fitted constant
    silently becomes a guess.
    """
    from loci.router import MAX_SET_SCOPES, SET_FLOOR_RATIO
    assert SET_FLOOR_RATIO == 0.8
    assert MAX_SET_SCOPES == 8


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
    import time

    from loci.paths import atomic_write

    target = tmp_path / "big.json"
    payload = json.dumps({"k": {str(i): "x" * 200 for i in range(8000)}})
    torn, done, failed = [0], [False], []

    # `done` is set from a finally and the reader carries its own deadline
    # because the first version of this test could not fail: when the writer
    # died -- which is what Windows did, PermissionError out of os.replace
    # against a destination the reader held open -- `done` stayed False and the
    # reader spun until GitHub killed the job at its six-hour ceiling. A test
    # that hangs on a real bug reports nothing at all.
    def writer():
        try:
            for _ in range(10):
                atomic_write(target, payload)
        except BaseException as exc:  # recorded, then asserted on below
            failed.append(exc)
        finally:
            done[0] = True

    def reader():
        deadline = time.monotonic() + 60
        while not done[0] and time.monotonic() < deadline:
            try:
                if target.exists():
                    json.loads(target.read_text())
            except json.JSONDecodeError:
                torn[0] += 1
            except (FileNotFoundError, UnicodeDecodeError, PermissionError):
                pass

    # Daemon threads with bounded joins: whatever goes wrong from here, the
    # test fails in two minutes instead of pinning a runner until it is killed.
    w = threading.Thread(target=writer, daemon=True)
    r = threading.Thread(target=reader, daemon=True)
    w.start(), r.start(), w.join(120), r.join(120)
    assert not w.is_alive() and not r.is_alive(), "the durable write never returned"
    assert not failed, f"the durable write itself failed: {failed[0]!r}"
    assert torn[0] == 0


def test_a_durable_write_retries_a_rename_windows_refuses(tmp_path, monkeypatch):
    """Windows will not replace a destination another handle has open.

    Simulated, because it cannot be provoked on POSIX -- which is the whole
    problem: the bug this guards hung every Windows CI job to its six-hour
    ceiling while Linux and macOS stayed green across four Python versions.
    """
    import loci.paths as P

    target = tmp_path / "store.json"
    real, calls = P.os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(32, "being used by another process")
        real(src, dst)

    monkeypatch.setattr(P.os, "name", "nt")
    monkeypatch.setattr(P.os, "replace", flaky)
    P.atomic_write(target, '{"ok": true}')

    assert json.loads(target.read_text()) == {"ok": True}
    assert len(calls) == 3, "the rename was not retried"


def test_a_durable_write_reports_a_real_permission_error_at_once(tmp_path, monkeypatch):
    """Off Windows the same exception means what it says, so retrying only delays it."""
    import loci.paths as P

    calls = []

    def denied(src, dst):
        calls.append(1)
        raise PermissionError(13, "permission denied")

    monkeypatch.setattr(P.os, "name", "posix")
    monkeypatch.setattr(P.os, "replace", denied)
    with pytest.raises(PermissionError):
        P.atomic_write(tmp_path / "store.json", "{}")

    assert calls == [1]


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


# -- update ----------------------------------------------------------------
def test_update_refuses_an_empty_registry_and_names_setup(tmp_path, monkeypatch,
                                                          capsys):
    """`update` on a machine with nothing registered is a first run in disguise.

    It must not traceback its way through five steps that all have nothing to
    do; it must say which command the user actually wanted.
    """
    import loci.paths as P
    from loci.update import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    assert run() == 1
    assert "loci setup" in capsys.readouterr().out


def test_update_rebuilds_every_graph_not_only_the_missing_ones(tmp_path,
                                                               monkeypatch):
    """This is the entire reason the command exists.

    `loci graphs` skips a scope that already has a graph and `loci setup` does
    the same unless forced, so a project whose code moved keeps routing on the
    symbols it had the day it was registered -- silently, since nothing else
    reports it. If `update` ever inherits that filter it stops being an update.
    """
    import loci.paths as P
    import loci.setup as S
    from loci.update import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    _fake_repo(corpus, "alpha", "Signs the admin session cookie with a rotating "
                                "HMAC key and refuses SameSite None on localhost.")
    _fake_repo(corpus, "beta", "Compresses ZIM archives with zstd and streams "
                               "them to object storage from a manifest table.")
    S.run([corpus], assume_yes=True, graphs=False, embed=False, calibrate=False)

    seen: list = []
    monkeypatch.setattr(S, "build_graphs",
                        lambda sb, scopes, **kw: seen.extend(s.name for s in scopes) or [])
    # A graph backend that claims every scope already has one: the filter this
    # test exists to catch would read exactly this and build nothing.
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/graphify")
    from loci.backends import get_structure_backend
    sb = get_structure_backend()
    monkeypatch.setattr(type(sb), "sources",
                        lambda self, sc: [{"kind": "code", "nodes": 10, "path": "x"}])
    monkeypatch.setattr("loci.backends.get_structure_backend", lambda *a, **k: sb)

    run(assume_yes=True, scan=False, embed=False, calibrate=False)
    assert sorted(seen) == ["alpha", "beta"]


def test_update_finds_a_project_created_after_the_last_scan(tmp_path, monkeypatch):
    """The registry records scope roots; none of them answers "where would a
    NEW repository appear?". Without the roots a scan was pointed at, `update`
    with no arguments is blind to exactly the case the user runs it for.
    """
    import loci.paths as P
    import loci.setup as S
    from loci.scopes import load_scopes
    from loci.update import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    _fake_repo(corpus, "alpha", "Signs the admin session cookie with a rotating "
                                "HMAC key and refuses SameSite None on localhost.")
    S.run([corpus], assume_yes=True, graphs=False, embed=False, calibrate=False)

    _fake_repo(corpus, "gamma", "Parses wrangler bindings and provisions D1 "
                                "databases from a declarative manifest file.")
    run(assume_yes=True, graphs=False, embed=False, calibrate=False)

    assert {s.id for s in load_scopes()} == {"alpha", "gamma"}


def test_a_registry_rewrite_that_scanned_nothing_keeps_the_scan_roots(tmp_path,
                                                                     monkeypatch):
    """`group add`, `add` and `groups infer` all rewrite the registry wholesale
    without scanning anything. If that dropped the recorded roots, the next
    `loci update` would silently stop looking for new projects -- and the only
    symptom is a repository that never turns up.
    """
    import loci.paths as P
    from loci.scopes import load_roots, load_scopes, save_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    root = tmp_path / "corpus"
    root.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=tmp_path / "a")], roots=[root])
    assert load_roots() == [root]

    save_scopes(load_scopes())            # what every non-scanning caller does
    assert load_roots() == [root]


def test_update_does_not_opt_you_into_a_model_download(tmp_path, monkeypatch):
    """An update refreshes what you have. Someone who chose lexical search ran
    this command to keep that working, not to spend 130MB finding out that the
    default changed under them.
    """
    import loci.paths as P
    import loci.setup as S
    from loci.update import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    _fake_repo(corpus, "alpha", "Signs the admin session cookie with a rotating "
                                "HMAC key and refuses SameSite None on localhost.")
    S.run([corpus], assume_yes=True, graphs=False, embed=False, calibrate=False)

    import loci.index as I
    monkeypatch.setattr(I, "build_embeddings",
                        lambda *a, **k: pytest.fail("update started an encode "
                                                    "with no embeddings on disk"))
    run(assume_yes=True, graphs=False, calibrate=False)


# -- agent skill -----------------------------------------------------------
def test_the_skill_ships_inside_the_package():
    """It is read through importlib.resources, so a packaging mistake breaks
    `loci skill install` for everyone who pip-installed and for nobody working
    in a source checkout -- which is the definition of a silent regression.
    """
    from loci.agent_skill import skill_text

    text = skill_text()
    assert text.startswith("---\n")
    assert "name: loci" in text


def test_every_command_the_skill_documents_exists():
    """The skill tells an agent which commands to run. A renamed or dropped
    subcommand leaves it confidently instructing the model to run something
    that has not existed for two releases, and nothing else would catch it.
    """
    import argparse

    from loci.agent_skill import skill_text
    from loci.cli import build_parser

    sub = next(a for a in build_parser()._actions
               if isinstance(a, argparse._SubParsersAction))
    usage = skill_text().split("## Usage")[1].split("```")[1]
    documented = set()
    for line in usage.splitlines():
        parts = line.split()
        # `/loci "<question>"` and `/loci <anything else>` are the two entries
        # that name no subcommand: one is the bare-question path, the other is
        # the passthrough rule.
        if len(parts) >= 2 and parts[0] == "/loci" and parts[1].isalpha():
            documented.add(parts[1])
    assert documented, "the Usage block parsed to nothing"
    assert documented <= set(sub.choices), (
        f"documented but not a command: {sorted(documented - set(sub.choices))}")


def test_skill_install_is_idempotent_and_will_not_clobber_an_edit(tmp_path):
    """Someone who tuned the skill for their own workflow must not lose it to a
    routine `loci skill install` after an upgrade. An identical file is left
    alone too, so the mtime does not move for no reason.
    """
    from loci.agent_skill import install

    wrote, dest, _ = install(tmp_path)
    assert wrote and dest.is_file()

    wrote, _, detail = install(tmp_path)
    assert not wrote and detail == "already up to date"

    dest.write_text(dest.read_text(encoding="utf-8") + "\nmine\n", encoding="utf-8")
    wrote, _, detail = install(tmp_path)
    assert not wrote and "--force" in detail
    assert "mine" in dest.read_text(encoding="utf-8")

    wrote, _, _ = install(tmp_path, force=True)
    assert wrote and "mine" not in dest.read_text(encoding="utf-8")


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


# -- ask: confinement, end to end ------------------------------------------
def test_ask_confines_to_a_hard_group(tmp_path):
    """End-to-end: a hard group turns a would-be answer into an abstention that
    names the group, rather than into the in-group runner-up.

    Alpha's node count is deliberately 100 rather than the 500 the other router
    fixtures use. The cwd that anchors the group also hands Beta CWD_BOOST
    (4.0), and at 500 Alpha scores 3.54 -- so BETA wins the corpus, Beta is
    inside the group, and there is nothing out-of-group left to refuse. The
    fixture only tests what it claims while the out-of-group scope actually
    wins: 4.66 to 4.15.
    """
    from loci.ask import ask
    from loci.groups import Policy

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 100),
                   b=("Beta", str(b), {"flange": 25}, 400))
    registry = [Scope(id="a", name="Alpha", root=a, groups=["me"]),
                Scope(id="b", name="Beta", root=b, groups=["client:acme"])]
    policy = Policy(default_mode="soft", groups={"client:acme": "hard"})

    answer = ask("how does the widget gizmo sprocket work", cwd=str(b),
                 index=index, store={}, policy=policy, registry=registry,
                 with_structure=False)

    assert answer.routing.abstain
    assert answer.routing.abstain_reason == "out_of_group"
    assert answer.routing.group == "client:acme"
    assert answer.routing.mode == "hard"
    assert answer.scopes == [], "an abstention must not answer from the runner-up"


def test_ask_without_a_group_routes_exactly_as_before(tmp_path):
    """The upgrade guarantee, end to end.

    A policy with no groups and a registry with no memberships must produce the
    pre-change RouteResult EXACTLY, not merely a non-abstaining one: comparing
    `selected` alone would miss a group label, a mode, or a penalty leaking into
    the ungrouped path. cwd is passed so the anchor really is found and the
    inertness comes from the empty membership, not from never looking.
    """
    from loci.ask import ask
    from loci.groups import Policy

    a = tmp_path / "a"
    a.mkdir()
    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                   b=("Beta", "/b", {"flange": 25}, 400))
    q = "how does the widget gizmo sprocket work"

    answer = ask(q, cwd=str(a), index=index, store={},
                 policy=Policy(), registry=[Scope(id="a", name="Alpha", root=a)],
                 with_structure=False)

    assert not answer.routing.abstain
    assert answer.routing.selected == ["a"]
    assert answer.routing.group is None
    assert answer.routing.to_json() == route(q, index, cwd=str(a)).to_json()


def test_forcing_a_scope_still_bypasses_groups(tmp_path):
    """--scope is an explicit override and must outrank any group policy.

    cwd sits inside Beta, whose group is `hard`. Without the bypass the
    confinement is eligible={"b"} and routing returns Beta on its cwd boost, so
    Alpha here is reachable only through the bypass. Asked without a cwd there
    is nothing to confine, and both assertions hold with the bypass deleted --
    which is no test at all.
    """
    from loci.ask import ask
    from loci.groups import Policy

    b = tmp_path / "b"
    b.mkdir()
    index = _index(a=("Alpha", "/a", {"widget": 40}, 500),
                   b=("Beta", str(b), {"flange": 25}, 400))
    registry = [Scope(id="a", name="Alpha", root=Path("/a"), groups=["me"]),
                Scope(id="b", name="Beta", root=b, groups=["client:acme"])]

    answer = ask("how does the widget work", cwd=str(b), force_scopes=["a"],
                 index=index, store={},
                 policy=Policy(groups={"client:acme": "hard"}), registry=registry,
                 with_structure=False)

    assert answer.routing.selected == ["a"]
    assert not answer.routing.abstain


def test_a_hard_group_abstention_reaches_ask_with_its_own_reason(tmp_path):
    """The two abstentions hard mode must NOT claim as its own, end to end.

    `route` pins all four outcomes, but only against hand-built arguments;
    until now nothing wired a real Policy to them. `--group` carries the other
    half of the coverage: it is the only path that confines WITHOUT a cwd, and a
    cwd signal on the leading in-group scope suppresses the deictic branch, so a
    cwd-anchored group is the wrong place to look for that outcome.
    """
    from loci.ask import ask
    from loci.groups import Policy

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                   b=("Beta", str(b), {"flange": 25}, 400))
    registry = [Scope(id="a", name="Alpha", root=a, groups=["me"]),
                Scope(id="b", name="Beta", root=b, groups=["client:acme"])]
    confined = dict(group="client:acme", index=index, store={},
                    policy=Policy(default_mode="soft",
                                  groups={"client:acme": "hard"}),
                    registry=registry, with_structure=False)

    deictic = ask("how does this project start up?", **confined)
    nothing = ask("xyzzy plugh frotz", **confined)

    assert deictic.routing.abstain_reason == "deictic"
    assert nothing.routing.abstain_reason == "no_evidence"
    # Both really were confined -- `group=` reached `confinement`, not just the
    # reporting fields -- so neither reason is the unconfined router's default.
    assert deictic.routing.group == nothing.routing.group == "client:acme"
    assert deictic.routing.mode == nothing.routing.mode == "hard"


def test_a_rendered_abstention_says_what_caused_it():
    """Hard mode converts answers into abstentions, and one that does not name
    its cause is indistinguishable from a bug.

    `out_of_group` is specifically NOT "not specific enough to route": the
    question was specific, it was aimed outside the group. Naming the cause has
    to REPLACE that headline rather than trail it, or the line asserts a false
    cause and then contradicts itself.
    """
    from loci.ask import Answer, render
    from loci.types import RouteResult

    index = _index(a=("Alpha", "/a", {"widget": 40}, 500))

    def headline(reason, group=None, ranked=("a",)):
        rt = RouteResult(question="q", query_tokens=[], ranked=list(ranked),
                         selected=[], abstain=True, top_score=0.0, top_matched=0,
                         group=group, abstain_reason=reason)
        return render(Answer(question="q", routing=rt), index=index).splitlines()[0]

    assert "client:acme" in headline("out_of_group", "client:acme")
    assert "not specific enough" not in headline("out_of_group", "client:acme")
    assert "points at its subject" in headline("deictic")
    assert "not enough of the question" in headline("no_evidence")
    assert headline(None) == "ABSTAINED - not specific enough to route."
    # An empty candidate list is only the GROUP's fault when a group is in
    # force. Without one it means an empty index, and "no indexed project is in
    # group None" is not a sentence.
    assert headline("no_evidence", ranked=()) == \
        "ABSTAINED - not enough of the question exists in any project."


def test_an_unknown_group_asked_through_ask_answers_from_nothing(tmp_path):
    """`eligible` is tri-state, and `ask` is where its two falsy states meet.

    `confinement` returns `set()` for a group nobody is in and `None` for no
    confinement at all. `eligible=conf.eligible or None` collapses them and
    answers `--group typo` from the whole corpus -- and every other test in this
    section passes with that mutation in place, because a group somebody IS in
    is truthy.
    """
    from loci.ask import ask, render
    from loci.groups import Policy

    a = tmp_path / "a"
    a.mkdir()
    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                   b=("Beta", "/b", {"flange": 25}, 400))
    registry = [Scope(id="a", name="Alpha", root=a, groups=["me"])]

    answer = ask("how does the widget gizmo sprocket work", group="typo",
                 index=index, store={}, policy=Policy(), registry=registry,
                 with_structure=False)

    assert answer.routing.abstain
    assert answer.routing.ranked == []
    assert answer.scopes == [], "an unknown group answered from the whole corpus"

    # And it must not blame the question. `eligible=set()` leaves `route` weighing
    # the question against an empty candidate set, so it reports `no_evidence` --
    # true of that set, false of the question. An abstention has to name its real
    # cause, and this is the likeliest user error the feature has.
    text = render(answer, index=index)
    assert text.splitlines()[0] == "ABSTAINED - no indexed project is in group typo."
    assert "not enough of the question" not in text


def test_a_soft_group_demotes_through_ask_but_still_includes(tmp_path):
    """Soft is the DEFAULT mode, and `demoted=None` -- soft doing nothing at all
    -- passed every other test in this file.

    Alpha leads this corpus unpenalised (4.66 to Beta's 4.15, which is Beta's
    cwd boost). Anchoring in Beta's group demotes Alpha's evidence base to
    2.33 and Beta takes the lead. Alpha must still come back in the answer:
    demoting is what separates soft from hard, and `eligible` wired where
    `demoted` belongs would drop it entirely.
    """
    from loci.ask import ask
    from loci.groups import Policy

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 100),
                   b=("Beta", str(b), {"flange": 25}, 400))
    q = "how does the widget gizmo sprocket work"

    def routing(groups):
        return ask(q, cwd=str(b), index=index, store={}, policy=Policy(),
                   registry=[Scope(id="a", name="Alpha", root=a),
                             Scope(id="b", name="Beta", root=b, groups=groups)],
                   with_structure=False).routing

    plain = routing(None)
    soft = routing(["me"])

    assert plain.ranked[0] == "a", "fixture lost its point: Alpha must lead unpenalised"
    assert soft.ranked[0] == "b", "the soft group did not demote Alpha"
    assert "a" in soft.selected, "soft demotes; only hard excludes"
    assert (soft.group, soft.mode) == ("me", "soft")


@pytest.mark.parametrize("body", ['{"scopes": ', '[]', '{"scopes": ["me"]}',
                                  '{"scopes": [{"name": "Alpha"}]}'])
def test_a_malformed_registry_degrades_to_unconfined(tmp_path, monkeypatch, body):
    """`ask` reads the registry on every call now, and `load_scopes` raises.

    `load_policy` swallows every malformed shape on the stated ground that a
    traceback out of `loci ask` is a worse answer than "no groups configured".
    The registry is read on that same new path -- until confinement arrived
    `ask` ran off the index alone -- so the sibling loader must not differ.
    These four bodies raise four different types (JSONDecodeError,
    AttributeError, TypeError, KeyError), which is why the catch is broad.

    The guard is in `ask`, NOT in `load_scopes`: a silent [] there would let the
    next `upsert` write a registry with every registered scope discarded.
    """
    import loci.paths as P
    from loci.ask import ask
    from loci.groups import Policy
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    P.registry_file().write_text(body, encoding="utf-8")
    with pytest.raises(Exception):
        load_scopes()             # the fixture really is malformed

    a = tmp_path / "a"
    a.mkdir()
    index = _index(a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
                   b=("Beta", "/b", {"flange": 25}, 400))

    # cwd is passed so the degraded registry is actually WALKED: `scope_for_cwd`
    # iterates it, so degrading to None rather than [] raises a TypeError one
    # frame further down and the guard buys nothing.
    answer = ask("how does the widget gizmo sprocket work", cwd=str(a),
                 index=index, store={}, policy=Policy(), with_structure=False)

    assert answer.routing.selected == ["a"]
    assert answer.routing.group is None


# -- cli: groups -----------------------------------------------------------
@pytest.fixture
def loci_home(tmp_path, monkeypatch):
    """$LOCI_HOME is the only thing between a CLI test and the developer's own
    ~/.loci: every test below writes a real registry and reads it back.
    """
    import loci.paths as P
    monkeypatch.setenv(P.ENV_HOME, str(tmp_path))
    return tmp_path


def _install_index(index) -> None:
    """Put an `_index()` dict where `loci route` and `loci ask` will find it."""
    import loci.paths as P
    P.ensure_home()
    P.scope_index_file().write_text(json.dumps(index), encoding="utf-8")


def _two_scope_corpus(home):
    """Alpha in group `me`, Beta in `client:acme`, and an index that answers
    only about Beta. A `--group me` question about Beta's vocabulary is the
    shape every group-abstention test below needs.
    """
    from loci.scopes import save_scopes

    a, b = home / "a", home / "b"
    a.mkdir()
    b.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=a, groups=["me"]),
                 Scope(id="b", name="Beta", root=b, groups=["client:acme"])])
    _install_index(_index(
        a=("Alpha", str(a), {"widget": 40, "gizmo": 30, "sprocket": 20}, 500),
        b=("Beta", str(b), {"flange": 25, "grommet": 10, "bolt": 5}, 400)))
    return a, b


def test_groups_infer_reaches_its_own_handler():
    """`groups` carries a default handler AND a nested sub-parser with another.
    Which one argparse leaves on the namespace is not obvious, and getting it
    wrong makes `loci groups infer` silently print the listing instead.
    """
    from loci.cli import build_parser, cmd_groups, cmd_groups_infer

    p = build_parser()
    assert p.parse_args(["groups"]).func is cmd_groups
    assert p.parse_args(["groups", "infer"]).func is cmd_groups_infer


def test_groups_listing_names_where_each_mode_came_from(loci_home, capsys):
    """The one objection to per-group-with-fallback is "two places to look".
    One line of output is the answer.

    Asserted per ROW, not over the whole text: "default" also appears in the
    `default mode:` trailer, so a substring check passes with every row wrong.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy
    from loci.scopes import save_scopes

    a, b, c = loci_home / "a", loci_home / "b", loci_home / "c"
    for d in (a, b, c):
        d.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=a, groups=["me", "client:acme"]),
                 Scope(id="b", name="Beta", root=b, groups=["me"]),
                 Scope(id="c", name="Gamma", root=c, groups=[])])
    save_policy(Policy(default_mode="soft", groups={"client:acme": "hard"}))

    assert main(["groups"]) == 0
    out = capsys.readouterr().out
    rows = {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.startswith("  ")}
    assert rows["client:acme"][1:4] == ["hard", "declared", "1"]
    assert rows["me"][1:4] == ["soft", "default", "2"]
    assert "1 scope(s) in no group: Gamma" in out
    assert "default mode: soft" in out


def test_group_add_and_rm_round_trip(loci_home):
    """`upsert` UNIONS `groups` so a re-scan cannot erase a structural label,
    which means a membership edit routed through it can never remove anything.
    These commands write the registry directly for exactly that reason.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    a = loci_home / "a"
    a.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=a)])

    assert main(["group", "add", "a", "client:acme"]) == 0
    assert main(["group", "add", "a", "me"]) == 0
    assert load_scopes()[0].group_set() == {"client:acme", "me"}

    assert main(["group", "rm", "a", "client:acme"]) == 0
    assert load_scopes()[0].groups == ["me"]

    assert main(["group", "rm", "a", "me"]) == 0
    assert load_scopes()[0].groups == [], \
        "removing the last group must be a deliberate [], not None"


def test_group_rm_refuses_a_label_it_cannot_actually_remove(loci_home, capsys):
    """`discover` recomputes a monorepo's containment label on every scan and
    `upsert` unions it back in, so a removal here reappears at the next
    `loci scan`. Silently no-opping is the worst of the three options.

    Both ends of the label are refused: `discover` writes it onto each
    sub-project AND onto the container itself, which "is a member of its own
    containment group".
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    mono = loci_home / "mono"
    client = mono / "client"
    client.mkdir(parents=True)
    # Two things make this structural, and each has its own allow-test below:
    # `.git` (only a repository is ever a `parent` in `discover`) and the
    # depth-1 marker (only `subscopes` output is ever labelled).
    (mono / ".git").mkdir()
    (client / "package.json").write_text("{}", encoding="utf-8")
    save_scopes([Scope(id="mono", name="Mono", root=mono, groups=["mono"]),
                 Scope(id="mono-client", name="Mono/client", root=client,
                       groups=["client:acme", "mono"])])

    assert main(["group", "rm", "mono-client", "mono"]) == 2
    assert "loci scan" in capsys.readouterr().err
    assert load_scopes()[1].group_set() == {"client:acme", "mono"}

    assert main(["group", "rm", "mono", "mono"]) == 2
    err = capsys.readouterr().err
    assert "loci scan" in err
    assert "Mono lives inside Mono" not in err, "a scope contains itself"
    assert load_scopes()[0].group_set() == {"mono"}

    # The refusal is about CONTAINMENT, not about being a sub-scope: a label
    # the filesystem does not assert comes off the same scope normally.
    assert main(["group", "rm", "mono-client", "client:acme"]) == 0
    assert load_scopes()[1].group_set() == {"mono"}


def test_group_rm_allows_a_label_no_scan_would_recompute(loci_home):
    """`discover` asserts containment for depth-1 workspace markers and for
    `.loci.json` declarations -- nothing else. Two scopes registered
    independently that merely happen to nest carry no structural relationship,
    so refusing the removal promises a scan that will never restore the label
    and leaves the user unable to undo their own `group add`.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    outer = loci_home / "outer"
    deep = outer / "x" / "y"          # depth 2, and no marker anywhere above it
    deep.mkdir(parents=True)
    # A real repository, so this test isolates the depth/marker reason rather
    # than passing for want of a `.git`.
    (outer / ".git").mkdir()
    save_scopes([Scope(id="outer", name="Outer", root=outer),
                 Scope(id="deep", name="Deep", root=deep)])

    assert main(["group", "add", "deep", "outer"]) == 0
    assert main(["group", "rm", "deep", "outer"]) == 0
    assert load_scopes()[1].groups == []


def test_group_rm_allows_a_container_label_with_nothing_inside_it(loci_home, capsys):
    """`discover` labels a repository with its own id only when it actually
    holds sub-projects (`if subs:`). A scope carrying that label with nothing
    inside it was told it "lives inside" itself -- a sentence about a
    relationship that does not exist, blocking a removal no scan would undo.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    mono = loci_home / "mono"
    mono.mkdir()
    (mono / ".git").mkdir()      # a repository, but one holding no sub-projects
    save_scopes([Scope(id="mono", name="Mono", root=mono, groups=["mono"])])

    assert main(["group", "rm", "mono", "mono"]) == 0
    assert load_scopes()[0].groups == []


def test_group_rm_allows_a_label_from_a_directory_scan_never_visits(loci_home):
    """`discover` collects only git repositories (`(d / ".git").exists()`), so a
    plain directory never becomes a `parent` and never hands out its id --
    `loci scan` answers "no git repositories found" for the very tree the
    refusal cites. Blocking the removal there promises a scan that cannot
    produce the label, and there is no `--force`.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    nogit = loci_home / "nogit"
    pkg = nogit / "pkg"
    pkg.mkdir(parents=True)
    # A genuine depth-1 marker: `subscopes(nogit)` really does return `pkg`, so
    # only the repository check can refuse this removal.
    (pkg / "package.json").write_text("{}", encoding="utf-8")
    save_scopes([Scope(id="nogit", name="Nogit", root=nogit),
                 Scope(id="pkg", name="Pkg", root=pkg, groups=["nogit"])])

    assert main(["group", "rm", "pkg", "nogit"]) == 0
    assert load_scopes()[1].groups == []


def test_group_rm_refuses_a_label_held_by_a_worktree_or_submodule(loci_home):
    """`discover` collects on `(d / ".git").exists()`, and a git worktree or a
    submodule carries `.git` as a FILE, not a directory.

    Checking `.is_dir()` in the guard instead would make exactly those
    containers' labels look removable -- and they would silently reappear at the
    next scan, which is the whole failure the refusal exists to prevent.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    mono, client = loci_home / "mono", loci_home / "mono" / "client"
    client.mkdir(parents=True)
    (mono / ".git").write_text("gitdir: /elsewhere/.git/worktrees/mono\n",
                               encoding="utf-8")
    (client / "package.json").write_text("{}", encoding="utf-8")
    save_scopes([Scope(id="mono", name="Mono", root=mono, groups=["mono"]),
                 Scope(id="mono-client", name="Mono/client", root=client,
                       groups=["mono"])])

    assert main(["group", "rm", "mono-client", "mono"]) == 2
    assert load_scopes()[1].group_set() == {"mono"}


def test_group_rm_allows_a_container_id_borrowed_as_a_plain_label(loci_home):
    """Pins the predicate as a WHOLE, which the other allow-tests cannot.

    In both of those the holder has empty `subs`, so `if not subs` alone carries
    them -- and an unconditional `return holder`, a predicate strictly broader
    than the one this guard replaced, passes every one of them. Here the holder
    genuinely holds a sub-project, so it survives every earlier guard and only
    "is this scope the container, or one of its sub-projects?" can still say no.

    A scope is free to join a group whose name happens to be some container's
    id without being anything of that container's.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    mono, solo = loci_home / "mono", loci_home / "solo"
    client = mono / "client"
    client.mkdir(parents=True)
    (mono / ".git").mkdir()
    (client / "package.json").write_text("{}", encoding="utf-8")
    solo.mkdir()
    save_scopes([Scope(id="mono", name="Mono", root=mono, groups=["mono"]),
                 Scope(id="mono-client", name="Mono/client", root=client,
                       groups=["mono"]),
                 Scope(id="solo", name="Solo", root=solo, groups=["mono"])])

    assert main(["group", "rm", "solo", "mono"]) == 0
    assert load_scopes()[2].groups == []

    # The fixture is discriminating: the same label on the container and on its
    # real sub-project is still refused, so the allowance above is about THIS
    # scope's relationship and not about the holder having become harmless.
    assert main(["group", "rm", "mono", "mono"]) == 2
    assert main(["group", "rm", "mono-client", "mono"]) == 2


def test_group_set_rejects_an_unknown_mode(loci_home):
    """MODES is derived from STRICTNESS precisely so the mode names are written
    once; a parser spelling them out again is another copy to drift from it.
    A rejected mode must also leave no half-written policy behind.
    """
    import loci.paths as P
    from loci.cli import main
    from loci.groups import MODES, load_policy

    with pytest.raises(SystemExit):
        main(["group", "set", "client:acme", "--mode", "nonsense"])
    assert not P.groups_file().exists(), "a rejected mode still wrote the policy"

    assert main(["group", "set", "client:acme"]) == 2, "`set` needs a mode"
    assert not P.groups_file().exists()

    for mode in MODES:
        assert main(["group", "set", "client:acme", "--mode", mode]) == 0
        assert load_policy().mode_for("client:acme") == (mode, "declared")


def test_group_add_without_a_group_name_is_an_error_not_a_traceback(loci_home):
    """`group` is nargs="?" because `set` does not take one. Left as None it
    reaches `sorted(current | {None})`, which raises TypeError as soon as the
    scope has any group at all.
    """
    from loci.cli import main
    from loci.scopes import save_scopes

    a = loci_home / "a"
    a.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=a, groups=["me"])])

    assert main(["group", "add", "a"]) == 2
    assert main(["group", "rm", "a"]) == 2


@pytest.mark.parametrize("argv", [
    ["scopes"],
    ["route", "how", "does", "the", "flange", "work"],
    ["ask", "how", "does", "the", "flange", "work"],
])
def test_an_unknown_group_is_rejected_before_it_confines_to_nothing(
        loci_home, capsys, argv):
    """An unknown group resolves to `eligible=set()` -- confined to nothing --
    and the abstention that follows is honest but reads like a bug for what is
    almost always a typo. All three surfaces reject it where the user sees it.

    `ask` must reject BEFORE it queries: reaching `ask()` here would need the
    episode and structure backends, which is the point -- the check is early.
    """
    from loci.cli import main

    _two_scope_corpus(loci_home)
    assert main(argv + ["--group", "typo"]) == 2
    err = capsys.readouterr().err
    assert "typo" in err and "client:acme" in err and "me" in err


def test_a_group_named_only_in_the_policy_is_still_known(loci_home, capsys):
    """`loci group set` declares a mode before anyone joins the group, so the
    known set is the union of the registry's and the policy's -- not the
    registry alone. Known-but-empty (1) must not read as unknown (2).
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy
    from loci.scopes import save_scopes

    a = loci_home / "a"
    a.mkdir()
    save_scopes([Scope(id="a", name="Alpha", root=a, groups=["me"])])
    save_policy(Policy(groups={"client:acme": "hard"}))

    assert main(["scopes", "--group", "client:acme"]) == 1
    assert "no scopes in group client:acme" in capsys.readouterr().out


def test_scopes_filters_to_one_group(loci_home, capsys):
    from loci.cli import main

    _two_scope_corpus(loci_home)
    assert main(["scopes", "--group", "me"]) == 0
    out = capsys.readouterr().out
    assert "Alpha" in out and "Beta" not in out
    assert "1 scope(s)" in out


def test_route_explain_names_the_effective_mode_and_where_it_came_from(
        loci_home, capsys):
    """"Two places to look for a mode" is the standing objection to
    per-group-with-fallback, so `--explain` has to answer it. Re-deriving the
    source at the print site -- `"group" if r.mode else "default"` -- reports
    "declared" for every group that has a mode at all, which is all of them.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy

    _two_scope_corpus(loci_home)
    argv = ["route", "how", "does", "the", "flange", "work",
            "--group", "me", "--explain", "--no-cwd"]

    save_policy(Policy(default_mode="soft"))
    assert main(argv) == 0
    assert "group:  me  mode=soft (default)" in capsys.readouterr().out

    save_policy(Policy(default_mode="soft", groups={"me": "hard"}))
    assert main(argv) == 0
    assert "group:  me  mode=hard (declared)" in capsys.readouterr().out


def test_route_says_why_it_abstained_without_being_asked_to_explain(
        loci_home, capsys):
    """A hard group turns an answer into an abstention -- the change most
    likely to read as a regression. Behind `--explain` is not where a user
    meets it.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy

    _two_scope_corpus(loci_home)
    save_policy(Policy(groups={"me": "hard"}))

    assert main(["route", "how", "does", "the", "flange", "grommet", "bolt",
                 "work", "--group", "me", "--no-cwd"]) == 0
    out = capsys.readouterr().out
    assert "ABSTAIN" in out and "out_of_group" in out


def test_route_explain_names_the_scope_the_group_excluded(loci_home, capsys):
    """`ranked` is filtered to the eligible set, so the scope that CAUSED an
    out_of_group abstention is missing from the one output meant to explain it.
    `detail` still carries every scope; `--explain` has to show them.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy

    _two_scope_corpus(loci_home)
    save_policy(Policy(groups={"me": "hard"}))

    assert main(["route", "how", "does", "the", "flange", "grommet", "bolt",
                 "work", "--group", "me", "--explain", "--no-cwd"]) == 0
    out = capsys.readouterr().out
    assert "Beta" in out, "the scope holding the answer is never named"
    assert "excluded by group me" in out


def test_an_abstention_gives_advice_that_can_actually_work():
    """Two defects in one trailer. "or from inside the project directory" is
    what CAUSES an out_of_group confinement, so it is advice guaranteeing the
    abstention it is offered against; and an empty `ranked` renders as a
    dangling "candidates: " with nothing after it, which reads as a truncated
    line rather than as "none".
    """
    from loci.ask import Answer, render
    from loci.types import RouteResult

    index = _index(a=("Alpha", "/a", {"widget": 40}, 500))

    def body(reason, group=None, ranked=("a",)):
        rt = RouteResult(question="q", query_tokens=[], ranked=list(ranked),
                         selected=[], abstain=True, top_score=0.0, top_matched=0,
                         group=group, abstain_reason=reason)
        return render(Answer(question="q", routing=rt), index=index)

    confined = body("out_of_group", "client:acme")
    assert "--scope" in confined
    assert "inside the project directory" not in confined
    assert "inside the project directory" in body("no_evidence"), \
        "the ordinary abstention still wants the cwd hint"

    unknown = body("no_evidence", "typo", ranked=())
    assert "candidates:" not in unknown
    assert "loci groups" in unknown
    assert all(ln.strip() for ln in unknown.splitlines()), "a blank rendered line"


def test_a_policy_that_cannot_be_constructed_is_an_error_not_a_traceback(
        loci_home, capsys, monkeypatch):
    """`Policy.__post_init__` refuses an unknown `default_mode`. `load_policy`
    normalizes every field it reads, so no shipped path reaches that raise --
    but the loader dropping one normalization is a one-line change away, and a
    CLI is not allowed to answer a user's text with a stack trace.
    """
    import loci.groups as G
    from loci.cli import main

    def boom():
        raise ValueError("unknown mode 'nonsense'; expected one of explicit, soft, hard")

    monkeypatch.setattr(G, "load_policy", boom)
    with pytest.raises(SystemExit) as exc:
        main(["groups"])
    assert exc.value.code == 2
    assert "unknown mode" in capsys.readouterr().err


def test_groups_infer_adds_a_label_without_erasing_what_the_user_asserted(
        loci_home, hermetic_git, capsys):
    """`client:*` is never inferrable -- a client relationship is not visible in
    git -- so inference has to ADD. Replacing `groups` would delete exactly the
    labels only the user can supply.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    repo = _repo(loci_home, "mine", origin="git@github.com:me/mine.git")
    save_scopes([Scope(id="mine", name="Mine", root=repo, groups=["client:acme"])])

    assert main(["groups", "infer"]) == 0
    assert load_scopes()[0].group_set() == {"client:acme", "me"}
    assert "+ Mine" in capsys.readouterr().out

    assert main(["groups", "infer"]) == 0
    assert "0 scope(s) labelled" in capsys.readouterr().out, "not idempotent"


def test_groups_infer_does_not_let_one_monorepo_outvote_the_corpus(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """The same inversion `loci scan` was fixed against, in the command `doctor`
    tells users to run.

    git resolves `origin` by walking upward, so each of a monorepo's four
    sub-scopes reports the parent's org again. Counting REGISTRY scopes rather
    than repositories is 5 votes for `big` against 3 for `acme`, and a wrong
    identity inverts every classification at once: `scan` files the stranger's
    monorepo under `vendor:big` and `groups infer`, run straight afterwards on
    the very registry `scan` wrote, files the user's own three repositories
    under `vendor:acme`.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    mono = _repo(corpus, "mono", origin="git@github.com:big/mono.git")
    for sub in ("a", "b", "c", "d"):
        (mono / sub).mkdir()
        (mono / sub / "package.json").write_text("{}", encoding="utf-8")
    for name in ("one", "two", "three"):
        _repo(corpus, name, origin=f"https://github.com/acme/{name}.git")

    assert main(["scan", "--split", str(corpus)]) == 0
    before = {s.name: s.group_set() for s in load_scopes()}
    assert before["one"] == {"me"}, "fixture lost its point"
    capsys.readouterr()

    assert main(["groups", "infer"]) == 0
    out = capsys.readouterr().out
    assert "you are acme" in out, "inference disagreed with the scan that registered these"

    after = {s.name: s.group_set() for s in load_scopes()}
    assert after == before, "inference rewrote labels the scan got right"
    assert after["one"] == {"me"}
    assert after["mono/a"] == {"mono", "vendor:big"}


def test_groups_infer_replaces_a_stale_provenance_label(
        loci_home, hermetic_git, capsys):
    """`classify` answers ONE question from one repository's own git, and answers
    it afresh every run. Here it answers `me` -- the remote org IS the identity
    -- so the stored `vendor:big` is that same question answered contradictorily,
    and a hard group over either then admits the other's members. Pure union also
    made the command unable to repair a registry it had corrupted: re-running
    only added.

    What the user asserted is not provenance and must survive. So does `me` when
    the fresh reading is a vendor's -- the test below.
    """
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    repo = _repo(loci_home, "mine", origin="git@github.com:me/mine.git")
    save_scopes([Scope(id="mine", name="Mine", root=repo,
                       groups=["client:acme", "vendor:big"])])

    assert main(["groups", "infer"]) == 0
    assert load_scopes()[0].group_set() == {"client:acme", "me"}, \
        "the contradicting provenance label survived, or the assertion did not"
    assert "(was vendor:big)" in capsys.readouterr().out, \
        "a replacement was reported as a bare addition"

    assert main(["groups", "infer"]) == 0
    assert "0 scope(s) labelled" in capsys.readouterr().out, "not idempotent"


def test_groups_infer_never_strips_me_from_work_under_a_second_org(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """A personal GitHub account plus an employer's is an ordinary corpus, and
    `infer_identity` reports exactly ONE dominant org, so half of the user's own
    work classifies as a vendor's however plainly it is theirs.

    Reproduced the way it happens: `loci scan ~/work` first, whose identity is
    inferred from that scan alone and is therefore `workorg`, so the three work
    repositories are correctly labelled `me`. `loci scan ~/personal` then adds
    five more and a stranger's, and `groups infer` -- which reads the whole
    registry -- flips the identity to `myname`. Retracting `me` there deletes
    the user's own work from their own group: `loci group set me --mode hard`
    confines to the personal half, and it takes `loci group add w1 me` three
    times to undo, each one destroyed by the next run of this command.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.groups import members
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    work, personal = tmp_path / "work", tmp_path / "personal"
    work.mkdir()
    personal.mkdir()
    for n in ("w1", "w2", "w3"):
        _repo(work, n, origin=f"git@github.com:workorg/{n}.git")
    for n in ("p1", "p2", "p3", "p4", "p5"):
        _repo(personal, n, origin=f"git@github.com:myname/{n}.git")
    _repo(personal, "borrowed", origin="git@github.com:stranger/borrowed.git")

    assert main(["scan", str(work)]) == 0
    assert main(["scan", str(personal)]) == 0
    before = {s.name: s.group_set() for s in load_scopes()}
    assert before["w1"] == {"me"}, \
        "fixture lost its point: the work scan must have called these yours"
    assert before["borrowed"] == {"vendor:stranger"}
    capsys.readouterr()

    assert main(["groups", "infer"]) == 0
    assert "you are myname" in capsys.readouterr().out, \
        "fixture lost its point: the identity has to FLIP for this to bite"

    after = load_scopes()
    by_name = {s.name: s.group_set() for s in after}
    # The fresh reading is recorded -- w1's remote really is workorg's -- but it
    # is recorded BESIDE `me`, not instead of it. Equality, not a subset check:
    # a subset check passes for the bug this pins.
    assert by_name["w1"] == {"me", "vendor:workorg"}
    assert by_name["w2"] == {"me", "vendor:workorg"}
    assert by_name["w3"] == {"me", "vendor:workorg"}
    assert by_name["p1"] == {"me"}, "the personal half was disturbed"
    assert by_name["borrowed"] == {"vendor:stranger"}

    # The outcome the label exists for, rather than the label: `me --mode hard`
    # has to still reach the user's own work.
    assert members(["me"], after) == {"w1", "w2", "w3", "p1", "p2", "p3", "p4", "p5"}

    assert main(["groups", "infer"]) == 0
    assert "0 scope(s) labelled" in capsys.readouterr().out, "not idempotent"


def test_groups_infer_names_the_scopes_it_left_in_both_groups(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """What not retracting `me` costs, and the escape hatch that pays it.

    Before this branch, a stranger's repository that an early scan had called
    yours was repaired by the next `groups infer`: `+ stranger vendor:evilcorp
    (was me)`. It no longer is -- `me` stays, and `loci group set me --mode
    hard` then ADMITS the stranger, because a hard group admits its own members.
    git cannot tell that case from your own work under an employer's org, so the
    command does not guess; it must at least SAY so and name the one edit that
    decides, or the escape hatch is undiscoverable.

    Reproduced end to end, including that the edit sticks: an early scan with
    tied orgs labels everything `me`, the corpus grows until `myname` dominates,
    and inference then leaves `stranger` in both groups.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.groups import Policy, confinement
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    mine = _repo(corpus, "a", origin="git@github.com:myname/a.git")
    _repo(corpus, "stranger", origin="git@github.com:evilcorp/s.git")
    # Stays dual to the end, so the notice is still PRINTED on the last run.
    # Without it the final assertion passes for the wrong reason -- the notice
    # is skipped wholesale -- and cannot tell "omits this scope" from "says
    # nothing at all".
    _repo(corpus, "w1", origin="git@github.com:workorg/w1.git")

    # One org each: `infer_identity` refuses to name one, `classify` returns
    # `me` for everything, and the stranger is registered as the user's own.
    assert main(["scan", str(corpus)]) == 0
    assert {s.name: s.group_set() for s in load_scopes()}["stranger"] == {"me"}, \
        "fixture lost its point: the stranger has to arrive mislabelled"

    for n in ("b", "c"):
        _repo(corpus, n, origin=f"git@github.com:myname/{n}.git")
    assert main(["scan", str(corpus)]) == 0
    capsys.readouterr()

    assert main(["groups", "infer"]) == 0
    out = capsys.readouterr().out
    assert "you are myname" in out
    assert "stranger (vendor:evilcorp)" in out, \
        "the scope left in both groups was not named, nor its vendor group"
    assert "loci group rm" in out, "the one edit that decides was not offered"

    # Standing state, not a change report: printed on a run that changed nothing.
    assert main(["groups", "infer"]) == 0
    again = capsys.readouterr().out
    assert "0 scope(s) labelled" in again
    assert "stranger (vendor:evilcorp)" in again, \
        "the notice is tied to `changed`, so it vanishes exactly when it is the "\
        "only thing left to say"

    # The defect the notice exists to disclose, stated as behaviour.
    scopes = load_scopes()
    assert {s.name: s.group_set() for s in scopes}["stranger"] \
        == {"me", "vendor:evilcorp"}
    conf = confinement(Policy(groups={"me": "hard"}), scopes, cwd=mine)
    assert "stranger" in conf.eligible, "fixture lost its point"

    # And the offered edit works, and survives the command that suggested it.
    assert main(["group", "rm", "stranger", "me"]) == 0
    assert main(["groups", "infer"]) == 0
    after = capsys.readouterr().out
    # On the PAIR, not on the word. The notice's own prose contains "a
    # stranger's", so `"stranger" not in after` is satisfied by boilerplate and
    # would fail on any mutant that merely keeps the notice alive.
    assert "w1 (vendor:workorg)" in after, \
        "the notice is gone, so the next assertion proves nothing"
    assert "stranger (vendor:evilcorp)" not in after, \
        "the notice still names a scope that is no longer in both groups"

    scopes = load_scopes()
    assert {s.name: s.group_set() for s in scopes}["stranger"] == {"vendor:evilcorp"}
    conf = confinement(Policy(groups={"me": "hard"}), scopes, cwd=mine)
    assert "stranger" not in conf.eligible, "`group rm` did not survive inference"


def test_two_clones_of_one_repo_stay_separately_addressable_end_to_end(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """The whole chain, built with `loci scan` alone -- no hand-written registry.

    `loci scan x y` over two clones of one upstream repo registers (`utils`,
    "utils") and (`utils-2`, "utils"). Every surface that then talks about them
    has to say WHICH, and the remedy it prints has to reach the one it named:
    the notice, doctor's two lines, `loci group rm`, and the confirmation that
    command prints back.

    The registry is deliberately persisted with `utils-2` FIRST. That is a real
    order -- `save_scopes` writes what it is given and `upsert` sorts on a name
    both scopes share, so which one leads is filesystem order -- and it is the
    half in which the old one-pass `resolve` answered `utils-2` to the string
    `utils`, making id `utils` unreachable and `loci group rm utils me` a
    successful no-op that repeated forever.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.doctor import group_report
    from loci.groups import Policy, confinement
    from loci.scopes import load_scopes, save_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    x, y = tmp_path / "x", tmp_path / "y"
    x.mkdir()
    y.mkdir()
    mine = _repo(x, "mine", origin="git@github.com:myname/mine.git")
    _repo(x, "utils", origin="git@github.com:evilcorp/utils.git")
    _repo(y, "utils", origin="git@github.com:evilcorp/utils.git")

    # Two orgs, one repo each way: `infer_identity` names neither, `classify`
    # returns `me` for everything, and both clones register as the user's own.
    assert main(["scan", str(x), str(y)]) == 0
    ids = {s.id for s in load_scopes()}
    assert ids == {"mine", "utils", "utils-2"}, \
        f"fixture lost its point: the id collision did not happen ({ids})"
    assert len({s.name for s in load_scopes()}) == 2, \
        "fixture lost its point: the NAMES have to collide"

    for n in ("b", "c"):
        _repo(x, n, origin=f"git@github.com:myname/{n}.git")
    assert main(["scan", str(x), str(y)]) == 0
    # The adverse order; see the docstring.
    save_scopes(sorted(load_scopes(), key=lambda s: s.id != "utils-2"))
    capsys.readouterr()

    assert main(["groups", "infer"]) == 0
    out = capsys.readouterr().out
    assert "utils (vendor:evilcorp)" in out and "utils-2 (vendor:evilcorp)" in out, \
        "the notice named a string that cannot tell the two clones apart"

    # `loci group rm` reaches the scope the notice named, and only that one.
    assert main(["group", "rm", "utils", "me"]) == 0
    assert "utils: vendor:evilcorp" in capsys.readouterr().out
    by_id = {s.id: s.group_set() for s in load_scopes()}
    assert by_id["utils"] == {"vendor:evilcorp"}, "the edit landed on the sibling"
    assert by_id["utils-2"] == {"me", "vendor:evilcorp"}, "both were edited"

    # doctor now has one of each kind, and must not merge them on the shared
    # name: `utils-2` is the pure-vendor clone the hard-group remedy works on.
    lines = group_report(load_scopes(), Policy())
    assert _listed(next(ln for ln in lines if "not yours" in ln)) == ["utils"]
    dual = next(ln for ln in lines if "does NOT keep them out" in ln)
    assert _listed(dual) == ["utils-2"]
    assert "loci group rm utils-2 me" in dual

    # And that handle reaches the other one, which is the whole point.
    assert main(["group", "rm", "utils-2", "me"]) == 0
    # The discriminating confirmation: this scope's NAME is "utils", so a
    # confirmation printed from the name is indistinguishable from the run that
    # edited the other clone -- success reported against a string naming both.
    assert "utils-2: vendor:evilcorp" in capsys.readouterr().out, \
        "the confirmation did not say which scope it edited"
    scopes = load_scopes()
    assert {s.id: s.group_set() for s in scopes}["utils-2"] == {"vendor:evilcorp"}
    conf = confinement(Policy(groups={"me": "hard"}), scopes, cwd=mine)
    assert conf.mode == "hard"
    assert not {"utils", "utils-2"} & conf.eligible, \
        "a remedy reported as applied left the scope in the routable set"


def test_groups_infer_retracts_nothing_from_an_unconfident_identity(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """`provenance.classify` already refuses to ACCUSE from an identity the
    module has disowned; withdrawing an accusation from one is the same error
    turned around. `classify` returns `me` for every scope in that state, so
    without the guard a corpus that merely lost its dominant org would have every
    `vendor:` label stripped at once and be told the vendors are all the user's.

    Two repositories under two orgs tie, and `infer_identity` names neither.
    """
    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes, save_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    a = _repo(corpus, "a", origin="git@github.com:alpha/a.git")
    b = _repo(corpus, "b", origin="git@github.com:beta/b.git")
    save_scopes([Scope(id="a", name="A", root=a, groups=["vendor:alpha"]),
                 Scope(id="b", name="B", root=b, groups=["vendor:beta"])])

    assert main(["groups", "infer"]) == 0
    out = capsys.readouterr().out
    assert "no dominant remote org" in out, \
        "fixture lost its point: the identity has to be UNCONFIDENT"
    assert "(was " not in out, "a retraction was reported from a disowned identity"

    by_name = {s.name: s.group_set() for s in load_scopes()}
    assert by_name["A"] == {"me", "vendor:alpha"}
    assert by_name["B"] == {"me", "vendor:beta"}


def test_ask_passes_its_group_through_to_confinement(loci_home, capsys):
    """Nothing else proves the flag is WIRED: `--group` parsed and then dropped
    answers from the whole corpus, and every rejection test above still passes
    -- those exercise the validator, not the pass-through.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy

    _two_scope_corpus(loci_home)
    argv = ["ask", "how", "does", "the", "flange", "grommet", "bolt", "work",
            "--no-cwd", "--no-structure", "--no-episodes"]

    assert main(argv) == 0
    assert "ROUTED -> Beta" in capsys.readouterr().out, "fixture lost its point"

    save_policy(Policy(groups={"me": "hard"}))
    assert main(argv + ["--group", "me"]) == 0
    assert "ABSTAINED - best match was outside group me." in capsys.readouterr().out


def test_route_survives_a_hand_edited_registry(loci_home, capsys):
    """`ask` degrades to unconfined rather than raising on a malformed
    scopes.json. `route` reports on that same decision and now reads the same
    file, so it must not differ -- a diagnostic that crashes where the thing it
    diagnoses answers is worse than no groups at all.
    """
    import loci.paths as P
    from loci.cli import main

    _install_index(_index(a=("Alpha", "/a", {"widget": 40, "gizmo": 30,
                                             "sprocket": 20}, 500)))
    P.registry_file().write_text('{"scopes": [{"name": "Alpha"}]}', encoding="utf-8")

    assert main(["route", "how", "does", "the", "widget", "gizmo", "sprocket",
                 "work", "--no-cwd"]) == 0
    assert "Alpha" in capsys.readouterr().out


def test_ask_says_when_scope_overrides_group(loci_home, capsys):
    """`--scope` outranks group policy by design -- the user has already
    answered the question groups exist to answer. But `cmd_ask` VALIDATES
    `--group` and then hands it to a call that discards it, and validating a
    flag you are about to ignore is what makes the silence misleading.
    """
    from loci.cli import main

    _two_scope_corpus(loci_home)
    argv = ["ask", "how", "does", "the", "flange", "work", "--no-cwd",
            "--no-structure", "--no-episodes", "--scope", "Beta"]

    assert main(argv) == 0
    assert "overrides" not in capsys.readouterr().err, "nothing to warn about"

    assert main(argv + ["--group", "me"]) == 0
    cap = capsys.readouterr()
    assert "--scope overrides --group me" in cap.err
    assert "ROUTED -> Beta" in cap.out, "the scope must still win"


def test_route_names_the_group_when_nothing_is_indexed_in_it(loci_home, capsys):
    """`eligible=set()` leaves `route` weighing the question against an empty
    candidate set, so it reports a reason true of that set and false of the
    question -- `out_of_group`, when the question was never the problem.
    `ask.render` corrects that; `route` is the DIAGNOSTIC surface and had no
    business being the less informative of the two.
    """
    from loci.cli import main
    from loci.groups import Policy, save_policy

    _two_scope_corpus(loci_home)
    save_policy(Policy(groups={"ghost": "hard"}))

    assert main(["route", "how", "does", "the", "flange", "grommet", "bolt",
                 "work", "--group", "ghost", "--no-cwd"]) == 0
    out = capsys.readouterr().out
    assert "no indexed project is in group ghost" in out
    assert "out_of_group" not in out
    assert "candidates:" not in out


# -- doctor: groups --------------------------------------------------------
def _listed(line: str) -> list[str]:
    """The scopes a `group_report` line names, as a list.

    Substring checks cannot do this job here: `"utils" in line` is satisfied by
    `utils-2`, which is the exact confusion the collision tests below are about.
    """
    return line.split(": ", 1)[1].split(" - ")[0].split(", ")


def test_doctor_names_vendor_groups_still_in_the_routable_set(tmp_path):
    """A stranger's repository competing for every question is the problem this
    feature exists to solve; doctor must say so out loud.

    The remedy is NOT `group set vendor:stranger --mode explicit`. A mode
    governs what a scope in that group does to questions asked from inside it,
    and `explicit` is the mode that confines least -- measured, the vendor
    still ranks. `test_forced_group_narrows_even_under_explicit` pins the same
    thing one layer down: under `explicit`, cwd yields `eligible is None`.

    The line is asserted whole rather than by keyword: a report that merely
    lists every group and its members satisfies "vendor:stranger appears"
    without telling anybody anything.
    """
    from loci.doctor import group_report
    from loci.groups import Policy

    scopes = [Scope(id="alpha", name="Alpha", root=tmp_path / "a", groups=["me"]),
              Scope(id="odysseus", name="Odysseus", root=tmp_path / "b",
                    groups=["vendor:stranger"])]
    vendor = next(ln for ln in group_report(scopes, Policy())
                  if ln.startswith("vendor:"))

    assert vendor.startswith("vendor:stranger: odysseus - not yours, and still "
                             "in the routable set.")
    assert "loci group set me --mode hard" in vendor
    assert "--mode explicit" not in vendor, "advice that does not do what it says"


def test_doctor_does_not_call_a_scope_in_me_a_vendors_with_an_inert_remedy(
        tmp_path):
    """`groups infer` never retracts `me`, so a repository of yours under an
    employer's org carries `me` AND `vendor:workorg`. The vendor line then told
    the user their own work was "not yours", and handed them a remedy that does
    nothing to it: `confinement` resolves a hard `me` to `members(["me"])`, and
    this scope is IN that set.

    Both halves are checked, and the second is checked against `confinement`
    rather than against the sentence -- the failure was a true-sounding line, so
    a test that only reads lines could not have caught it. Odysseus is here to
    keep the split discriminating: reporting every vendor member on the new line
    would satisfy every assertion about w1 while destroying the original one.
    """
    from loci.doctor import group_report
    from loci.groups import Policy, confinement

    p1 = tmp_path / "p1"
    scopes = [Scope(id="p1", name="p1", root=p1, groups=["me"]),
              Scope(id="w1", name="w1", root=tmp_path / "w1",
                    groups=["me", "vendor:workorg"]),
              Scope(id="odysseus", name="Odysseus", root=tmp_path / "o",
                    groups=["vendor:stranger"])]
    p1.mkdir()
    lines = group_report(scopes, Policy())

    not_yours = [ln for ln in lines if "not yours" in ln]
    assert len(not_yours) == 1
    assert _listed(not_yours[0]) == ["odysseus"], \
        "the genuine vendor lost its remedy, or the user's own work was on it"

    dual = next(ln for ln in lines if ln.startswith("vendor:workorg:"))
    assert _listed(dual) == ["w1"]
    assert "does NOT keep them out" in dual
    assert "loci group rm w1 me" in dual, \
        "the advice must carry a handle, not a `<scope>` placeholder"

    # What the two lines CLAIM, checked against the code they describe. `me`
    # hard, asked from inside p1: w1 is admitted (so the first line's remedy
    # would have been inert on it) and Odysseus is not (so the first line is
    # true of Odysseus).
    conf = confinement(Policy(groups={"me": "hard"}), scopes, cwd=p1)
    assert conf.mode == "hard"
    assert "w1" in conf.eligible
    assert "odysseus" not in conf.eligible


def test_doctor_partitions_vendor_members_by_id_not_by_name(tmp_path):
    """`make_scope` takes `name` from the directory and only `_unique_id`
    uniquifies, so `loci scan x y` over two clones of one upstream repo
    registers (`utils`, "utils") and (`utils-2`, "utils").

    Keyed on the name, one dual-labelled clone put its identically-named sibling
    into `dual` too, and `outside` -- computed by subtracting that set -- came
    out EMPTY. The sibling is a plain vendor scope with no `me` on it, the one
    kind the hard-group remedy actually works on, and it silently lost the line
    that offers it. The two ids also make the advice's handle load-bearing:
    `loci group rm utils me` and `loci group rm utils-2 me` are different edits,
    and a name cannot express either.
    """
    from loci.doctor import group_report
    from loci.groups import Policy

    scopes = [Scope(id="p1", name="p1", root=tmp_path / "p1", groups=["me"]),
              Scope(id="utils", name="utils", root=tmp_path / "x" / "utils",
                    groups=["me", "vendor:evilcorp"]),
              Scope(id="utils-2", name="utils", root=tmp_path / "y" / "utils",
                    groups=["vendor:evilcorp"])]
    lines = group_report(scopes, Policy())

    not_yours = next(ln for ln in lines if "not yours" in ln)
    assert _listed(not_yours) == ["utils-2"], \
        "the pure-vendor clone lost the remedy that works on it"

    dual = next(ln for ln in lines if "does NOT keep them out" in ln)
    assert _listed(dual) == ["utils"]
    assert "loci group rm utils me" in dual


def test_doctor_names_ungrouped_scopes(tmp_path):
    """A scope in no group is the state every registry written before this
    feature is in: no `--group` reaches it and no cwd confines for it.

    Beta is here to keep the check discriminating -- a report that names every
    scope it was handed satisfies "Alpha appears" without having found anything.
    """
    from loci.doctor import group_report
    from loci.groups import Policy

    scopes = [Scope(id="a", name="Alpha", root=tmp_path / "a"),
              Scope(id="b", name="Beta", root=tmp_path / "b", groups=["me"])]
    lines = "\n".join(group_report(scopes, Policy()))

    assert "1 scope(s) in no group: Alpha" in lines
    assert "loci groups infer" in lines
    assert "Beta" not in lines


def test_doctor_names_the_confinement_nobody_chose(tmp_path):
    """`discover` writes a containment label onto every sub-project of a
    monorepo, `DEFAULT_MODE` is soft, and `ask` reads both on every call. So the
    first question asked from inside a re-scanned monorepo demotes every outside
    scope, and nothing anywhere says so. Silent registration is the failure this
    feature exists to close; a silent default is the same failure re-labelled.

    A DECLARED mode is not silent -- the user typed it -- and an `explicit`
    default confines nothing, so neither is reported.
    """
    from loci.doctor import group_report
    from loci.groups import Policy

    scopes = [Scope(id="mono", name="Mono", root=tmp_path / "m", groups=["mono"]),
              Scope(id="mono-client", name="Mono/client",
                    root=tmp_path / "m" / "c", groups=["mono"]),
              Scope(id="loci", name="Loci", root=tmp_path / "l", groups=["me"])]

    lines = "\n".join(group_report(scopes, Policy()))
    assert "which nobody chose: me, mono" in lines
    assert "x0.5" in lines, "the penalty that does it is never named"
    assert "loci group set <group> --mode explicit" in lines

    declared = "\n".join(group_report(
        scopes, Policy(groups={"mono": "soft", "me": "soft"})))
    assert "nobody chose" not in declared
    assert "mono: mode soft (declared), 2 member(s)" in declared

    chosen = "\n".join(group_report(scopes, Policy(default_mode="explicit")))
    assert "nobody chose" not in chosen


def test_doctor_names_a_declared_group_nobody_is_in(tmp_path):
    """`loci group set` declares a mode for a name that need not exist yet, so a
    typo produces a group that confines nothing, and `--group` on the name the
    user meant then reports an unknown group rather than an empty one.
    """
    from loci.doctor import group_report
    from loci.groups import Policy

    scopes = [Scope(id="a", name="Alpha", root=tmp_path / "a", groups=["me"])]
    lines = "\n".join(group_report(scopes, Policy(groups={"clinet:acme": "hard"})))

    assert "'clinet:acme'" in lines
    assert "no members" in lines
    assert "hard" in lines


# -- scan: provenance before registration ----------------------------------
def _owned_corpus(base):
    """Two repos owned by ACME and one by a stranger.

    Two, not one: `infer_identity` refuses a tie, so a corpus of one repo each
    yields no confident identity and classifies everything as `me` -- which
    passes an assertion about `mine` while proving nothing about `theirs`.
    """
    corpus = base / "corpus"
    corpus.mkdir()
    made = []
    for name, origin in (("mine", "git@github.com:ACME/mine.git"),
                         ("mine2", "https://github.com/acme/mine2.git"),
                         ("theirs", "https://github.com/stranger/x.git")):
        r = _repo(corpus, name, origin=origin)
        # Long enough to clear MIN_CONTENT_WORDS, so `loci setup` has something
        # to index: a shorter README is discarded as a stub and the scope then
        # has nothing indexable at all, which aborts the run before its report.
        (r / "README.md").write_text(
            f"# {name}\n\nSigns the admin session cookie with a rotating HMAC "
            f"key and refuses SameSite None on localhost.\n", encoding="utf-8")
        made.append(r)
    return (corpus, *made)


def test_scan_summary_groups_repositories_by_owner(tmp_path, hermetic_git):
    from loci.provenance import Identity
    from loci.setup import group_summary

    _corpus, mine, mine2, theirs = _owned_corpus(tmp_path)

    summary = group_summary([make_scope(mine), make_scope(mine2),
                             make_scope(theirs)],
                            Identity(org="acme", email="me@example.com",
                                     confident=True))

    assert {s.name for s in summary["me"]} == {"mine", "mine2"}
    assert {s.name for s in summary["vendor:stranger"]} == {"theirs"}


def test_scan_reports_the_split_and_labels_it_with_nobody_to_ask(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """Non-interactive keeps today's behaviour -- register everything -- but
    silence is the bug, not the registration. The report is printed either way,
    and what was registered carries its provenance label.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus, *_ = _owned_corpus(tmp_path)

    assert main(["scan", str(corpus)]) == 0
    out = capsys.readouterr().out

    assert "you are acme" in out
    assert "vendor:stranger" in out and "theirs" in out
    registered = {s.name: s.group_set() for s in load_scopes()}
    assert set(registered) == {"mine", "mine2", "theirs"}, "silence, not registration"
    assert registered["theirs"] == {"vendor:stranger"}
    assert registered["mine"] == {"me"}


def test_scan_puts_a_vendor_group_to_a_vote_and_never_your_own(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """The asymmetry is the flow. The user asked to scan their own projects, so
    a prompt about `me` has one honest answer and is worth deleting; a stranger's
    repository entering the corpus is the decision they were never offered.

    What was declined is then reported. Registering in silence is the failure
    this feature closes, and declining in silence is the same failure inverted:
    the corpus is quietly smaller than the scan found and nothing says so.
    """
    import builtins
    import sys as _sys

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus, *_ = _owned_corpus(tmp_path)

    class _Tty:
        def isatty(self):
            return True

    asked: list[str] = []
    monkeypatch.setattr(_sys, "stdin", _Tty())
    monkeypatch.setattr(builtins, "input",
                        lambda prompt="": (asked.append(prompt), "n")[1])

    assert main(["scan", str(corpus)]) == 0
    assert len(asked) == 1, f"asked about more than the vendor group: {asked}"
    assert "vendor:stranger" in asked[0]
    assert {s.name for s in load_scopes()} == {"mine", "mine2"}

    out = capsys.readouterr().out
    assert "vendor:stranger: 1 left out" in out
    assert "loci add <path>" in out, "no way back to what was declined"


def test_setup_counts_repositories_and_scopes_separately(tmp_path, monkeypatch,
                                                         capsys):
    """One repository holding four projects registers five scopes, and the line
    called them all repositories -- in the onboarding flow, where a wrong number
    is least likely to be questioned.
    """
    import loci.paths as P
    from loci.setup import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus = tmp_path / "corpus"
    mono = _fake_repo(corpus, "mono", "Signs the admin session cookie with a "
                                      "rotating HMAC key and refuses SameSite "
                                      "None on localhost.")
    (mono / "pkg").mkdir()
    (mono / "pkg" / "package.json").write_text("{}", encoding="utf-8")

    assert run([corpus], assume_yes=True, graphs=False, embed=False,
               calibrate=False, split=True) == 0
    out = capsys.readouterr().out

    assert "1 git repository under" in out
    assert "2 scope(s), 2 new" in out
    assert "2 git repositories" not in out


# -- scopes: inherited groups ----------------------------------------------
def test_subscopes_inherit_the_groups_the_registry_holds(tmp_path):
    """`discover` cannot do this: it reads the filesystem and never the
    registry, so the parent it builds always has `groups is None`. At
    registration time the registry IS in hand, and a user who tagged the
    monorepo `client:acme` needs its sub-projects in `client:acme` too -- that
    is what makes a hard group confine to the client's work rather than to one
    directory of it.

    Both halves of "sub-scope" are load-bearing, and each has a negative here:
    the containment LABEL (Sibling carries it and is nowhere near the parent)
    and the PATH (Nested lies inside the parent and never claimed the label).

    PROVENANCE is never inherited, in either direction: `classify` reads one
    repository's own git, so a vendored repo nested in the user's monorepo is
    still the vendor's, and a monorepo OF vendored code does not make its
    sub-projects the same vendor's. Both are asserted -- dropping either half of
    `is_provenance` leaves the other half's case passing.
    """
    from loci.scopes import inherit_parent_groups

    mono, third = tmp_path / "mono", tmp_path / "third"
    for d in ("client", "vendored", "nested"):
        (mono / d).mkdir(parents=True)
    (third / "sub").mkdir(parents=True)
    (tmp_path / "sibling").mkdir()

    scopes = [Scope(id="mono", name="Mono", root=mono,
                    groups=["mono", "client:acme", "me"]),
              Scope(id="mono-client", name="Mono/client", root=mono / "client",
                    groups=["mono", "me"]),
              # A repository vendored into the monorepo: `third_party/`,
              # `external/`, a submodule carrying a workspace marker.
              Scope(id="mono-vendored", name="Mono/vendored",
                    root=mono / "vendored",
                    groups=["mono", "vendor:stranger"]),
              Scope(id="third", name="Third", root=third,
                    groups=["third", "vendor:alpha"]),
              Scope(id="third-sub", name="Third/sub", root=third / "sub",
                    groups=["third", "vendor:beta"]),
              Scope(id="nested", name="Nested", root=mono / "nested",
                    groups=["me"]),
              Scope(id="sibling", name="Sibling", root=tmp_path / "sibling",
                    groups=["mono"])]

    by_id = {s.id: s.group_set() for s in inherit_parent_groups(scopes)}

    assert by_id["mono-client"] == {"mono", "client:acme", "me"}
    assert by_id["mono-vendored"] == {"mono", "client:acme", "vendor:stranger"}, \
        "a vendored repo was put back in `me` by its container"
    assert by_id["third-sub"] == {"third", "vendor:beta"}, \
        "a container's vendor label was asserted of a sub-project"
    assert by_id["nested"] == {"me"}, "inherited without ever claiming the label"
    assert by_id["sibling"] == {"mono"}, "inherited from a container it is not in"
    assert by_id["mono"] == {"mono", "client:acme", "me"}, "the parent changed"


def test_doctor_prints_the_group_report_it_was_given(loci_home, capsys):
    """Nothing else proves the report is WIRED. `group_report` returning the
    right lines into nobody's hands is the same silence the whole feature is
    against, and every test above it exercises the function directly.

    The exit code stays a COVERAGE verdict -- this fixture's scopes have no
    structure sources, hence 1 -- and a group finding must not change it: a
    vendor scope in the corpus is not a broken install.
    """
    from loci.cli import main
    from loci.scopes import save_scopes

    a, b = loci_home / "a", loci_home / "b"
    a.mkdir()
    b.mkdir()
    save_scopes([Scope(id="alpha", name="Alpha", root=a, groups=["me"]),
                 Scope(id="odysseus", name="Odysseus", root=b,
                       groups=["vendor:stranger"])])
    _install_index(_index(alpha=("Alpha", str(a), {"widget": 40}, 500),
                          odysseus=("Odysseus", str(b), {"flange": 25}, 400)))

    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "\ngroups\n" in out
    # The ID, which is what `loci group rm` takes and what `loci scopes` lists.
    assert "vendor:stranger: odysseus" in out
    assert "loci group set me --mode hard" in out


def test_scan_does_not_let_one_monorepo_outvote_the_corpus(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """git resolves `origin` by walking upward, so every sub-scope of a monorepo
    reports its parent's org again. Inferring identity from scope roots lets one
    foreign repository holding four projects outvote three of the user's own,
    and a wrong identity does not degrade gracefully -- it inverts every
    classification at once, filing the user's own work under `vendor:`.

    5 votes for `big` against 3 for `acme` is exactly that inversion; 1 against
    3 is the answer.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    mono = _repo(corpus, "mono", origin="git@github.com:big/mono.git")
    for sub in ("a", "b", "c", "d"):
        (mono / sub).mkdir()
        (mono / sub / "package.json").write_text("{}", encoding="utf-8")
    for name in ("one", "two", "three"):
        _repo(corpus, name, origin=f"https://github.com/acme/{name}.git")

    assert main(["scan", "--split", str(corpus)]) == 0
    groups = {s.name: s.group_set() for s in load_scopes()}

    assert groups["one"] == {"me"}, "the user's own repository filed as a vendor's"
    assert groups["mono"] == {"mono", "vendor:big"}
    assert groups["mono/a"] == {"mono", "vendor:big"}


def test_scan_reads_who_you_are_from_the_whole_scan_not_the_new_arrivals(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """A re-scan that picks up one repository would infer identity from that
    one repository, whose org is then unanimously "yours" -- so the first
    stranger's project a user clones gets filed as their own work, which is the
    exact registration this feature exists to stop.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus, *_ = _owned_corpus(tmp_path)
    assert main(["scan", str(corpus)]) == 0

    _repo(corpus, "newcomer", origin="https://github.com/outsider/n.git")
    assert main(["scan", str(corpus)]) == 0

    groups = {s.name: s.group_set() for s in load_scopes()}
    assert groups["newcomer"] == {"vendor:outsider"}
    assert groups["mine"] == {"me"}, "the first scan's labels were rewritten"


def test_setup_reports_a_declined_group_in_the_skip_report(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """`skipped:` exists because a setup that silently omits a step leaves the
    user believing they have something they do not. A declined group is exactly
    that: the corpus is smaller than the scan found, and only this line says so
    or names the way back.

    The count is unit-free on purpose. `summary[g]` holds SCOPES, so "(5 repos)"
    was the same miscount the repository line eleven lines above it fixes.
    """
    import builtins
    import sys as _sys

    import loci.paths as P
    from loci.scopes import load_scopes
    from loci.setup import run

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    corpus, *_ = _owned_corpus(tmp_path)

    class _Tty:
        def isatty(self):
            return True

    asked: list[str] = []
    monkeypatch.setattr(_sys, "stdin", _Tty())
    monkeypatch.setattr(builtins, "input",
                        lambda prompt="": (asked.append(prompt), "n")[1])

    assert run([corpus], graphs=False, embed=False, calibrate=False) == 0
    out = capsys.readouterr().out

    assert len(asked) == 1, f"asked about more than the vendor group: {asked}"
    assert "skipped:" in out
    assert ("vendor:stranger (1 left out) loci scan <root>, or loci add <path>"
            in out)
    assert {s.name for s in load_scopes()} == {"mine", "mine2"}


def test_a_vendored_repo_does_not_inherit_its_containers_provenance(
        tmp_path, monkeypatch, hermetic_git, capsys):
    """`vendor/` is in SKIP_DIRS; `third_party/`, `external/`, `packages/<fork>`
    and any submodule carrying a workspace marker are not, so a foreign
    repository really does register as a sub-scope of the user's monorepo.

    `classify` reads its own git and gets it right. Inheriting the container's
    `me` then put it straight back into the routable set -- and `doctor` would
    go on naming `loci group set me --mode hard` as the way to keep it out while
    that command no longer did. The last two assertions are that claim, checked
    rather than assumed.
    """
    import builtins

    import loci.paths as P
    from loci.cli import main
    from loci.groups import Policy, confinement
    from loci.scopes import load_scopes

    monkeypatch.setenv(P.ENV_HOME, str(tmp_path / "home"))
    monkeypatch.setattr(builtins, "input",
                        lambda *a: pytest.fail("scan prompted with no terminal"))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    mine = _repo(corpus, "mine", origin="git@github.com:acme/mine.git")
    _repo(corpus, "mine2", origin="git@github.com:acme/mine2.git")
    mono = _repo(corpus, "mono", origin="git@github.com:acme/mono.git")
    vendored = _repo(mono, "vendored", origin="https://github.com/stranger/v.git")
    (vendored / "package.json").write_text("{}", encoding="utf-8")

    assert main(["scan", "--split", str(corpus)]) == 0
    scopes = load_scopes()
    groups = {s.name: s.group_set() for s in scopes}

    assert groups["mono/vendored"] == {"mono", "vendor:stranger"}
    assert groups["mono"] == {"mono", "me"}, "the container lost its own label"

    conf = confinement(Policy(groups={"me": "hard"}), scopes, cwd=mine)
    assert conf.mode == "hard"
    assert "mono-vendored" not in conf.eligible, \
        "the remedy `doctor` names does not keep the vendor out"
    assert {"mine", "mine2", "mono"} <= conf.eligible
