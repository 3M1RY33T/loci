"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .paths import BuildLock, groups_file, home

# One string, two parsers. `scan` and `setup` reach the same `discover` call and
# a flag that means one thing under one command and another under the other is
# worse than no flag; the modes list in `group set` is derived for the same
# reason. Off by default -- scopes.WORKSPACE_MARKERS records the measurement
# that put it there.
SPLIT_HELP = ("split a monorepo into one scope per depth-1 workspace marker "
              "(package.json, pyproject.toml, Cargo.toml, go.mod). Off by "
              "default; a repo-local .loci.json splits either way.")


def _load_index_or_die():
    from .index import load_index
    try:
        return load_index()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _policy_or_die():
    """The group policy, or a clean error.

    `Policy.__post_init__` refuses an unknown `default_mode`, and `load_policy`
    normalizes every field it reads -- so this catch guards the constructor's
    contract rather than a shape the shipped loader can produce today. It is
    here because `groups.json` is the one file a user is invited to author by
    hand, and no path from a user's own text to a Python traceback is one this
    CLI should have.
    """
    from .groups import load_policy
    try:
        return load_policy()
    except ValueError as exc:
        print(f"error: {groups_file()}: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _registry_or_empty():
    """The scope registry, degrading to [] rather than raising.

    `ask` already does exactly this on a hand-edited `scopes.json`, on the
    ground that a traceback is a worse answer than "no groups configured". The
    commands that report on the same routing decision must not differ, or
    `loci route` crashes where `loci ask` answers.
    """
    from .scopes import load_scopes
    try:
        return load_scopes()
    except Exception:
        return []


def _unknown_group(group: str | None, scopes, policy) -> bool:
    """True -- having said so on stderr -- when `group` names nothing.

    An unknown group resolves to `eligible=set()`, which the router correctly
    reads as "confined to nothing", and the abstention that follows is honest.
    But it is also what a typo produces, and an abstention is a poor way to
    report one. This is the same information one layer up, where the user can
    still see the word they typed.

    A group is known if any scope asserts it OR the policy gives it a mode:
    `loci group set` declares a mode before anyone has joined.
    """
    if not group:
        return False
    known = {g for s in scopes for g in s.group_set()} | set(policy.groups)
    if group in known:
        return False
    print(f"error: unknown group {group!r}; have "
          f"{', '.join(sorted(known)) or '(none)'}", file=sys.stderr)
    return True


# ==========================================================================
def cmd_scan(args) -> int:
    """Register what is under `roots`, having first said who owns it.

    Shares `accept_by_group` with `loci setup` rather than growing a second
    report: the two commands do the same thing to the same registry, and a
    corpus labelled by one and not the other is worse than either.
    """
    from .scopes import (discover, inherit_parent_groups, load_scopes,
                         save_scopes, upsert)
    from .setup import accept_by_group

    roots = [Path(r) for r in (args.roots or [Path.cwd()])]
    found = discover(roots, max_depth=args.depth, markers=args.split)
    if not found:
        print("no git repositories found; use `loci add <path>` instead")
        return 1
    registry = load_scopes()
    known = {s.id for s in registry}
    fresh = [s for s in found if s.id not in known]

    accepted: list = []
    declined: dict = {}
    if fresh:
        accepted, declined = accept_by_group(
            fresh, found, interactive=sys.stdin.isatty())
    else:
        print("  nothing new; every repository here is already registered")

    for sc in found:
        if sc.id in known:
            registry = upsert(registry, sc)
    for sc in accepted:
        registry = upsert(registry, sc)
    save_scopes(inherit_parent_groups(registry))

    print(f"\n{len(registry)} scope(s) registered -> {home()/'scopes.json'}")
    for g, listed in declined.items():
        print(f"  {g}: {len(listed)} left out; `loci scan` again to reconsider, "
              f"or `loci add <path>` for one of them")
    print("next:  loci graphs   # optional, adds code symbols (free)")
    print("       loci index")
    return 0


def cmd_add(args) -> int:
    from .scopes import (PRESERVED_FIELDS, load_scopes, make_scope, save_scopes,
                         upsert)
    from .scopes import DEFAULT_EPISODE_GLOBS
    globs = list(DEFAULT_EPISODE_GLOBS) + list(args.glob or [])
    sc = make_scope(Path(args.path), name=args.name,
                    aliases=args.alias or None, episode_globs=globs)
    # `upsert` keeps a re-scan from clobbering user edits, but a flag here IS
    # the user naming the value, so an explicitly named field must beat the
    # preserved one -- otherwise `loci add --alias` on an already registered
    # path silently does nothing.
    preserve = tuple(f for f in PRESERVED_FIELDS
                     if not (f == "aliases" and args.alias)
                     and not (f == "episode_globs" and args.glob))
    registry = upsert(load_scopes(), sc, preserve=preserve)
    save_scopes(registry)
    # Report the registered scope, not the requested one: the fields the user
    # did not name were carried over, and printing what was asked for would
    # misreport them.
    sc = next(s for s in registry if s.id == sc.id)
    print(f"registered {sc.name} ({sc.id}) -> {sc.root}")
    print(f"  aliases: {', '.join(sc.aliases)}")
    print(f"  episodes: {', '.join(sc.episode_globs)}")
    return 0


def cmd_scopes(args) -> int:
    from .scopes import load_scopes
    scopes = load_scopes()
    if not scopes:
        print("no scopes registered. Try: loci scan ~/code")
        return 1
    if args.group:
        if _unknown_group(args.group, scopes, _policy_or_die()):
            return 2
        scopes = [s for s in scopes if args.group in s.group_set()]
        if not scopes:
            # Known but empty -- a policy may name a group before any scope
            # joins it. Distinct from the unknown-group error above.
            print(f"no scopes in group {args.group}")
            return 1
    for s in scopes:
        print(f"  {s.id:<22} {s.name:<20} {s.root}")
    print(f"\n{len(scopes)} scope(s)")
    return 0


def cmd_groups(args) -> int:
    from .groups import members
    from .scopes import load_scopes

    # `load_scopes`, not the degrading reader: this reports the registry rather
    # than a routing decision, and an unreadable one degraded to [] prints
    # "no groups yet" -- a confident false statement about the user's setup.
    scopes = load_scopes()
    policy = _policy_or_die()
    names = sorted({g for s in scopes for g in s.group_set()} | set(policy.groups))
    if not names:
        print("no groups yet. Try: loci groups infer")
        return 1
    print(f"  {'group':<30} {'mode':<10} {'from':<10} members")
    for g in names:
        # The RESOLVED mode and where it came from, per group. Membership lives
        # in the registry and mode lives in the policy, and this line is the
        # whole answer to "two places to look".
        mode, source = policy.mode_for(g)
        print(f"  {g:<30} {mode:<10} {source:<10} {len(members([g], scopes))}")
    ungrouped = [s.name for s in scopes if not s.group_set()]
    if ungrouped:
        shown = ", ".join(ungrouped[:6])
        more = f" +{len(ungrouped) - 6} more" if len(ungrouped) > 6 else ""
        print(f"\n{len(ungrouped)} scope(s) in no group: {shown}{more}")
    print(f"\ndefault mode: {policy.default_mode}")
    return 0


def cmd_groups_infer(args) -> int:
    """Re-read provenance from git and rewrite the provenance label.

    Identity comes from REPOSITORY roots only, exactly as `accept_by_group`
    reads it. git resolves `origin` by walking upward, so each sub-scope of a
    monorepo votes again for its parent's org and one foreign monorepo outvotes
    the corpus -- and a wrong identity does not degrade gracefully, it inverts
    every classification at once. This command is the one `doctor` tells users
    to run, so it read the registry's scope roots and disagreed with the scan
    that produced them.

    The new label is ADDED; a contradicted `vendor:` label is retracted with it.
    `classify` answers one question -- whose repository is this -- from one
    repository's own git, and answers it afresh on every run, so a registry
    holding `vendor:big` and `vendor:acme` for one scope is not two facts but
    one question answered twice. Pure union also made this command unable to
    repair itself: correcting the identity above and re-running only ever ADDED
    the right label beside the wrong one, leaving recovery manual and per scope.

    `me` is not retracted, and `is_retractable` argues that out: the identity is
    a single dominant org, so the user's own work under a second one classifies
    as `vendor:` and stripping `me` there deletes their repositories from their
    own group. Nothing is retracted at all from an unconfident identity, which
    is `classify`'s guard for `classify`'s reason.

    What the user asserted survives either way: `client:*` is never inferrable
    and structural containment labels are recomputed by `loci scan`, so neither
    is ever in the retracted set. A `vendor:` label asserted by hand with
    `loci group add` is not durable -- like a containment label, it is recomputed
    rather than remembered.
    """
    from .provenance import classify, infer_identity, is_retractable
    from .scopes import load_scopes, repositories, save_scopes

    scopes = load_scopes()
    if not scopes:
        print("no scopes registered. Try: loci scan ~/code", file=sys.stderr)
        return 1
    identity = infer_identity([s.root for s in repositories(scopes)])
    print(f"  {identity.describe()}")
    changed = 0
    for s in scopes:
        g = classify(s.root, identity)
        current = s.group_set()
        stale = {p for p in current if is_retractable(p, identity)} - {g}
        if g in current and not stale:
            continue
        s.groups = sorted((current - stale) | {g})
        changed += 1
        was = f"  (was {', '.join(sorted(stale))})" if stale else ""
        print(f"  + {s.name:<24} {g}{was}")
    save_scopes(scopes)
    print(f"\n{changed} scope(s) labelled -> {home() / 'scopes.json'}")

    # Every run, not only the runs that changed something. Not retracting `me`
    # costs one thing, and this is it: a scope a scan called yours before it
    # could name your org keeps `me` forever, and `me --mode hard` then ADMITS
    # it, because a hard group admits its own members. git cannot tell "my work
    # under my employer's org" from "a stranger's repository mislabelled once",
    # so the command does not guess -- it names both and hands over the one
    # edit that decides. Silence here would leave the escape hatch undiscovered.
    dual = [s for s in scopes if "me" in s.group_set()
            and any(p.startswith("vendor:") for p in s.group_set())]
    if dual:
        listed = ", ".join(f"{s.name} ({v})" for s in dual
                           for v in sorted(p for p in s.group_set()
                                           if p.startswith("vendor:")))
        print(f"\n{len(dual)} scope(s) are in `me` AND a vendor group: {listed}")
        print("  `me` is never retracted here -- your own work under a second "
              "org is still yours. But `loci group set me --mode hard` admits "
              "these, so if one is a stranger's, `loci group rm <scope> me` "
              "settles it; inference will not put `me` back while it can name "
              "your org.")
    return 0


def _containment_holder(sc, group: str, scopes):
    """The scope whose structural containment label `group` is, or None.

    `discover` assigns exactly one structural label, and only in one shape: a
    git REPOSITORY holding depth-1 workspace markers (or `.loci.json`
    declarations) labels each of those sub-projects -- and itself, being "a
    member of its own containment group" -- with its own id. It recomputes that
    on every scan and `upsert` unions it back, so removing one by hand does not
    survive.

    Matched against `subscopes` rather than against "is an ancestor of". Two
    scopes registered independently, one merely nested somewhere under the
    other, have no structural relationship at all: refusing a removal there
    promises a scan that will never restore the label, and leaves the user
    unable to undo their own `group add`. `subscopes` is also what `discover`
    itself calls, so the two cannot disagree about what a sub-project is.

    `markers=True` here against a default of False in `discover`, deliberately.
    The registry does not record which flags produced a label, so the question
    this can answer is not "would the next scan restore it" but "could a scan
    restore it" -- and a marker label exists at all only because some scan ran
    with `--split`, which the same user is liable to run again. The two errors
    are not symmetric: refusing wrongly is visible and says why, allowing
    wrongly is the silent reappearance the guard exists to prevent.
    """
    from .scopes import subscopes

    holder = next((p for p in scopes if p.id == group), None)
    if holder is None:
        return None
    # Only a repository is ever a `parent` in `discover` -- it collects on
    # `(d / ".git").exists()` and reports "no git repositories found" for a tree
    # without one, so a plain directory never hands out its id no matter what it
    # holds. `.exists()` not `.is_dir()`, matching `discover`: a worktree or
    # submodule carries `.git` as a FILE and scan takes it.
    if not (holder.root / ".git").exists():
        return None
    # `if subs:` is the same guard `discover` applies before labelling anything,
    # so a container with nothing inside it never carried the label structurally
    # and must not be told it lives inside itself.
    subs = subscopes(holder.root, markers=True)
    if not subs:
        return None
    if holder.id == sc.id or sc.root.resolve() in subs:
        return holder
    return None


def cmd_group(args) -> int:
    from .groups import MODES, save_policy
    from .scopes import load_scopes, resolve, save_scopes

    if args.action == "set":
        if args.mode not in MODES:
            print(f"error: `group set` needs --mode ({'|'.join(MODES)})",
                  file=sys.stderr)
            return 2
        policy = _policy_or_die()
        policy.groups[args.name] = args.mode
        save_policy(policy)
        print(f"{args.name} -> {args.mode}")
        return 0

    if not args.group:
        print(f"error: `group {args.action}` needs a group: "
              f"loci group {args.action} <scope> <group>", file=sys.stderr)
        return 2

    scopes = load_scopes()
    sc = resolve(scopes, args.name)
    if sc is None:
        print(f"error: unknown scope {args.name!r}", file=sys.stderr)
        return 2
    current = sc.group_set()
    if args.action == "add":
        sc.groups = sorted(current | {args.group})
    else:
        # A containment label is structural: `discover` recomputes it from the
        # filesystem and `upsert` unions it back in, so removing one here
        # reappears at the next scan that sees the same shape. Refuse, not no-op.
        holder = _containment_holder(sc, args.group, scopes)
        if holder is not None:
            where = ("holds sub-projects and is a member of its own containment "
                     "group" if holder.id == sc.id
                     else f"is a sub-project of {holder.name}")
            print(f"error: {args.group!r} is structural -- {sc.name} {where}, and "
                  f"`loci scan` recomputes that label from the filesystem "
                  f"(always for .loci.json, with --split for workspace markers), "
                  f"so removing it here would not survive the next one.",
                  file=sys.stderr)
            return 2
        sc.groups = sorted(current - {args.group})
    # Written straight to the registry, NOT through `upsert`: `upsert` unions
    # `groups` so a re-scan cannot erase a structural label, which also means a
    # removal routed through it removes nothing at all.
    save_scopes(scopes)
    print(f"{sc.name}: {', '.join(sc.groups) or '(no groups)'}")
    return 0


def cmd_index(args) -> int:
    from .index import build
    from .scopes import load_scopes
    scopes = load_scopes()
    if not scopes:
        print("no scopes registered. Try: loci scan ~/code", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"indexing {len(scopes)} scope(s)")
    with BuildLock():
        idx = build(scopes, episodes=not args.no_episodes,
                    episode_vocab=args.episode_vocab, verbose=not args.quiet,
                    force=args.force)
    print(f"\n{len(idx['scopes'])} scopes, {len(idx['postings'])} tokens -> {home()}")
    missing = [m["name"] for m in idx["scopes"].values() if not m.get("sources")]
    if missing:
        shown = ", ".join(missing[:4]) + (f" +{len(missing) - 4} more" if len(missing) > 4 else "")
        print(f"{len(missing)} scope(s) have no code symbols ({shown}) - they route on "
              f"prose alone.\n  `loci graphs` adds them (free, no model calls).")
    if not args.no_episodes:
        from .index import embeddings_status
        stale = embeddings_status()
        if stale is None:
            print("optional next: loci embed   (semantic episode search)")
        elif stale:
            print(f"WARNING: embeddings are stale for {', '.join(stale)} - "
                  f"semantic search is OFF for those scopes until you run `loci embed`.")
    return 0


def cmd_embed(args) -> int:
    from .index import build_embeddings
    try:
        with BuildLock("embed"):
            out = build_embeddings(args.model)
    except ImportError:
        print("error: needs sentence-transformers. pip install 'loci-mem[embeddings]'",
              file=sys.stderr)
        return 2
    print(f"\nvectors -> {out} ({out.stat().st_size // 1024}KB)")
    return 0


def cmd_route(args) -> int:
    from .groups import confinement
    from .router import route
    index = _load_index_or_die()
    policy, registry = _policy_or_die(), _registry_or_empty()
    if _unknown_group(args.group, registry, policy):
        return 2
    cwd = None if args.no_cwd else (args.cwd or os.getcwd())
    # Resolved exactly as `ask` resolves it, from the REGISTRY: the index holds
    # no `groups`, so a `Scope` rebuilt from it can never carry membership.
    conf = confinement(policy, registry, cwd=cwd, forced_group=args.group)
    r = route(" ".join(args.question), index, cwd=cwd,
              eligible=conf.eligible, demoted=conf.demoted,
              strict_group=conf.strict, group=conf.group, mode=conf.mode)
    names = {s: m["name"] for s, m in index["scopes"].items()}
    if args.json:
        print(json.dumps(r.to_json(), indent=2))
        return 0
    if r.abstain:
        # The reason goes on the PLAIN line, not behind --explain: a hard group
        # turns answers into abstentions, and one that does not say why is
        # indistinguishable from a bug. An empty candidate list is dropped
        # rather than printed as a dangling "candidates: ".
        # `eligible` filtered every scope out, so `route` weighed the question
        # against an empty candidate set and reported a reason true of that set
        # and false of the question. `ask.render` makes the same correction;
        # the diagnostic surface must not be the less informative of the two.
        why = (f"no indexed project is in group {r.group}"
               if r.group and not r.ranked else r.abstain_reason)
        cands = ", ".join(names[s] for s in r.ranked)
        print(f"ABSTAIN (matched={r.top_matched}{f', {why}' if why else ''})"
              f" -> ask the user" + (f"; candidates: {cands}" if cands else ""))
    else:
        print(f"-> {', '.join(names[s] for s in r.selected)}")
    if args.explain:
        print()
        if r.group:
            # `conf.source` rather than a re-derivation from `r.mode`: every
            # confined question has a mode, so "does it have one" cannot tell a
            # declared mode from the default it fell back to.
            print(f"group:  {r.group}  mode={r.mode} ({conf.source})")
        print(f"tokens: {r.query_tokens}")
        for sid in r.ranked:
            d = r.detail[sid]
            mark = "*" if sid in r.selected else " "
            print(f" {mark} {d['name']:<20} score={d['score']:<9} "
                  f"matched={d['matched']:<3} signals={d['signals'] or '-'}")
            if d["top_tokens"]:
                print(f"      {d['top_tokens']}")
        # `ranked` is filtered to the eligible set, so under a group the scope
        # that CAUSED an out_of_group abstention is absent from the very output
        # meant to explain it. `detail` still holds every scope it scored.
        outside = sorted((s for s in r.detail if s not in r.ranked),
                         key=lambda s: -r.detail[s]["score"])
        if outside:
            print(f"  -- excluded by group {r.group} --")
            for sid in outside:
                d = r.detail[sid]
                print(f" x {d['name']:<20} score={d['score']:<9} "
                      f"matched={d['matched']:<3} signals={d['signals'] or '-'}")
    return 0


def cmd_ask(args) -> int:
    from .ask import ask, render
    from .index import load_episodes
    index = _load_index_or_die()
    policy, registry = _policy_or_die(), _registry_or_empty()
    if _unknown_group(args.group, registry, policy):
        return 2
    if args.scope and args.group:
        # An explicit --scope outranks group policy: the user has already
        # answered the question groups exist to answer. Said out loud because
        # the line above VALIDATES a flag this call is about to discard, and
        # validating something you then ignore is what makes silence misleading.
        print(f"note: --scope overrides --group {args.group}", file=sys.stderr)
    store = load_episodes()
    forced = None
    if args.scope:
        by_name = {m["name"].lower(): s for s, m in index["scopes"].items()}
        forced = []
        for want in args.scope:
            sid = want if want in index["scopes"] else by_name.get(want.lower())
            if not sid:
                print(f"error: unknown scope {want!r}; have "
                      f"{', '.join(m['name'] for m in index['scopes'].values())}",
                      file=sys.stderr)
                return 2
            forced.append(sid)
    if args.fast:
        # Loading sentence_transformers costs ~1.6s of import plus model
        # construction, which dominates a single CLI question.
        from .backends import episodes as _ep
        _ep._EMB = {}
    cwd = None if args.no_cwd else (args.cwd or os.getcwd())
    answer = ask(" ".join(args.question), cwd=cwd, budget=args.budget,
                 episodes_k=args.k, dfs=args.dfs, rerank=args.rerank,
                 with_structure=not args.no_structure,
                 with_episodes=not args.no_episodes,
                 force_scopes=forced, group=args.group,
                 policy=policy, registry=registry, index=index, store=store)
    print(json.dumps(answer.to_json(), indent=2) if args.json
          else render(answer, index=index, chars=args.chars))
    return 0


def cmd_graphs(args) -> int:
    import shutil
    from .backends import get_structure_backend
    from .scopes import load_scopes

    if not shutil.which("graphify"):
        print("error: graphify is not on PATH. pip install 'loci-mem[graphify]'",
              file=sys.stderr)
        return 2
    scopes = load_scopes()
    if not scopes:
        print("no scopes registered. Try: loci scan ~/code", file=sys.stderr)
        return 1
    if args.scope:
        from .scopes import resolve
        picked = []
        for want in args.scope:
            sc = resolve(scopes, want)
            if not sc:
                print(f"error: unknown scope {want!r}", file=sys.stderr)
                return 2
            picked.append(sc)
        scopes = picked
    sb = get_structure_backend()
    todo = [s for s in scopes if args.all or not sb.sources(s)]
    if not todo:
        print("every scope already has a structure graph. Nothing to do.")
        return 0

    print(f"building structure graphs for {len(todo)} scope(s) "
          f"(AST-only, no model calls, no API cost)")
    from .setup import build_graphs
    failed = build_graphs(sb, todo, timeout=args.timeout)
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    print("\nnext: loci index")
    return 1 if failed else 0


def cmd_setup(args) -> int:
    from .setup import run
    try:
        return run(args.roots, assume_yes=args.yes, graphs=args.graphs,
                   embed=args.embed, calibrate=args.calibrate, depth=args.depth,
                   model=args.model, force=args.force, timeout=args.timeout,
                   split=args.split)
    except KeyboardInterrupt:
        # Every step persists as it finishes, and `loci index` reuses scopes
        # whose fingerprint is unchanged, so a re-run resumes rather than
        # repeating. Say so: the alternative is a user who assumes the half-run
        # left something broken.
        print("\naborted. Finished steps are already on disk; re-run "
              "`loci setup` to pick up from there.")
        return 130


def cmd_doctor(args) -> int:
    from .doctor import check, group_report, render
    from .index import embeddings_status, load_episodes
    from .scopes import load_scopes
    index = _load_index_or_die()
    scopes = load_scopes()
    healths = check(index, load_episodes(), scopes)
    print(render(healths, embeddings_status()))
    # Coverage is one gap; who a scope belongs to and what that does to routing
    # is the other, and no other command reports the second without being asked.
    lines = group_report(scopes, _policy_or_die())
    if lines:
        print("\ngroups")
        for ln in lines:
            print(f"  {ln}")
    return 0 if all(h.ok for h in healths) else 1


def cmd_eval(args) -> int:
    from .eval import render, run
    index = _load_index_or_die()
    if not index["scopes"]:
        print("no scopes indexed. Try: loci scan ~/code && loci index", file=sys.stderr)
        return 1
    result = run(index)
    if args.json:
        print(json.dumps({
            "n_scopes": result["n_scopes"], "chance": result["chance"],
            "families": {k: {"name": f.name, "n": f.n, "correct": f.correct,
                             "rate": f.rate, "abstained": f.abstained,
                             "misses": f.misses}
                         for k, f in result["families"].items()}}, indent=2))
        return 0
    print(render(result, show_misses=args.misses))
    fams = result["families"]
    return 0 if fams["cwd"].rate >= 0.95 else 1


def cmd_calibrate(args) -> int:
    from .calibrate import fit, load, render, save
    index = _load_index_or_die()
    if args.show:
        cal = load()
        print(render(cal) if cal else "not calibrated yet. Run: loci calibrate")
        return 0 if cal else 1
    if len(index["scopes"]) < 2:
        print("error: calibration needs at least 2 scopes to compare", file=sys.stderr)
        return 1
    from .index import load_episodes
    cal = fit(index, load_episodes())
    if not args.dry_run:
        save(cal)
    print(render(cal))
    print(f"\n{'saved to ' + str(__import__('loci.calibrate', fromlist=['x']).calibration_file()) if not args.dry_run else '(dry run, not saved)'}")
    return 0 if cal.trustworthy else 1


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import main as mcp_main
    except ImportError:
        print("error: needs the mcp package. pip install 'loci-mem[mcp]'", file=sys.stderr)
        return 2
    return mcp_main()


# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    from .groups import MODES

    p = argparse.ArgumentParser(
        prog="loci",
        description="Scoped memory for coding agents: a router in front of a "
                    "structure store and an episode store.")
    p.add_argument("--version", action="version", version=f"loci {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="one-shot: scan, graph, index, embed, calibrate")
    s.add_argument("roots", nargs="*", type=Path,
                   help="where to look for git repos (prompts if omitted)")
    s.add_argument("-y", "--yes", action="store_true",
                   help="take every default without prompting")
    s.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=None,
                   help="build structure graphs (default: ask)")
    s.add_argument("--embed", action=argparse.BooleanOptionalAction, default=None,
                   help="build local embeddings (default: ask)")
    s.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=None,
                   help="fit routing thresholds (default: ask)")
    s.add_argument("--depth", type=int, default=2, help="how deep to scan for repos")
    s.add_argument("--model", help="embedding model (default: bge-small)")
    s.add_argument("--force", action="store_true",
                   help="rebuild everything, even where nothing changed")
    s.add_argument("--timeout", type=int, default=600, help="per-scope graph timeout")
    s.add_argument("--split", action=argparse.BooleanOptionalAction, default=False,
                   help=SPLIT_HELP)
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("scan", help="discover git repos under roots and register them")
    s.add_argument("roots", nargs="*", type=Path)
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--split", action=argparse.BooleanOptionalAction, default=False,
                   help=SPLIT_HELP)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("add", help="register one scope explicitly")
    s.add_argument("path", type=Path)
    s.add_argument("--name")
    s.add_argument("--alias", action="append")
    s.add_argument("--glob", action="append",
                   help="extra episode glob; absolute paths allowed for external vaults")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("scopes", help="list registered scopes")
    s.add_argument("--group", help="restrict to one group of projects")
    s.set_defaults(func=cmd_scopes)

    s = sub.add_parser("groups", help="list groups, their mode, and members")
    gsub = s.add_subparsers(dest="groups_cmd")
    gi = gsub.add_parser("infer", help="label scopes by git provenance")
    gi.set_defaults(func=cmd_groups_infer)
    s.set_defaults(func=cmd_groups)

    s = sub.add_parser("group", help="edit group membership or policy")
    s.add_argument("action", choices=["set", "add", "rm"])
    s.add_argument("name", help="group name for `set`, scope name for `add`/`rm`")
    s.add_argument("group", nargs="?", help="group to add or remove")
    # `choices=MODES`, not a fourth copy of the names. groups.py derives MODES
    # from STRICTNESS for exactly this reason: a mode listed in one place and
    # not the other resolves to a KeyError deep inside `confining_groups`.
    s.add_argument("--mode", choices=MODES,
                   help="for `set`: what this group does at query time")
    s.set_defaults(func=cmd_group)

    s = sub.add_parser("index", help="build the routing index and episode store")
    s.add_argument("-q", "--quiet", action="store_true")
    s.add_argument("--no-episodes", action="store_true")
    s.add_argument("--force", action="store_true",
                   help="re-parse every scope even if nothing changed")
    s.add_argument("--episode-vocab", action="store_true",
                   help="fold episode prose into routing vocabulary (see README)")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("embed", help="encode episode chunks for semantic search")
    s.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    s.set_defaults(func=cmd_embed)

    s = sub.add_parser("route", help="show which scope(s) a question routes to")
    s.add_argument("question", nargs="+")
    s.add_argument("--explain", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--cwd")
    s.add_argument("--no-cwd", action="store_true",
                   help="ignore the working directory (routing gets much worse)")
    s.add_argument("--group", help="restrict to one group of projects")
    s.set_defaults(func=cmd_route)

    s = sub.add_parser("ask", help="route, then query both stores in scope")
    s.add_argument("question", nargs="+")
    s.add_argument("--scope", action="append", help="force a scope, bypassing routing")
    s.add_argument("--budget", type=int, default=2000)
    s.add_argument("-k", type=int, default=3, help="episode hits per scope")
    s.add_argument("--chars", type=int, default=400)
    s.add_argument("--dfs", action="store_true")
    s.add_argument("--rerank", action="store_true", help="cross-encoder rerank (opt-in)")
    s.add_argument("--fast", action="store_true",
                   help="skip semantic ranking; ~5s faster per one-shot invocation")
    s.add_argument("--no-structure", action="store_true")
    s.add_argument("--no-episodes", action="store_true")
    s.add_argument("--json", action="store_true")
    s.add_argument("--cwd")
    s.add_argument("--no-cwd", action="store_true")
    s.add_argument("--group", help="restrict to one group of projects")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("graphs", help="build missing structure graphs via graphify")
    s.add_argument("scope", nargs="*", help="limit to these scopes (default: all missing)")
    s.add_argument("--all", action="store_true", help="rebuild even where one exists")
    s.add_argument("--timeout", type=int, default=600)
    s.set_defaults(func=cmd_graphs)

    s = sub.add_parser("doctor", help="report coverage gaps per scope")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("calibrate", help="fit routing thresholds to YOUR corpus")
    s.add_argument("--show", action="store_true", help="print the current calibration")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("eval", help="measure routing on YOUR corpus (no labels needed)")
    s.add_argument("--misses", action="store_true", help="list what it got wrong")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("mcp", help="run the MCP server on stdio")
    s.set_defaults(func=cmd_mcp)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
