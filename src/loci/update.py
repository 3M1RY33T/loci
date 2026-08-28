"""Refresh a memory that already exists.

`setup` and `update` run the same five steps and differ only in what they
assume. Setup assumes nothing is there: it asks where the projects live, offers
every optional step, and treats a model download as a reasonable thing to
propose. Update assumes a working install, so it prompts for nothing, opts you
into nothing you did not already have, and refreshes what is there.

The one thing it rebuilds unconditionally is the structure graph, because that
is the only staleness in loci that is otherwise silent. `loci graphs` builds a
graph that is MISSING and skips every scope that already has one; `loci setup`
does the same unless forced. So a project whose code moved since the day it was
registered keeps routing on the symbols it had that day, and nothing says so.
Every other kind of drift already announces itself: `index` re-parses a scope
whose fingerprint moved, and `doctor` names embeddings that no longer line up
with the store.

Rebuilding every graph is affordable because graphify does not rewrite a graph
whose code did not change -- measured on this repository, a second consecutive
`graphify update` left graph.json byte-identical with its mtime untouched.
loci's own fingerprint reads that mtime, so refreshing the graphs does not
force a reindex of the scopes that stood still.

Prompting is left to setup for the same reason. The answers here are already
determined by what is on disk -- embeddings get refreshed because embeddings
exist -- and a question whose only honest answer is "the same as last time" is
a question worth deleting. The exception is a repository that is not yours
turning up in a rescan, which is asked exactly as `scan` asks it.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .paths import BuildLock, home

MAX_NAMED = 6


def _named(scopes: list, limit: int = MAX_NAMED) -> str:
    shown = ", ".join(s.name for s in scopes[:limit])
    return shown + (f" +{len(scopes) - limit} more" if len(scopes) > limit else "")


def run(roots: list[Path] | None = None, *, assume_yes: bool = False,
        scan: bool | None = None, graphs: bool | None = None,
        embed: bool | None = None, calibrate: bool | None = None,
        depth: int = 2, model: str | None = None, force: bool = False,
        timeout: int = 600, split: bool = False) -> int:
    from .index import DEFAULT_EMBED_MODEL
    from .scopes import (discover, inherit_parent_groups, load_roots,
                         load_scopes, repositories, save_scopes, upsert)
    from .setup import accept_by_group, build_graphs

    interactive = sys.stdin.isatty() and not assume_yes
    model = model or DEFAULT_EMBED_MODEL
    # (what was not refreshed, the command that refreshes it). An update that
    # quietly leaves a step out is worse than setup doing the same: the user
    # ran this one BECAUSE they believed something was out of date.
    skipped: list[tuple[str, str]] = []

    registry = load_scopes()
    if not registry:
        print("nothing to update: no projects are registered yet.")
        print("  loci setup ~/code        scan a directory and build everything")
        print("  loci add <path>          register one directory explicitly")
        return 1

    print(f"loci update  ->  {home()}")

    # -- 1. projects -------------------------------------------------------
    print("\n[1/5] projects")
    given = [Path(r).expanduser() for r in (roots or [])]
    remembered = [] if given else load_roots()
    want = given or remembered
    if scan is False:
        want = []
        skipped.append(("scan for new projects", "loci update"))
    # A root recorded months ago may be gone, unmounted, or on a disk that is
    # not plugged in. Scanning it would report "0 repositories" and read as an
    # empty directory rather than an absent one, so it is named and dropped
    # here instead -- and left in the registry, because next week it is back.
    gone_roots = [p for p in want if not p.is_dir()]
    want = [p for p in want if p.is_dir()]

    if want:
        where = ", ".join(str(p) for p in want)
        print(f"  scanning {where}"
              f"{'   (remembered from your last scan)' if remembered else ''}")
        found = discover(want, max_depth=depth, markers=split)
        repos = repositories(found)
        known = {s.id for s in registry}
        fresh = [s for s in found if s.id not in known]
        print(f"  {len(repos)} git repositor{'y' if len(repos) == 1 else 'ies'} "
              f"-> {len(found)} scope(s), {len(fresh)} new")
        accepted: list = []
        if fresh:
            accepted, declined = accept_by_group(fresh, found,
                                                 interactive=interactive)
            for g, listed in declined.items():
                skipped.append((f"{g} ({len(listed)} left out)",
                                "loci update, or loci add <path>"))
        # Scopes already known are re-upserted so `upsert` carries their
        # preserved fields forward; a declined group is simply never added.
        for s in found:
            if s.id in known:
                registry = upsert(registry, s)
        for s in accepted:
            registry = upsert(registry, s)
        save_scopes(inherit_parent_groups(registry), roots=want)
        registry = load_scopes()
    elif scan and not want:
        print("  --scan, but there is no directory to scan and none recorded; "
              "`loci update <root>` names one")
    elif scan is not False:
        print("  no scan root recorded, so no new projects were looked for")
        print("  `loci update <root>` scans one and remembers it")
        skipped.append(("scan for new projects", "loci update <root>"))
    else:
        print("  skipped (--no-scan); refreshing what is already registered")

    for p in gone_roots:
        print(f"  scan root {p} is not there right now; left recorded, not scanned")

    # A project deleted from disk keeps its vocabulary in the index and keeps
    # competing for every question until something says so. Reported rather
    # than removed: the registry entry carries groups and aliases the user set
    # by hand, and dropping those on a directory that is merely unmounted today
    # is the more expensive mistake.
    gone = [s for s in registry if not Path(s.root).is_dir()]
    if gone:
        print(f"  {len(gone)} registered project(s) no longer on disk: {_named(gone)}")
        print("  left registered and left alone; `doctor` reports them as "
              "coverage gaps until you drop them from scopes.json")
    print(f"  {len(registry)} project(s) registered")

    # -- 2. structure graphs ----------------------------------------------
    print("\n[2/5] structure graphs   (what calls what)")
    from .backends import get_structure_backend
    sb = get_structure_backend()
    if graphs is False:
        print("  skipped (--no-graphs); every project keeps the symbols it was "
              "last built with")
        skipped.append(("structure graphs", "loci graphs --all"))
    elif not shutil.which("graphify"):
        print("  graphify is not installed; every project routes on prose alone.")
        print("  later:  pip install 'loci-mem[graphify]' && loci graphs --all")
        skipped.append(("structure graphs",
                        "pip install 'loci-mem[graphify]' && loci graphs --all"))
    else:
        live = [s for s in registry if Path(s.root).is_dir()]
        print(f"  rebuilding {len(live)} graph(s) -- AST only, no model calls, "
              f"and graphify skips one whose code did not change")
        failed = build_graphs(sb, live, timeout=timeout)
        if failed:
            print(f"  {len(failed)} failed: {', '.join(failed)}")
            skipped.append((f"{len(failed)} graph(s)", "loci graphs --all"))

    # -- 3. index ----------------------------------------------------------
    print("\n[3/5] index   (routing + episode store)")
    from .index import build as build_index
    with BuildLock():
        idx = build_index(registry, verbose=True, force=force)
    if not idx["scopes"]:
        print("\n  nothing was indexable in any project. `loci doctor` says why.")
        return 1
    print(f"  {len(idx['scopes'])} scope(s), {len(idx['postings'])} tokens")

    # -- 4. semantic search ------------------------------------------------
    print("\n[4/5] semantic search")
    import importlib.util

    from .paths import embeddings_file
    have_vectors = embeddings_file().is_file()
    # Refresh what exists; do not opt anyone in. An update that answers "yes"
    # to a question nobody asked spends a 130MB download on a machine whose
    # owner chose lexical search, and spends it in a command they ran to keep
    # what they had working.
    if embed is False:
        print("  skipped (--no-embed)")
        skipped.append(("embeddings", "loci embed"))
    elif not (embed or have_vectors):
        print("  no embeddings to refresh; `loci embed` turns semantic search on")
        skipped.append(("embeddings", "loci embed"))
    elif importlib.util.find_spec("sentence_transformers") is None:
        print("  sentence-transformers is not installed; episode search stays lexical.")
        print("  later:  pip install 'loci-mem[embeddings]' && loci embed")
        skipped.append(("embeddings",
                        "pip install 'loci-mem[embeddings]' && loci embed"))
    else:
        print(f"  re-encoding every chunk with {model}")
        from .index import build_embeddings
        with BuildLock("embed"):
            out = build_embeddings(model)
        print(f"  vectors -> {out} ({out.stat().st_size // 1024}KB)")

    # -- 5. calibration ----------------------------------------------------
    print("\n[5/5] calibration")
    from .calibrate import calibration_file
    have_cal = calibration_file().is_file()
    if calibrate is False:
        print("  skipped (--no-calibrate)")
        skipped.append(("calibration", "loci calibrate"))
    elif len(idx["scopes"]) < 2:
        print("  needs 2+ projects to compare; keeping the built-in default floor")
    elif not (calibrate or have_cal):
        print("  never calibrated; `loci calibrate` fits the floor to this corpus")
        skipped.append(("calibration", "loci calibrate"))
    else:
        # The floor measures how much vocabulary the corpus shares, so a corpus
        # that just gained or lost a project is measuring something else until
        # this runs again.
        from .calibrate import fit, save
        from .calibrate import render as render_cal
        from .index import load_episodes
        cal = fit(idx, load_episodes())
        save(cal)
        print("\n".join(("  " + ln).rstrip() for ln in render_cal(cal).splitlines()))

    # -- what you actually ended up with -----------------------------------
    print("\n" + "=" * 74)
    from .doctor import check
    from .doctor import render as render_doctor
    from .index import embeddings_status, load_episodes
    print(render_doctor(check(idx, load_episodes(), registry), embeddings_status()))

    if skipped:
        print("\nskipped:")
        for what, how in skipped:
            print(f"  {what:<22} {how}")
    return 0
