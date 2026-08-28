"""One command that takes a fresh install to a working memory.

The build steps are a dependency chain, not a preference:

    graphs   -> index      the routing index is built FROM graph.json, so
                           indexing first leaves every project routing on prose
    index    -> embed      embed encodes episode chunks, which do not exist
                           until index has written the store
    embed    -> calibrate  calibrate fits a per-scope semantic floor, and with
                           no vectors to fit it from it falls back to a default

Running that order wrong does not fail loudly. It produces an install that
works and quietly retrieves worse -- which is the failure `doctor` exists to
name, so setup ends by running it.

Prompts appear only where the answer is genuinely the user's: which directories
hold their projects, and whether to spend a model download on semantic search.
Indexing is not offered as a choice, because nothing works without it. When
stdin is not a terminal every prompt takes its default rather than blocking,
because an agent or a CI job running `loci setup` must not hang on a question
nobody is there to answer.
"""
from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path

from .paths import BuildLock, home

# Directories people actually keep repositories in. Probed rather than assumed:
# an empty prompt is a worse first experience than a wrong default the user can
# overtype, and a wrong default is worse than one that was checked to exist.
CANDIDATE_ROOTS = (
    "code", "Code", "src", "dev", "Developer", "projects", "Projects",
    "repos", "workspace", "git", "Documents/GitHub", "Documents/code",
)

MAX_LISTED = 12


def suggest_roots() -> list[Path]:
    h = Path.home()
    return [h / c for c in CANDIDATE_ROOTS if (h / c).is_dir()]


# ---------------------------------------------------------------------------
# prompting
# ---------------------------------------------------------------------------
def _confirm(question: str, *, default: bool, interactive: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    if not interactive:
        print(f"  {question} [{hint}] -> {'yes' if default else 'no'}")
        return default
    while True:
        try:
            raw = input(f"  {question} [{hint}] ").strip().lower()
        except EOFError:
            print()
            return default
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  answer y or n")


def _prompt_paths(question: str, default: list[Path], *,
                  interactive: bool) -> list[Path]:
    shown = " ".join(str(p) for p in default) or "(none)"
    if not interactive:
        print(f"  {question} -> {shown}")
        return default
    print(f"  {question}")
    try:
        raw = input(f"  [{shown}] ").strip()
    except EOFError:
        print()
        return default
    if not raw:
        return default
    # shlex, not split(): a path with a space in it is normal on macOS and
    # Windows, and silently scanning two nonexistent halves of one directory
    # would look like "found 0 repositories".
    return [Path(p).expanduser() for p in shlex.split(raw)]


# ---------------------------------------------------------------------------
# who owns what
# ---------------------------------------------------------------------------
def group_summary(scopes: list, identity) -> dict:
    """Discovered scopes bucketed by inferred provenance group."""
    from .provenance import classify

    out: dict = {}
    for s in scopes:
        out.setdefault(classify(s.root, identity), []).append(s)
    return out


def accept_by_group(fresh: list, found: list, *,
                    interactive: bool) -> tuple[list, dict]:
    """Print who owns what, ask about what is not yours, label what was taken.

    `found` is everything the scan turned up and `fresh` the subset not already
    registered. Identity is inferred from all of it -- one new repository is no
    evidence of who you are -- and from its REPOSITORY roots only, because git
    resolves `origin` by walking upward: counting sub-scopes lets one monorepo
    outvote the corpus and label the user's own work as a vendor's.

    Returns (accepted, declined-by-group). Shared by `loci setup` and
    `loci scan`, which differ only in what they do with the answer.

    Registering in silence is the failure this exists to close -- a stranger's
    repository entered the corpus with no label and competed for every question
    -- so the split is printed whether or not anyone is there to read it. Only
    the QUESTION depends on a terminal, and with none it takes its default,
    which is still "register everything".

    `me` is not put to a vote. The user asked to scan their own projects, and a
    prompt whose only honest answer is yes is a prompt worth deleting.
    """
    from .provenance import infer_identity
    from .scopes import repositories

    identity = infer_identity([s.root for s in repositories(found)])
    print(f"  {identity.describe()}")
    summary = group_summary(fresh, identity)
    for g in sorted(summary):
        listed = summary[g]
        print(f"\n  {g:<28} {len(listed)} new")
        for s in listed[:MAX_LISTED]:
            print(f"    + {s.name:<22} {s.root}")
        if len(listed) > MAX_LISTED:
            print(f"    ... and {len(listed) - MAX_LISTED} more")

    accepted: list = []
    declined: dict = {}
    for g in sorted(summary):
        if g == "me" or _confirm(f"register {g} ({len(summary[g])})?",
                                 default=True, interactive=interactive):
            for s in summary[g]:
                # Labelled here rather than at discovery: provenance is read
                # from git, and only what was accepted is ever registered.
                s.groups = sorted(s.group_set() | {g})
            accepted += summary[g]
        else:
            declined[g] = summary[g]
    return accepted, declined


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def build_graphs(sb, scopes: list, *, timeout: int = 600) -> list[str]:
    """Run graphify's extractor over each scope. Returns the names that failed.

    Shared with `loci graphs` so the two cannot drift: one progress format, one
    definition of what counts as a failure.
    """
    failed: list[str] = []
    for sc in scopes:
        print(f"  {sc.name:<22} ", end="", flush=True)
        ok, detail = sb.build(sc, timeout=timeout)
        if not ok:
            print(detail)
            failed.append(sc.name)
            continue
        nodes = sb.node_count(sc)
        print(f"{nodes} symbols" if nodes else "no symbols found")
    return failed


def run(roots: list[Path] | None = None, *, assume_yes: bool = False,
        graphs: bool | None = None, embed: bool | None = None,
        calibrate: bool | None = None, depth: int = 2,
        model: str | None = None, force: bool = False,
        timeout: int = 600, split: bool = False) -> int:
    from .index import DEFAULT_EMBED_MODEL
    from .scopes import (discover, inherit_parent_groups, load_scopes,
                         repositories, save_scopes, upsert)

    interactive = sys.stdin.isatty() and not assume_yes
    model = model or DEFAULT_EMBED_MODEL
    # (what was not done, the command that does it later). Printed at the end:
    # a setup that silently skips a step leaves the user believing they have
    # semantic search when they do not.
    skipped: list[tuple[str, str]] = []

    print(f"loci setup  ->  {home()}")
    if not interactive:
        print("non-interactive: taking the default at every prompt")

    # -- 1. projects -------------------------------------------------------
    print("\n[1/5] projects")
    registry = load_scopes()
    want = [Path(r).expanduser() for r in (roots or [])]
    if not want:
        if registry:
            print(f"  {len(registry)} project(s) already registered")
            if _confirm("scan for more?", default=False, interactive=interactive):
                want = _prompt_paths("where do your projects live?  (space separated)",
                                     suggest_roots() or [Path.cwd()],
                                     interactive=interactive)
        else:
            want = _prompt_paths("where do your projects live?  (space separated)",
                                 suggest_roots() or [Path.cwd()],
                                 interactive=interactive)

    if want:
        found = discover(want, max_depth=depth, markers=split)
        repos = repositories(found)
        known = {s.id for s in registry}
        fresh = [s for s in found if s.id not in known]
        where = ", ".join(str(p) for p in want)
        print(f"  {len(repos)} git repositor{'y' if len(repos) == 1 else 'ies'} "
              f"under {where} -> {len(found)} scope(s), {len(fresh)} new")
        if not found:
            print("  nothing there looks like a git repository. `loci add <path>` "
                  "registers a directory that is not one.")
        else:
            accepted: list = []
            if fresh:
                accepted, declined = accept_by_group(fresh, found,
                                                    interactive=interactive)
                for g, listed in declined.items():
                    # "left out", not "repos": `listed` is `summary[g]` and
                    # holds SCOPES, so a declined vendor monorepo would print
                    # "(5 repos)" for one repository -- the same miscount the
                    # count above exists to fix, eleven lines away from it.
                    skipped.append((f"{g} ({len(listed)} left out)",
                                    "loci scan <root>, or loci add <path>"))
            # Scopes already known are re-upserted so `upsert` carries their
            # preserved fields forward; a declined vendor group is simply never
            # added.
            for s in found:
                if s.id in known:
                    registry = upsert(registry, s)
            for s in accepted:
                registry = upsert(registry, s)
            save_scopes(inherit_parent_groups(registry), roots=want)

    if not registry:
        print("\n  no projects registered, and loci needs at least one:")
        print("    loci setup ~/code        scan a directory for git repos")
        print("    loci add <path>          register one directory explicitly")
        return 1
    print(f"  {len(registry)} project(s) registered -> {home() / 'scopes.json'}")

    # -- 2. structure graphs ----------------------------------------------
    print("\n[2/5] structure graphs   (what calls what)")
    from .backends import get_structure_backend
    sb = get_structure_backend()
    if not shutil.which("graphify"):
        print("  graphify is not installed; every project will route on prose alone.")
        print("  later:  pip install 'loci-mem[graphify]' && loci graphs")
        skipped.append(("structure graphs",
                        "pip install 'loci-mem[graphify]' && loci graphs"))
    else:
        todo = list(registry) if force else [s for s in registry if not sb.sources(s)]
        if not todo:
            print("  every project already has a graph")
        elif graphs is False:
            print("  skipped (--no-graphs)")
            skipped.append(("structure graphs", "loci graphs"))
        elif graphs or _confirm(
                f"build graphs for {len(todo)} project(s)?  "
                f"(AST only -- no model calls, no API cost)",
                default=True, interactive=interactive):
            failed = build_graphs(sb, todo, timeout=timeout)
            if failed:
                print(f"  {len(failed)} failed: {', '.join(failed)}")
                skipped.append((f"{len(failed)} graph(s)", "loci graphs"))
        else:
            skipped.append(("structure graphs", "loci graphs"))

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
    print("\n[4/5] semantic search   (optional)")
    import importlib.util
    if importlib.util.find_spec("sentence_transformers") is None:
        print("  sentence-transformers is not installed; episode search stays lexical.")
        print("  later:  pip install 'loci-mem[embeddings]' && loci embed")
        skipped.append(("embeddings",
                        "pip install 'loci-mem[embeddings]' && loci embed"))
    elif embed is False:
        print("  skipped (--no-embed)")
        skipped.append(("embeddings", "loci embed"))
    elif embed or _confirm(
            f"encode episodes with {model}?  (one-time ~130MB model download; "
            f"finds answers that share no words with the question)",
            default=True, interactive=interactive):
        from .index import build_embeddings
        with BuildLock("embed"):
            out = build_embeddings(model)
        print(f"  vectors -> {out} ({out.stat().st_size // 1024}KB)")
    else:
        skipped.append(("embeddings", "loci embed"))

    # -- 5. calibration ----------------------------------------------------
    print("\n[5/5] calibration   (optional)")
    if len(idx["scopes"]) < 2:
        print("  needs 2+ projects to compare; keeping the built-in default floor")
    elif calibrate is False:
        print("  skipped (--no-calibrate)")
        skipped.append(("calibration", "loci calibrate"))
    elif calibrate or _confirm(
            "fit the routing thresholds to this corpus?  (seconds, no model calls)",
            default=True, interactive=interactive):
        from .calibrate import fit, save
        from .calibrate import render as render_cal
        from .index import load_episodes
        cal = fit(idx, load_episodes())
        save(cal)
        print("\n".join(("  " + ln).rstrip() for ln in render_cal(cal).splitlines()))
    else:
        skipped.append(("calibration", "loci calibrate"))

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

    print("\nready. Ask from inside a project -- the working directory is the")
    print("strongest routing signal there is, not a tiebreaker:")
    print('    cd <a project> && loci ask "how is this deployed?"')
    print('    loci ask "..." --fast     skip semantic ranking, ~5s faster')
    print("    loci mcp                  serve ask/scopes/doctor to an agent")
    return 0
