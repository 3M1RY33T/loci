"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .paths import BuildLock, home


def _load_index_or_die():
    from .index import load_index
    try:
        return load_index()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


# ==========================================================================
def cmd_scan(args) -> int:
    from .scopes import discover, load_scopes, save_scopes, upsert
    roots = [Path(r) for r in (args.roots or [Path.cwd()])]
    found = discover(roots, max_depth=args.depth)
    if not found:
        print("no git repositories found; use `loci add <path>` instead")
        return 1
    existing = load_scopes()
    for sc in found:
        existing = upsert(existing, sc)
        print(f"  + {sc.name:<20} {sc.root}")
    save_scopes(existing)
    print(f"\n{len(found)} scope(s) registered -> {home()/'scopes.json'}")
    print("next:  loci graphs   # optional, adds code symbols (free)")
    print("       loci index")
    return 0


def cmd_add(args) -> int:
    from .scopes import load_scopes, make_scope, save_scopes, upsert
    from .scopes import DEFAULT_EPISODE_GLOBS
    globs = list(DEFAULT_EPISODE_GLOBS) + list(args.glob or [])
    sc = make_scope(Path(args.path), name=args.name,
                    aliases=args.alias or None, episode_globs=globs)
    save_scopes(upsert(load_scopes(), sc))
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
    for s in scopes:
        print(f"  {s.id:<22} {s.name:<20} {s.root}")
    print(f"\n{len(scopes)} scope(s)")
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
    from .router import route
    index = _load_index_or_die()
    cwd = None if args.no_cwd else (args.cwd or os.getcwd())
    r = route(" ".join(args.question), index, cwd=cwd)
    names = {s: m["name"] for s, m in index["scopes"].items()}
    if args.json:
        print(json.dumps(r.to_json(), indent=2))
        return 0
    if r.abstain:
        print(f"ABSTAIN (matched={r.top_matched}) -> ask the user; "
              f"candidates: {', '.join(names[s] for s in r.ranked)}")
    else:
        print(f"-> {', '.join(names[s] for s in r.selected)}")
    if args.explain:
        print(f"\ntokens: {r.query_tokens}")
        for sid in r.ranked:
            d = r.detail[sid]
            mark = "*" if sid in r.selected else " "
            print(f" {mark} {d['name']:<20} score={d['score']:<9} "
                  f"matched={d['matched']:<3} signals={d['signals'] or '-'}")
            if d["top_tokens"]:
                print(f"      {d['top_tokens']}")
    return 0


def cmd_ask(args) -> int:
    from .ask import ask, render
    from .index import load_episodes
    index = _load_index_or_die()
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
                 force_scopes=forced, index=index, store=store)
    print(json.dumps(answer.to_json(), indent=2) if args.json
          else render(answer, index=index, chars=args.chars))
    return 0


def cmd_graphs(args) -> int:
    import shutil
    import subprocess
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
    failed = []
    for sc in todo:
        print(f"  {sc.name} ... ", end="", flush=True)
        try:
            p = subprocess.run(["graphify", "update", str(sc.root)],
                               capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"timed out after {args.timeout}s"); failed.append(sc.name); continue
        if p.returncode != 0:
            print("failed"); failed.append(sc.name); continue
        nodes = sb.node_count(sc)
        print(f"{nodes} symbols" if nodes else "no symbols found")
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    print("\nnext: loci index")
    return 1 if failed else 0


def cmd_doctor(args) -> int:
    from .doctor import check, render
    from .index import embeddings_status, load_episodes
    from .scopes import load_scopes
    index = _load_index_or_die()
    healths = check(index, load_episodes(), load_scopes())
    print(render(healths, embeddings_status()))
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
    cal = fit(index)
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
    p = argparse.ArgumentParser(
        prog="loci",
        description="Scoped memory for coding agents: a router in front of a "
                    "structure store and an episode store.")
    p.add_argument("--version", action="version", version=f"loci {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="discover git repos under roots and register them")
    s.add_argument("roots", nargs="*", type=Path)
    s.add_argument("--depth", type=int, default=2)
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("add", help="register one scope explicitly")
    s.add_argument("path", type=Path)
    s.add_argument("--name")
    s.add_argument("--alias", action="append")
    s.add_argument("--glob", action="append",
                   help="extra episode glob; absolute paths allowed for external vaults")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("scopes", help="list registered scopes")
    s.set_defaults(func=cmd_scopes)

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
