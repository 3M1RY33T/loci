#!/usr/bin/env python3
"""Run `loci eval` across generated corpus shapes, optionally sweeping a constant.

    python evals/sweep.py                          # every shape, shipped constants
    python evals/sweep.py --const router.SIZE_PRIOR --values 0.05,0.15,0.30
    python evals/sweep.py --shape baseline --json

The output to look for is NOT the best value. It is the WIDTH of the range that
works: a constant peaking at one value on one shape and falling away steeply is
fitted to noise, while one that is flat across a band on every shape is a real
setting. Report the flat region; prefer its middle.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from corpus import CorpusSpec, build, standard_shapes  # noqa: E402

SHAPE_NAMES = ["baseline", "many-scopes", "disjoint", "similar", "size-skew",
               "prose-heavy", "code-heavy", "undocumented", "camelCase",
               "runtogether", "latin", "cjk", "vendored"]


def set_const(dotted: str, value) -> object:
    """Set `module.CONST`, returning the previous value."""
    mod_name, _, const = dotted.rpartition(".")
    mod = importlib.import_module(f"loci.{mod_name}")
    old = getattr(mod, const)
    setattr(mod, const, type(old)(value) if not isinstance(old, bool) else value)
    return old


def run_shape(spec: CorpusSpec, *, with_graphs: bool = False) -> dict:
    """Build one corpus, index it, and score routing on it."""
    from loci.backends import episodes as ep
    from loci.eval import run as eval_run
    from loci.index import build as build_index
    from loci.index import load_index
    from loci.scopes import make_scope, save_scopes

    corpus_root = Path(tempfile.mkdtemp(prefix="loci-sweep-corpus-"))
    home = Path(tempfile.mkdtemp(prefix="loci-sweep-home-"))
    prev_home = os.environ.get("LOCI_HOME")
    os.environ["LOCI_HOME"] = str(home)
    try:
        t = time.time()
        generated = build(spec, corpus_root)
        scopes = [make_scope(g.root, name=g.name) for g in generated]
        save_scopes(scopes)
        if not with_graphs:
            from loci.backends import graphify as G
            G.GraphifyBackend.available = lambda self: False
            G.GraphifyBackend.graph_paths = lambda self, scope: []
        ep.reset_caches()
        build_index(scopes, verbose=False, force=True)
        index = load_index()
        res = eval_run(index)
        fams = res["families"]
        return {
            "scopes": len(index["scopes"]),
            "tokens": len(index["postings"]),
            "secs": round(time.time() - t, 1),
            "cwd": round(fams["cwd"].rate, 3),
            "nocwd": round(fams["nocwd"].rate, 3),
            "nonsense": round(fams["nonsense"].rate, 3),
            "signature": round(fams["signature"].rate, 3),
            "contended": round(fams["contended"].rate, 3),
            "set": round(fams["contended"].set_total / fams["contended"].n, 2)
                   if fams["contended"].n else 0.0,
        }
    finally:
        if prev_home is None:
            os.environ.pop("LOCI_HOME", None)
        else:
            os.environ["LOCI_HOME"] = prev_home
        shutil.rmtree(corpus_root, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", help="dotted constant to sweep, e.g. router.SIZE_PRIOR")
    ap.add_argument("--values", help="comma-separated values for --const")
    ap.add_argument("--shape", action="append", help="limit to named shapes")
    ap.add_argument("--graphs", action="store_true", help="also build code graphs")
    ap.add_argument("--overlap", help="comma-separated overlap values, sweeping difficulty")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.overlap:
        shapes = [(f"ov={o}", CorpusSpec(n_scopes=12, overlap=float(o), seed=11))
                  for o in args.overlap.split(",")]
    else:
        shapes = list(zip(SHAPE_NAMES, standard_shapes()))
    if args.shape and not args.overlap:
        wanted = set(args.shape)
        shapes = [(n, s) for n, s in shapes if n in wanted]
        if not shapes:
            print(f"no shape matched {args.shape}; have: {', '.join(SHAPE_NAMES)}",
                  file=sys.stderr)
            return 1

    values = [None]
    if args.const:
        if not args.values:
            print("--const requires --values", file=sys.stderr)
            return 1
        values = [float(v) if "." in v else int(v) for v in args.values.split(",")]

    rows = []
    hdr = f"{'shape':<14}"
    if args.const:
        hdr += f" {args.const.split('.')[-1]:>10}"
    hdr += (f" {'cwd':>5} {'nocwd':>6} {'nons':>5} {'signat':>7} {'contend':>8} "
            f"{'set':>5} {'secs':>5}")
    if not args.json:
        print(hdr)
        print("-" * len(hdr))

    for name, spec in shapes:
        for v in values:
            old = set_const(args.const, v) if args.const else None
            try:
                r = run_shape(spec, with_graphs=args.graphs)
            finally:
                if args.const:
                    set_const(args.const, old)
            r["shape"] = name
            if args.const:
                r[args.const] = v
            rows.append(r)
            if not args.json:
                line = f"{name:<14}"
                if args.const:
                    line += f" {v:>10}"
                line += (f" {r['cwd']:>5.0%} {r['nocwd']:>6.0%} {r['nonsense']:>5.0%} "
                         f"{r['signature']:>7.0%} {r['contended']:>8.0%} {r['set']:>5.2f} "
                         f"{r['secs']:>5.1f}")
                print(line, flush=True)

    if args.json:
        print(json.dumps(rows, indent=2))
    elif args.const:
        print(f"\nLook for the widest FLAT band, not the peak. A value that wins on one "
              f"shape\nand loses on another is fitted to that shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
