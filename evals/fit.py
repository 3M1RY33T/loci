#!/usr/bin/env python3
"""Phase 3: fit the constants that were never fitted, across both test beds.

Eight constants shipped with no evidence behind them at all -- chosen because
they seemed reasonable, with no alternative ever tested. This sweeps each one
over BOTH beds at once, because Phase 1 established that neither alone is
sufficient:

    synthetic corpora   say whether a value is STABLE across corpus shapes, but
                        are too clean to exercise anything that depends on two
                        scopes scoring close together
    real repositories   have vocabulary nobody designed and terms that are
                        central to one project and marginal in another, which is
                        the only place widening and size-bias are visible

What to read: the widest FLAT band, not the peak. A value that wins on one bed
and loses on the other is fitted to that bed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from corpus import CorpusSpec, build as build_corpus  # noqa: E402
from real_corpus import DEFAULT_ROOT, SUBSETS  # noqa: E402

SYNTHETIC = {
    "syn-baseline": CorpusSpec(n_scopes=12, overlap=0.3, seed=11),
    "syn-similar": CorpusSpec(n_scopes=12, overlap=0.8, seed=11),
    "syn-skew": CorpusSpec(n_scopes=12, overlap=0.3, size_skew=0.8, seed=11),
    "syn-many": CorpusSpec(n_scopes=30, overlap=0.3, seed=11),
}


def set_const(dotted: str, value):
    mod_name, _, const = dotted.rpartition(".")
    mod = importlib.import_module(f"loci.{mod_name}")
    old = getattr(mod, const)
    setattr(mod, const, type(old)(value))
    return old


def _score(index) -> dict:
    from loci.eval import run
    fams = run(index)["families"]
    scored = {k: round(f.rate, 3) for k, f in fams.items()}
    con = fams.get("contended")
    scored["set"] = round(con.set_total / con.n, 2) if con and con.n else 0.0
    return scored


def build_synthetic(spec: CorpusSpec, home: Path):
    from loci.backends import episodes as ep
    from loci.backends import graphify as G
    from loci.index import build as bi
    from loci.index import load_index
    from loci.scopes import make_scope, save_scopes

    G.GraphifyBackend.available = lambda self: False
    G.GraphifyBackend.graph_paths = lambda self, scope: []
    root = Path(tempfile.mkdtemp(prefix="loci-fit-"))
    gen = build_corpus(spec, root)
    scopes = [make_scope(g.root, name=g.name) for g in gen]
    save_scopes(scopes)
    ep.reset_caches()
    bi(scopes, verbose=False, force=True)
    return load_index(), root


def build_real(subset: str, home: Path, corpus_root: Path):
    from loci.backends import episodes as ep
    from loci.backends import graphify as G
    from loci.index import build as bi
    from loci.index import load_index
    from loci.scopes import make_scope, save_scopes

    G.GraphifyBackend.available = lambda self: False
    G.GraphifyBackend.graph_paths = lambda self, scope: []
    present = [corpus_root / n for n in SUBSETS[subset]
               if (corpus_root / n / ".git").is_dir()]
    if len(present) < 2:
        return None, None
    scopes = [make_scope(p, name=p.name) for p in present]
    save_scopes(scopes)
    ep.reset_caches()
    bi(scopes, verbose=False, force=True)
    return load_index(), None


def beds(real_subsets: list[str], corpus_root: Path):
    """Yield (name, index) for every bed, building each once."""
    for name, spec in SYNTHETIC.items():
        home = Path(tempfile.mkdtemp(prefix="loci-fit-home-"))
        os.environ["LOCI_HOME"] = str(home)
        index, root = build_synthetic(spec, home)
        yield name, index
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)
    for subset in real_subsets:
        home = Path(tempfile.mkdtemp(prefix="loci-fit-home-"))
        os.environ["LOCI_HOME"] = str(home)
        index, _ = build_real(subset, home, corpus_root)
        if index is not None:
            yield f"real-{subset}", index
        shutil.rmtree(home, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--const", required=True)
    ap.add_argument("--values", required=True)
    ap.add_argument("--real", default="polyglot,prose",
                    help="comma-separated real subsets to include")
    ap.add_argument("--corpus-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    values = [float(v) if "." in v else int(v) for v in args.values.split(",")]
    real = [s for s in args.real.split(",") if s]
    rows = []

    name_w = 16
    hdr = (f"{'bed':<{name_w}} {args.const.split('.')[-1]:>12} "
           f"{'cwd':>5} {'nocwd':>6} {'nons':>5} {'signat':>7} {'detail':>7} "
           f"{'contend':>8} {'set':>5}")
    if not args.json:
        print(hdr)
        print("-" * len(hdr))

    for bed_name, index in beds(real, Path(args.corpus_root)):
        for v in values:
            old = set_const(args.const, v)
            try:
                s = _score(index)
            finally:
                set_const(args.const, old)
            rows.append({"bed": bed_name, args.const: v, **s})
            if not args.json:
                print(f"{bed_name:<{name_w}} {v:>12} {s['cwd']:>5.0%} {s['nocwd']:>6.0%} "
                      f"{s['nonsense']:>5.0%} {s['signature']:>7.0%} "
                      f"{s['detailed']:>7.0%} {s['contended']:>8.0%} {s['set']:>5.2f}",
                      flush=True)
        if not args.json:
            print()

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print("Read the widest FLAT band, not the peak. A value that wins on one bed "
              "and\nloses on another is fitted to that bed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
