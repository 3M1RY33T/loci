#!/usr/bin/env python3
"""A held-out corpus of real repositories, chosen for variety rather than convenience.

Generated corpora measure whether a constant is STABLE across corpus shapes.
They cannot measure whether it is any GOOD, because their vocabulary is
synthetic and their difficulty is whatever the generator happened to encode --
five separate generator artifacts during Phase 1 each produced a confidently
wrong number, and four of them looked like defects in loci.

This is the other half. Real repositories have vocabulary nobody designed,
documentation written for humans, commit histories with real arguments in them,
and terms that are central to one project and marginal in another. That last
property is what `WIDEN_RATIO` and `SIZE_PRIOR` depend on, and it is exactly
what a generator cannot fake -- Phase 1 measured both as unmeasurable on
synthetic data.

Every entry is a permissively licensed public repository, cloned shallow. The
`shape` field records why it is here: the set is only useful if it spans the
dimensions the generated corpora vary deliberately.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "loci-real-corpus"
CLONE_DEPTH = 50           # enough commit messages to be an episode source


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    shape: str

    @property
    def slug(self) -> str:
        return self.name


# Twenty repositories spanning the dimensions that Phase 1 varies synthetically.
# Deliberately NOT a list of the author's own projects, and deliberately not all
# Python: a tokenizer and a set of thresholds fitted on one language's naming
# conventions is exactly the failure this set exists to detect.
REPOS: list[Repo] = [
    # --- systems / compiled, identifier-dense, sparse prose
    Repo("cjson", "https://github.com/DaveGamble/cJSON", "C, small, terse docs"),
    Repo("sqlite-amalgam", "https://github.com/sqlite/sqlite", "C, huge, dense prose in comments"),
    Repo("ripgrep", "https://github.com/BurntSushi/ripgrep", "Rust, heavy README, benchmarks"),
    Repo("fd", "https://github.com/sharkdp/fd", "Rust, small, good docs"),
    Repo("zstd", "https://github.com/facebook/zstd", "C, polyglot bindings"),

    # --- Go and JVM: different naming conventions
    Repo("hugo", "https://github.com/gohugoio/hugo", "Go, large, docs-adjacent"),
    Repo("cobra", "https://github.com/spf13/cobra", "Go, library, moderate docs"),
    Repo("guava", "https://github.com/google/guava", "Java, camelCase, javadoc-heavy"),

    # --- dynamic languages
    Repo("flask", "https://github.com/pallets/flask", "Python, docs-heavy"),
    Repo("requests", "https://github.com/psf/requests", "Python, small, prose-heavy"),
    Repo("express", "https://github.com/expressjs/express", "JS, minimal, old-style"),
    Repo("svelte", "https://github.com/sveltejs/svelte", "TS monorepo, many packages"),
    Repo("rails", "https://github.com/rails/rails", "Ruby, huge, snake_case"),

    # --- prose-dominant and doc-only
    Repo("public-apis", "https://github.com/public-apis/public-apis", "docs only, no code"),
    Repo("free-programming-books", "https://github.com/EbookFoundation/free-programming-books",
         "docs only, many languages"),
    Repo("rfcs", "https://github.com/rust-lang/rfcs", "prose only, long-form argument"),

    # --- non-English documentation
    Repo("hutool", "https://github.com/dromara/hutool", "Java, Chinese documentation"),
    Repo("vue-zh", "https://github.com/vuejs/docs", "docs, i18n, mixed scripts"),

    # --- adversarial shapes
    Repo("awesome-python", "https://github.com/vinta/awesome-python", "a list, almost no prose"),
    Repo("dotfiles", "https://github.com/mathiasbynens/dotfiles", "shell, tiny, no structure"),
]

SUBSETS = {
    "smoke": ["cjson", "fd", "requests", "public-apis"],
    "polyglot": ["cjson", "ripgrep", "hugo", "guava", "flask", "express", "rails"],
    "prose": ["public-apis", "free-programming-books", "rfcs", "awesome-python"],
    "nonenglish": ["hutool", "vue-zh"],
    "all": [r.name for r in REPOS],
}


def by_name(name: str) -> Repo | None:
    return next((r for r in REPOS if r.name == name), None)


def fetch(repo: Repo, root: Path, *, depth: int = CLONE_DEPTH) -> tuple[bool, str]:
    dest = root / repo.slug
    if (dest / ".git").is_dir():
        return True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(
        ["git", "clone", "--depth", str(depth), "--quiet", repo.url, str(dest)],
        capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        return False, (p.stderr or p.stdout).strip().splitlines()[-1][:120]
    return True, "cloned"


def size_mb(path: Path) -> float:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    except OSError:
        return 0.0


def cmd_fetch(args) -> int:
    root = Path(args.root)
    names = SUBSETS.get(args.subset, [])
    if not names:
        print(f"unknown subset {args.subset}; have: {', '.join(SUBSETS)}", file=sys.stderr)
        return 1
    total = 0.0
    print(f"{'repo':<28} {'status':<10} {'MB':>7}  shape")
    print("-" * 78)
    for name in names:
        repo = by_name(name)
        t = time.time()
        ok, note = fetch(repo, root, depth=args.depth)
        mb = size_mb(root / repo.slug) if ok else 0.0
        total += mb
        status = note if ok else "FAILED"
        print(f"{repo.name:<28} {status:<10} {mb:>7.1f}  {repo.shape}"
              + ("" if ok else f"\n    {note}"), flush=True)
    print("-" * 78)
    print(f"{'total':<28} {'':<10} {total:>7.1f} MB in {root}")
    return 0


def cmd_eval(args) -> int:
    """Index the fetched repos as one corpus and score routing on it."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from loci.backends import episodes as ep
    from loci.eval import render, run
    from loci.index import build as build_index
    from loci.index import load_index
    from loci.scopes import make_scope, save_scopes

    root = Path(args.root)
    names = SUBSETS.get(args.subset, [])
    present = [root / n for n in names if (root / n / ".git").is_dir()]
    if len(present) < 2:
        print(f"need at least 2 fetched repos; run `real_corpus.py fetch --subset "
              f"{args.subset}` first", file=sys.stderr)
        return 1

    home = Path(args.home) if args.home else Path(root) / "_loci_home"
    os.environ["LOCI_HOME"] = str(home)
    if not args.graphs:
        from loci.backends import graphify as G
        G.GraphifyBackend.available = lambda self: False
        G.GraphifyBackend.graph_paths = lambda self, scope: []

    scopes = [make_scope(p, name=p.name) for p in present]
    save_scopes(scopes)
    ep.reset_caches()
    t = time.time()
    build_index(scopes, verbose=not args.quiet, force=args.force)
    index = load_index()
    print(f"\nindexed {len(index['scopes'])} real repos in {time.time() - t:.0f}s")
    result = run(index)
    print()
    print(render(result, show_misses=args.misses))
    if args.json:
        print(json.dumps({k: {"n": f.n, "rate": f.rate}
                          for k, f in result["families"].items()}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("list", help="show the manifest")
    f.set_defaults(func=lambda a: (print(f"{'repo':<28} {'subset':<12} shape"),
                                   print("-" * 78),
                                   [print(f"{r.name:<28} "
                                          f"{','.join(k for k, v in SUBSETS.items() if k != 'all' and r.name in v) or '-':<12} "
                                          f"{r.shape}") for r in REPOS], 0)[-1])

    f = sub.add_parser("fetch", help="shallow-clone a subset")
    f.add_argument("--subset", default="smoke", choices=list(SUBSETS))
    f.add_argument("--depth", type=int, default=CLONE_DEPTH)
    f.set_defaults(func=cmd_fetch)

    f = sub.add_parser("eval", help="index the fetched repos and score routing")
    f.add_argument("--subset", default="smoke", choices=list(SUBSETS))
    f.add_argument("--graphs", action="store_true")
    f.add_argument("--misses", action="store_true")
    f.add_argument("--force", action="store_true")
    f.add_argument("--quiet", action="store_true")
    f.add_argument("--home")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
