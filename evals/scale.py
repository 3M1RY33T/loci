#!/usr/bin/env python3
"""How does routing hold up as the number of scopes grows?

Ten real projects cannot answer that, so this generates synthetic ones with the
shape that matters: a distinctive domain vocabulary per scope, plus a large
shared pool of ordinary engineering English that every scope also uses. The
shared pool is the point -- routing is easy when vocabularies are disjoint, and
the real question is whether a discriminative term still wins once fifty scopes
are all talking about handlers, configs, retries and queues.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Each scope needs its OWN discriminable vocabulary. An earlier version cycled a
# fixed list of ten domains, so at fifty scopes five of them shared identical
# terms and the gold label was genuinely ambiguous -- that measured the
# generator, not the router. Domain words are now suffixed per scope so any
# failure is the router's.
CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"


def coin(rng: random.Random) -> str:
    """A pronounceable pseudo-word, unique per scope, that no corpus shares."""
    return "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(3))


DOMAINS = [
    ("invoice", ["invoice", "billing", "ledger", "dunning", "proration"]),
    ("telemetry", ["telemetry", "histogram", "gauge", "exporter", "otlp"]),
    ("scheduler", ["scheduler", "cron", "backfill", "quartz", "misfire"]),
    ("payroll", ["payroll", "timesheet", "garnishment", "accrual", "w2"]),
    ("routing", ["waypoint", "geofence", "isochrone", "polyline", "gtfs"]),
    ("imaging", ["dicom", "voxel", "radiograph", "windowing", "pacs"]),
    ("ledger", ["settlement", "clearing", "nostro", "iso20022", "swift"]),
    ("catalog", ["sku", "variant", "merchandising", "planogram", "upc"]),
    ("identity", ["saml", "oidc", "assertion", "scim", "passkey"]),
    ("shipping", ["manifest", "bol", "tariff", "incoterm", "dunnage"]),
]
SHARED = ("handler config retry queue worker client server request response cache "
          "timeout error logging metrics database migration schema index test "
          "build deploy release version module service endpoint payload").split()


def make_scope(root: Path, i: int, rng: random.Random,
               overlap: float = 0.0) -> tuple[str, list[str]]:
    """`overlap` is the share of a scope's domain terms it hands to its peers.

    At 0.0 every scope is perfectly discriminable, which is not what real
    projects look like. At 0.8 four of five domain terms are common to every
    scope in that domain, which is closer to a monorepo full of services that
    all talk about the same nouns.
    """
    domain, base = DOMAINS[i % len(DOMAINS)]
    mark = coin(rng)
    n_shared = int(round(overlap * len(base)))
    terms = list(base[:n_shared]) + [f"{t}{mark}" for t in base[n_shared:]]
    name = f"{domain}-{i:03d}"
    d = root / name
    (d / "src").mkdir(parents=True)
    filler = lambda n: " ".join(rng.choice(SHARED) for _ in range(n))
    (d / "README.md").write_text(
        f"# {name}\n\n## Overview\n\nThe {terms[0]} service handles {terms[1]} "
        f"processing. {filler(40)}\n\n## Design\n\nIt models {terms[2]} and "
        f"{terms[3]} explicitly. {filler(40)}\n")
    for j in range(4):
        (d / "src" / f"mod{j}.py").write_text(
            f'"""Module {j} of the {terms[0]} service.\n\n'
            f'Handles {terms[j % len(terms)]} records end to end, including the '
            f'{terms[(j + 1) % len(terms)]} path. {filler(30)}\n"""\n\n\n'
            f'def process_{terms[j % len(terms)]}(payload):\n'
            f'    """Process one {terms[j % len(terms)]} payload. {filler(20)}"""\n'
            f'    return payload\n')
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(d), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "-m", f"add {terms[0]} and {terms[1]} handling"],
                   check=True)
    return name, terms


def main() -> int:
    args = sys.argv[1:]
    overlaps = [0.0, 0.4, 0.6, 0.8]
    sizes = [int(x) for x in args] if args else [50]
    rng = random.Random(7)
    root = Path(tempfile.mkdtemp(prefix="loci-scale-"))
    home = Path(tempfile.mkdtemp(prefix="loci-scale-home-"))
    os.environ["LOCI_HOME"] = str(home)

    from loci.index import build, load_index
    from loci.router import route
    from loci.scopes import make_scope as mk
    from loci.scopes import save_scopes

    print(f"{'scopes':>7} {'overlap':>8} {'index':>8} {'route ms':>9} {'abstain':>8} "
          f"{'top-1':>9} {'set':>6}")
    print(f"{'':>7} {'':>8} {'':>8} {'':>9} {'':>8} {'of routed':>9}")
    print("-" * 60)
    for n in sizes:
      for overlap in overlaps:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        rng = random.Random(7)
        built = [make_scope(root, i, rng, overlap) for i in range(n)]
        scopes = [mk(root / name, name=name) for name, _ in built[:n]]
        save_scopes(scopes)

        t = time.time()
        build(scopes, verbose=False, force=True)
        idx_secs = time.time() - t
        index = load_index()

        hits = widen = sets = abst = 0
        t = time.time()
        for name, terms in built[:n]:
            gold = index_id(index, name)
            q = f"how are {terms[2]} and {terms[3]} records processed?"
            r = route(q, index)
            if r.abstain:
                abst += 1
                continue
            sets += len(r.selected)
            hits += r.ranked[0] == gold
            widen += gold in r.selected
        route_ms = (time.time() - t) / n * 1000
        routed = n - abst
        # Abstention is reported separately from error: at high overlap the
        # question genuinely cannot single out a scope, and refusing is right.
        print(f"{n:>7} {overlap:>8.1f} {idx_secs:>7.1f}s {route_ms:>8.1f} "
              f"{abst / n:>8.1%} {(hits / routed if routed else 0):>9.1%} "
              f"{(sets / routed if routed else 0):>6.2f}")

    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(home, ignore_errors=True)
    return 0


def index_id(index: dict, name: str) -> str:
    for sid, m in index["scopes"].items():
        if m["name"] == name:
            return sid
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
