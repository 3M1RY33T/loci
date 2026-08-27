#!/usr/bin/env python3
"""Generate synthetic corpora with controlled shape.

Constants in loci were fitted against ten repositories belonging to one person.
The question this exists to answer is not "what value is best" -- a single
corpus can answer that and be wrong -- but "over what range of corpus shapes
does a value hold". So every dimension here is one that plausibly changes the
right answer:

    n_scopes        routing is a comparison between scopes
    overlap         how much vocabulary same-domain scopes share
    size_skew       the real corpus spans 82 to 6,469 tokens, a 79x range
    prose_ratio     a docs monorepo and a C library sit at opposite ends
    doc_density     some projects have no README at all
    naming          snake_case vs camelCase changes what the tokenizer sees
    charset         the tokenizer splits on word characters; CJK has none
    vendor_noise    vendored and generated files that are not the user's code

Synthetic corpora are for measuring STABILITY across shapes. Two synthetic-corpus
bugs during development produced confidently wrong numbers -- a domain list that
cycled, making five scopes identical, and a term sampler that only drew a scope's
rarest words, putting the fitted band two points too high. Whether a value is any
GOOD is decided against real repositories, not these.
"""
from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Ordinary engineering English that every project uses. Routing is easy when
# vocabularies are disjoint; this is what makes the problem real.
SHARED = ("handler config retry queue worker client server request response cache "
          "timeout error logging metrics database migration schema index test build "
          "deploy release version module service endpoint payload session token "
          "buffer stream socket thread process memory disk network parser format").split()

DOMAINS = ["invoice", "telemetry", "scheduler", "payroll", "waypoint", "dicom",
           "settlement", "merchandising", "assertion", "manifest", "cadence",
           "rollup", "shard", "envelope", "quorum", "beacon", "lattice", "prism",
           "harbor", "meridian"]

# Domain-flavoured vocabulary that EVERY scope uses when overlap > 0. Distinct
# from SHARED (ordinary engineering English) because these read like real domain
# nouns and so compete with a scope's own terms rather than being filtered as
# noise.
COMMON_DOMAIN = ["record", "batch", "policy", "ledger", "entry", "account",
                 "transfer", "audit", "receipt", "posting"]

CONSONANTS = "bdfgklmnprstvz"
VOWELS = "aeiou"

# Accented Latin and CJK. The tokenizer uses a Unicode word-character class, so
# accented Latin should survive and CJK -- which has no word boundaries -- should
# not. That is a limitation worth measuring rather than assuming.
LATIN_MARKS = "áéíóúñçãõäöüß"
CJK = "設定処理管理検索記録監視配信認証暗号"

NAMING = {
    "snake": lambda parts: "_".join(parts),
    "camel": lambda parts: parts[0] + "".join(p.capitalize() for p in parts[1:]),
    "kebab": lambda parts: "-".join(parts),
    "flat": lambda parts: "".join(parts),
}

VENDOR_DIRS = ["node_modules/pkg", "vendor/lib", "third_party/dep"]


@dataclass
class CorpusSpec:
    n_scopes: int = 12
    overlap: float = 0.3
    size_skew: float = 0.0
    prose_ratio: float = 0.5
    doc_density: float = 1.0
    naming: str = "snake"
    charset: str = "ascii"
    vendor_noise: float = 0.0
    pair_share: float = 0.2
    seed: int = 0

    def label(self) -> str:
        return (f"n={self.n_scopes} ov={self.overlap} pair={self.pair_share} "
                f"skew={self.size_skew} "
                f"prose={self.prose_ratio} docs={self.doc_density} "
                f"name={self.naming} chars={self.charset} vendor={self.vendor_noise}")


@dataclass
class GeneratedScope:
    name: str
    root: Path
    terms: list[str] = field(default_factory=list)


def _coin(rng: random.Random, charset: str) -> str:
    base = "".join(rng.choice(CONSONANTS) + rng.choice(VOWELS) for _ in range(3))
    if charset == "latin":
        return base + rng.choice(LATIN_MARKS)
    if charset == "cjk":
        return "".join(rng.choice(CJK) for _ in range(2))
    if charset == "mixed":
        return base + (rng.choice(LATIN_MARKS) if rng.random() < 0.5 else "")
    return base


def _size_multiplier(i: int, n: int, skew: float) -> int:
    """1x for a uniform corpus, up to ~80x for the most skewed scope."""
    if skew <= 0 or n < 2:
        return 1
    rank = i / (n - 1)
    return max(1, int(round(1 + skew * rank * 79)))


def build(spec: CorpusSpec, root: Path) -> list[GeneratedScope]:
    rng = random.Random(spec.seed)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    name_fn = NAMING.get(spec.naming, NAMING["snake"])
    out: list[GeneratedScope] = []

    for i in range(spec.n_scopes):
        domain = DOMAINS[i % len(DOMAINS)]
        mark = _coin(rng, spec.charset)
        # `overlap` draws from a pool shared by EVERY scope, not just same-domain
        # ones. An earlier version shared terms only within a domain, and with
        # more domains than scopes each scope held a unique domain -- so the
        # parameter did nothing and the whole test bed was insensitive: sweeping
        # SIZE_PRIOR from 0.0 to 1.2 moved no metric at all, on a constant that
        # demonstrably matters on real corpora.
        # Ten terms, not five. With five, `overlap` moved in 20% steps and
        # produced a cliff rather than a gradient: at 0.7 a scope had exactly one
        # distinctive term while a signature question needs two, so the metric
        # went from 100% to 0% with nothing in between and no constant could be
        # seen to matter anywhere along it.
        base = [f"{domain}", f"{domain}_record", f"{domain}_index",
                f"{domain}_batch", f"{domain}_policy", f"{domain}_queue",
                f"{domain}_state", f"{domain}_event", f"{domain}_rule",
                f"{domain}_view"]
        n_shared = int(round(spec.overlap * len(base)))
        shared = [COMMON_DOMAIN[k % len(COMMON_DOMAIN)] for k in range(n_shared)]
        terms = shared + [f"{t}{mark}" for t in base[n_shared:]]
        # Terms held by exactly TWO scopes. Real corpora are full of these --
        # two services that both talk to the same queue, two projects that share
        # a deployment target -- and a generator that shares a term with every
        # scope or none cannot produce one. Without them the `contended` family
        # has nothing to score, and the constants governing how wide a result
        # set should be stay invisible.
        n_pair = int(round(spec.pair_share * len(base)))
        pair_terms: list[str] = []
        if n_pair and spec.n_scopes > 1:
            # Pair scope i with its neighbour, both sides deriving the same mark
            # so the term really is shared by exactly two.
            partner = i + 1 if i % 2 == 0 else i - 1
            if partner < spec.n_scopes:
                pair_mark = _coin(random.Random(spec.seed * 1000 + min(i, partner)),
                                  spec.charset)
                pair_terms = [f"link{k}{pair_mark}" for k in range(n_pair)]
        terms += pair_terms
        # Interleave. Concatenating put every shared term first, and the README
        # only ever uses terms[0..3] -- so past 50% overlap the prose contained
        # nothing distinctive at all while the unique terms sat in docstrings,
        # which are excluded from routing vocabulary by design. The metric then
        # fell off a cliff instead of degrading, and no constant could be seen
        # to matter along the way.
        rng.shuffle(terms)

        # Neutral names on purpose. Naming a scope after its own domain term
        # makes that term an alias, and the benchmark excludes aliases so a
        # question containing one does not route trivially. The shapes here are
        # meant to vary vocabulary overlap and corpus shape -- not whether the
        # scope name happens to collide with the vocabulary under test.
        name = f"svc-{i:03d}"
        d = root / name
        (d / "src").mkdir(parents=True)
        mult = _size_multiplier(i, spec.n_scopes, spec.size_skew)
        filler = lambda k: " ".join(rng.choice(SHARED) for _ in range(k))

        if rng.random() < spec.doc_density:
            body = [f"# {name}", "", "## Overview", "",
                    f"The {terms[0]} service owns {terms[1]} handling. {filler(60 * mult)}"]
            if pair_terms:
                # Pair terms must reach PROSE. Left only in docstrings they never
                # enter the routing vocabulary -- docstrings are excluded from it
                # by design -- so the contended family found nothing to score.
                body += ["", "## Integrations", "",
                         f"Shares {' and '.join(pair_terms)} with a sibling "
                         f"service. {filler(30 * mult)}"]
            if spec.prose_ratio > 0.3:
                body += ["", "## Design", "",
                         f"It models {terms[2]} and {terms[3]}. {filler(80 * mult)}"]
            (d / "README.md").write_text("\n".join(body) + "\n", encoding="utf-8")

        n_code = max(1, int(round((1 - spec.prose_ratio) * 8 * mult)))
        for j in range(n_code):
            t = terms[j % len(terms)]
            fn = name_fn(["process", t.replace("-", "_")])
            (d / "src" / f"mod{j}.py").write_text(
                f'"""Module {j} of the {terms[0]} service.\n\n'
                f'Handles {t} end to end including the {terms[(j + 1) % len(terms)]} '
                f'path. {filler(30)}\n"""\n\n\n'
                f'def {fn}(payload):\n'
                f'    """Process one {t} payload. {filler(20)}"""\n'
                f'    return payload\n', encoding="utf-8")

        if spec.vendor_noise > 0:
            n_vendor = int(round(spec.vendor_noise * n_code * 3))
            for k in range(n_vendor):
                vd = d / VENDOR_DIRS[k % len(VENDOR_DIRS)]
                vd.mkdir(parents=True, exist_ok=True)
                (vd / f"lib{k}.py").write_text(
                    f'"""Third-party library {k}. Not written by this project.\n\n'
                    f'{filler(60)}\n"""\n\n\ndef helper_{k}(x):\n    return x\n',
                    encoding="utf-8")

        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(d), "-c", "user.email=t@t",
                        "-c", "user.name=t", "commit", "-q", "-m",
                        f"add {terms[0]} handling and {terms[1]} storage"],
                       check=True, capture_output=True)
        out.append(GeneratedScope(name=name, root=d, terms=terms))
    return out


# Shapes worth sweeping over. Each varies ONE dimension from the baseline, so a
# constant that moves can be attributed to the dimension that moved.
def standard_shapes() -> list[CorpusSpec]:
    base = dict(n_scopes=12, overlap=0.3, seed=11)
    return [
        CorpusSpec(**base),                                    # baseline
        CorpusSpec(**{**base, "n_scopes": 30}),                # more scopes
        CorpusSpec(**{**base, "overlap": 0.0}),                # disjoint vocab
        CorpusSpec(**{**base, "overlap": 0.7}),                # similar projects
        CorpusSpec(**{**base, "size_skew": 0.8}),              # one giant scope
        CorpusSpec(**{**base, "prose_ratio": 0.9}),            # docs-heavy
        CorpusSpec(**{**base, "prose_ratio": 0.1}),            # code-heavy
        CorpusSpec(**{**base, "doc_density": 0.3}),            # mostly undocumented
        CorpusSpec(**{**base, "naming": "camel"}),             # camelCase
        CorpusSpec(**{**base, "naming": "flat"}),              # runtogether
        CorpusSpec(**{**base, "charset": "latin"}),            # accented Latin
        CorpusSpec(**{**base, "charset": "cjk"}),              # CJK identifiers
        CorpusSpec(**{**base, "vendor_noise": 1.0}),           # vendored files
    ]
