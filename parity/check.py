"""Replay the corpus and diff, under a split tolerance.

Two policies, because two different things are being checked:

  exact    the DECISION -- verdict, scope set, ordering, mode, reasons.
           A routing answer is a choice, and a choice is either the same
           choice or a regression. Ordering is part of it: an agent reads
           the first hit.

  1e-6     the SCORE. Floating-point arithmetic reassociates when the code
           computing it changes, and a difference in the seventh decimal is
           noise. A difference big enough to move an ordering shows up under
           the exact policy instead, which is why the two are separable.

The failure this file exists to prevent is a comparator that passes
everything: it would report a clean port that isn't one, and every later
phase would build on the report.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parity.record import _flags, _run, filename_for, load_manifest

TOLERANCE = 1e-6

# Fields describing HOW an answer was printed rather than WHAT it was. A
# dropped progress bar is the goal of Phase 0b, not a regression.
IGNORED_SUFFIXES = ("stdout_len", "stderr_tail")

_INDEX = re.compile(r"\[\d+\]$")


def _ignored(path: str) -> bool:
    """Whether this path names presentation rather than answer.

    The index is stripped first. The real path is `ask.stderr_tail[0]`, not
    `ask.stderr_tail`, and an endswith check that forgets that compares a torch
    progress bar's throughput -- which differs on every single run. It cost a
    6/39 on the first replay of the corpus against the build that produced it.
    """
    return _INDEX.sub("", path).endswith(IGNORED_SUFFIXES)


def _sort_edge_runs(text: str) -> str:
    """Sort each contiguous run of EDGE lines in a graph block.

    graphify emits them in a different order run to run: replaying the corpus
    against the very build that recorded it produced identical NODE lines, in
    identical order, and a permuted EDGE block. Two runs of one query against
    one unchanged graph disagreeing is proof the order carries no information,
    so comparing it would fail every port for a reason no port caused.

    NODE order is stable -- it is the BFS order, and it IS the answer -- so
    only EDGE runs are touched.
    """
    if "\nEDGE " not in text:
        return text
    lines = text.split("\n")
    out, run = [], []
    for line in lines:
        if line.startswith("EDGE "):
            run.append(line)
            continue
        if run:
            out.extend(sorted(run)); run = []
        out.append(line)
    if run:
        out.extend(sorted(run))
    return "\n".join(out)


@dataclass(frozen=True)
class Diff:
    path: str
    kind: str          # "exact" | "numeric" | "missing"
    expected: object
    actual: object

    def __str__(self) -> str:
        return (f"  {self.kind:<8} {self.path}\n"
                f"      expected: {self.expected!r}\n"
                f"      actual:   {self.actual!r}")


def compare(expected, actual, path: str = "") -> list[Diff]:
    """Structural diff. Floats get the tolerance; everything else is exact."""
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            if math.isclose(float(expected), float(actual),
                            rel_tol=0.0, abs_tol=TOLERANCE):
                return []
        except (TypeError, ValueError):
            pass
        return [Diff(path, "numeric", expected, actual)]

    if type(expected) is not type(actual):
        return [Diff(path, "exact", expected, actual)]

    if isinstance(expected, dict):
        out: list[Diff] = []
        for k in sorted(set(expected) | set(actual)):
            if k not in expected or k not in actual:
                out.append(Diff(f"{path}.{k}", "missing",
                                expected.get(k, "<absent>"),
                                actual.get(k, "<absent>")))
            else:
                out.extend(compare(expected[k], actual[k], f"{path}.{k}"))
        return out

    if isinstance(expected, list):
        # Never sorted before comparing. Order is the answer.
        if len(expected) != len(actual):
            return [Diff(path, "exact", f"len={len(expected)}", f"len={len(actual)}")]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out.extend(compare(e, a, f"{path}[{i}]"))
        return out

    if isinstance(expected, str):
        # Compared normalised, reported raw: the reader needs to see reality.
        if _sort_edge_runs(expected) == _sort_edge_runs(actual):
            return []
        return [Diff(path, "exact", expected, actual)]

    return [] if expected == actual else [Diff(path, "exact", expected, actual)]


def run(corpus: Path) -> int:
    """Replay the recorded corpus against the CURRENT build. Returns exit code."""
    manifest = load_manifest(corpus)
    # Replay under the instant the corpus was recorded at, or every score of a
    # chunk with a timestamp is compared against a clock that has since moved.
    if manifest.get("now"):
        os.environ["LOCI_NOW"] = manifest["now"]
    answers = corpus / "answers"
    failures = 0

    for q in manifest["questions"]:
        rec = json.loads((answers / filename_for(q.id)).read_text(encoding="utf-8"))
        flags = _flags(q)
        live = {
            "route": _run(["route", q.text, "--json", *flags]),
            "ask": _run(["ask", q.text, "--json", *flags]),
        }
        diffs = [d for surface in ("route", "ask")
                 for d in compare(rec[surface], live[surface], surface)
                 if not _ignored(d.path)]
        if diffs:
            failures += 1
            print(f"FAIL {q.id}  ({q.family})")
            for d in diffs[:8]:
                print(d)
            if len(diffs) > 8:
                print(f"      ... and {len(diffs) - 8} more")
        else:
            print(f"ok   {q.id}")

    total = len(manifest["questions"])
    print(f"\n{total - failures}/{total} identical")
    return 1 if failures else 0


if __name__ == "__main__":
    corpus = Path(__file__).resolve().parent / "corpus"
    if not (corpus / "manifest.json").is_file():
        sys.exit("No corpus. Run `python parity/record.py` against 0.5.0 first.")
    sys.exit(run(corpus))
