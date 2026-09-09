"""Record what Python 0.5.0 answers, before anything moves.

Invokes the CLI as a subprocess rather than importing loci, deliberately: the
CLI and the MCP server are the only two surfaces the Rust port must preserve
(Delroy never imports loci, it shells out with --json), so the oracle should
grade the surface rather than the internals behind it.

Run with LOCI_HOME pointed at the frozen snapshot:

    . parity/env.sh && python parity/record.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parity.questions import Question, generate

TIMEOUT = 300


def filename_for(question_id: str) -> str:
    """Question id -> a name every platform in the CI matrix can create.

    A colon is a legal POSIX filename character and an illegal Windows one, and
    the suite already runs on Linux, Windows and macOS. `+` is legal everywhere
    but reads as a path operator in enough tooling to be worth not having.
    """
    return question_id.replace(":", "__").replace("+", "-") + ".json"


def _flags(q: Question) -> list[str]:
    if q.no_cwd:
        return ["--no-cwd"]
    return ["--cwd", q.cwd] if q.cwd else []


def _run(args: list[str]) -> dict:
    """One CLI invocation, captured whole.

    A non-zero exit is recorded rather than raised: "0.5.0 fails this way" is a
    fact the port must reproduce, and dropping it would let the port turn a
    clean error into a wrong answer unnoticed.
    """
    p = subprocess.run([sys.executable, "-m", "loci.cli", *args],
                       capture_output=True, text=True, timeout=TIMEOUT)
    try:
        payload = json.loads(p.stdout) if p.stdout.strip() else None
    except json.JSONDecodeError:
        payload = None
    return {"argv": args, "returncode": p.returncode,
            "json": payload, "stdout_len": len(p.stdout),
            "stderr_tail": p.stderr.strip().splitlines()[-1:] or []}


def pin_clock() -> str:
    """Freeze scoring time for this process and every child, and return it.

    Episode scores carry a recency term, so a recorded answer decays against
    the wall clock -- see `episodes._now`. Recording and replaying under one
    pinned instant is what makes the corpus compare ranking against ranking.
    """
    from datetime import datetime, timezone
    stamp = os.environ.get("LOCI_NOW") or datetime.now(timezone.utc).isoformat()
    os.environ["LOCI_NOW"] = stamp
    return stamp


def write_manifest(out_dir: Path, questions: list[Question], *,
                   loci_version: str, index_version: int, now: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps({
        "loci_version": loci_version,
        "index_version": index_version,
        "loci_home": os.environ.get("LOCI_HOME", ""),
        "now": now,
        "questions": [q.to_json() for q in questions],
    }, indent=2), encoding="utf-8")


def load_manifest(out_dir: Path) -> dict:
    d = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    d["questions"] = [Question.from_json(q) for q in d["questions"]]
    return d


def record(out_dir: Path) -> int:
    """Record every question. Returns how many were written.

    The manifest carries the resolved questions so `check.py` replays exactly
    these rather than regenerating -- a regenerated set could differ if the
    index moved, and then the comparison means nothing.
    """
    from loci.index import INDEX_VERSION, load_index

    index = load_index()
    questions = generate(index)
    answers = out_dir / "answers"
    answers.mkdir(parents=True, exist_ok=True)

    import loci
    write_manifest(out_dir, questions,
                   loci_version=getattr(loci, "__version__", "unknown"),
                   index_version=INDEX_VERSION, now=pin_clock())

    for i, q in enumerate(questions, 1):
        flags = _flags(q)
        rec = {
            "question": q.to_json(),
            "route": _run(["route", q.text, "--json", *flags]),
            "ask": _run(["ask", q.text, "--json", *flags]),
        }
        (answers / filename_for(q.id)).write_text(
            json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  [{i:>3}/{len(questions)}] {q.id}", flush=True)

    # Corpus-wide surfaces, recorded once rather than per question.
    (out_dir / "global.json").write_text(json.dumps({
        "doctor": _run(["doctor"]),
        "scopes": _run(["scopes"]),
        "uses": _run(["uses"]),
    }, indent=2, sort_keys=True), encoding="utf-8")

    return len(questions)


if __name__ == "__main__":
    if not os.environ.get("LOCI_HOME"):
        sys.exit("LOCI_HOME is unset. Run `. parity/env.sh` first.")
    out = Path(__file__).resolve().parent / "corpus"
    print(f"clock pinned to {pin_clock()}")
    n = record(out)
    print(f"\nrecorded {n} questions to {out}")
