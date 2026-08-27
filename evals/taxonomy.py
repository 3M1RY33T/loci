"""Family A: corpus-independent questions.

A fixed list of things a developer asks about any project, written without
reference to any particular one, then applied mechanically to every scope. No
per-scope selection means no opportunity to pick questions that happen to work.

These are all DEICTIC -- they say "this project", never a name -- which makes
them the honest test of two things at once:

    with cwd     can the router use the working directory? (should be ~100%)
    without cwd  does it correctly refuse to guess?        (should abstain)

Both matter. A router that answers deictic questions without cwd is guessing,
and guessing right sometimes is worse than abstaining, because you cannot tell
the difference from the outside.
"""
from __future__ import annotations

QUESTIONS: list[tuple[str, str]] = [
    ("purpose", "What does this project do and who is it for?"),
    ("tests", "How do I run the tests?"),
    ("deploy", "How is this deployed and what runs it?"),
    ("entry", "What is the entry point and how does it start up?"),
    ("deps", "What external services or libraries does this depend on?"),
    ("risks", "What are the known risks, gaps or limitations here?"),
    ("setup", "How do I set up a development environment?"),
    ("config", "What configuration or environment variables does it need?"),
]


def generate(scope_ids: list[str]) -> list[dict]:
    """One item per (question, scope). Gold is the scope it is asked about."""
    out: list[dict] = []
    for sid in scope_ids:
        for key, q in QUESTIONS:
            out.append({
                "id": f"tax-{key}-{sid}",
                "family": "taxonomy",
                "question": q,
                "gold": [sid],
                "cwd": sid,
                "contamination": "none",
                "justification": "generic developer question, authored without "
                                 "reference to any corpus",
            })
    return out
