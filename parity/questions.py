"""A question set GENERATED from the frozen index, never written down.

Hand-written questions would name real projects, several of them private, in a
tracked file -- the same problem that put `evals/` in .gitignore. Generating
from `scope_index.json` at record time keeps every project name in the ignored
corpus directory instead.

Determinism is the other requirement. `record.py` runs against Python 0.5.0 and
`check.py` runs against whatever replaces it, and a set that differs between
the two runs compares nothing. Every sort here carries a lexical tie-break for
that reason, and `record.py` writes the resolved questions beside the
recordings so `check.py` replays those rather than regenerating.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

FAMILIES = ("scoped", "cross", "enumerative", "deictic", "no_evidence", "relational")


@dataclass(frozen=True)
class Question:
    id: str
    family: str
    text: str
    cwd: str | None = None
    no_cwd: bool = False

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Question":
        return Question(**d)


def distinctive_terms(index: dict, scope_id: str, k: int = 4) -> list[str]:
    """Tokens this scope owns outright, most frequent first.

    Sole ownership rather than merely high document frequency: a term two
    scopes share cannot produce a question with one correct verdict, and an
    ambiguous item in the corpus turns a real regression into a judgement call.
    """
    owned = [(post[scope_id], term)
             for term, post in index.get("postings", {}).items()
             if len(post) == 1 and scope_id in post]
    owned.sort(key=lambda x: (-x[0], x[1]))
    return [term for _, term in owned[:k]]


def _scope_ids(index: dict) -> list[str]:
    return sorted(index.get("scopes", {}))


def _by_vocabulary(index: dict) -> list[str]:
    """Scope ids richest in sole-owned terms first, lexical tie-break.

    Alphabetical order put the three THINNEST scopes first -- one of them
    owning a single term outright -- which is backwards for the family that
    needs the strongest signal in the corpus.
    """
    counts = [(len(distinctive_terms(index, sid, k=10_000)), sid)
              for sid in _scope_ids(index)]
    counts.sort(key=lambda x: (-x[0], x[1]))
    return [sid for _, sid in counts]


def generate(index: dict) -> list[Question]:
    """The full corpus, in a stable order.

    Six families, chosen because each is a path that fails SILENTLY -- a wrong
    answer here looks like an answer. Abstention families carry `no_cwd`
    because cwd answers "which project" outright, and handing that to a
    question about refusing to guess tests nothing.
    """
    out: list[Question] = []
    scopes = index.get("scopes", {})

    for sid in _scope_ids(index):
        terms = distinctive_terms(index, sid, k=4)
        if not terms:
            continue
        root = scopes[sid].get("root")
        text = f"how does {' '.join(terms[:3])} work"
        out.append(Question(f"scoped:{sid}", "scoped", text, cwd=root))
        # Same question without cwd: isolates vocabulary routing from location.
        out.append(Question(f"cross:{sid}", "cross", text, no_cwd=True))

    # Enumerative set mode: asks for a SET, and selects two or more scopes far
    # more often than any other path -- which is how the fan-out deadlock
    # became the first thing a user would hit.
    #
    # The two terms MUST come from different scopes. Pairing two terms a single
    # scope owns produces a question whose correct answer is that one scope,
    # which exercises ordinary routing and calls it set mode.
    rich = _by_vocabulary(index)[:3]
    for a, b in zip(rich, rich[1:]):
        ta, tb = distinctive_terms(index, a, k=1), distinctive_terms(index, b, k=1)
        if not ta or not tb:
            continue
        out.append(Question(
            f"enumerative:{a}+{b}", "enumerative",
            f"which of my projects use {ta[0]} or {tb[0]}", no_cwd=True))

    # Deictic: a pronoun with no referent must abstain, not guess.
    for i, text in enumerate((
        "why did it break",
        "what was the reason for that change",
        "how does this work",
    )):
        out.append(Question(f"deictic:{i}", "deictic", text, no_cwd=True))

    # No evidence: real words, absent from every scope's vocabulary.
    for i, text in enumerate((
        "how does the quantum flux capacitor calibrate",
        "what does the zzyzx subsystem do",
    )):
        out.append(Question(f"no_evidence:{i}", "no_evidence", text, no_cwd=True))

    # Relational: answered from the registry ahead of routing, so it exercises
    # a path the router never sees.
    for i, text in enumerate((
        "which of my projects uses another",
        "which of my projects depends on one of my other projects",
    )):
        out.append(Question(f"relational:{i}", "relational", text, no_cwd=True))

    return out
