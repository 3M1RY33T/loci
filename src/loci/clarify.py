"""The abstention, shaped as a question a host can render.

loci does not ask anything. It emits the question it WOULD ask, and the host
draws it with whatever it has -- Delroy's `ask_user_questions` card, Claude
Code's `AskUserQuestion`, a numbered list in a terminal. Putting the question in
the RESULT rather than in the protocol is what makes it portable: a client that
reaches loci as a stdio subprocess has no channel for a server-initiated
elicitation, and Delroy's chat path is exactly that.

The field names are `AskUserQuestion`'s. Delroy's `normalize_questions` reads
the same ones and accepts either `multiSelect` or `multi_select`, so one shape
serves both hosts with no translation layer in between -- and a translation
layer is the thing that would drift.

Nothing here calls a model. The question is templated, the option labels are
scope names, and the descriptions are the `claims` the router already computed,
so this module cannot disagree with the routing it explains.
"""
from __future__ import annotations

from .types import RouteResult

# Both hosts accept 2-4 options. Fewer than two is not a question; more than
# four is dropped by one host and rejected by the other.
MIN_OPTIONS = 2
MAX_OPTIONS = 4
# Claude Code caps the header at 12 characters and Delroy truncates it at
# QUESTION_HEADER_MAX_CHARS. One word that fits both.
HEADER = "Project"
# Per option, matching what the rendered candidate line already shows.
CLAIM_TERMS = 3

# One line per abstention cause, because the cause decides what the user is
# being asked FOR. A bare shortlist under `out_of_group` reads as "pick one of
# these" when the honest question is "your group says otherwise -- override?".
_WHY = {
    "deictic": "The question points at a project without naming it.",
    "no_evidence": "Not enough of the question exists in any one project.",
    "out_of_group": "The best match was outside the group in force.",
}
_WHY_DEFAULT = "The question was not specific enough to route."


def clarify(rt: RouteResult, *, limit: int = MAX_OPTIONS) -> dict | None:
    """The question to put to the user, or None when there is none to ask.

    None is the common case and is not a failure. A routed answer has no tie to
    break, and an abstention with nothing on its shortlist is a coverage report
    that `doctor` answers rather than a choice the user could make.
    """
    if not rt.abstain:
        return None

    options: list[dict[str, str]] = []
    for sid in rt.candidates[:limit]:
        detail = rt.detail.get(sid) or {}
        claims = list(detail.get("claims") or [])[:CLAIM_TERMS]
        options.append({
            "label": str(detail.get("name") or sid),
            # What put this scope on the list, which is the only basis the user
            # has for preferring one over another. Empty when the shortlist was
            # built by a caller that did not carry `claims`.
            "description": f"holds: {', '.join(claims)}" if claims else "",
        })

    if len(options) < MIN_OPTIONS:
        return None

    why = _WHY.get(rt.abstain_reason or "", _WHY_DEFAULT)
    return {
        "header": HEADER,
        "question": f"{why} Which project do you mean?",
        # Single-select: the router abstained because it could not pick ONE, and
        # the answer that unblocks it is one `--scope`. Both hosts still offer a
        # free-text field for "none of these", which is the escape hatch when
        # the right project is not on the list at all.
        "multiSelect": False,
        "options": options,
    }
