"""Stage 1: decide which scope(s) a question belongs to.

Deterministic. No model call anywhere in this module. The output is a small
ranked scope set or an explicit abstention -- never a confident guess.

    scope_idf(t) = log(1 + S / scope_df(t))    discriminative across scopes
    f(t, s)      = node_df(t, s) / size(s)     size-normalized prominence
    contrib      = scope_idf(t) * (1 + log1p(f * 1000))
    score(s)     = sum(contrib) / |Q| / size(s)**SIZE_PRIOR

Both normalizations are load-bearing, not cosmetic. Real corpora are wildly
uneven -- one scope 68x larger than the rest is normal -- and any raw-count
score elects the biggest one regardless of the question.
"""
from __future__ import annotations

import math
from pathlib import Path

import re

from .text import unique_tokens
from .text import tokens as vtokens
from .types import RouteResult

ALIAS_BOOST = 6.0      # the question names the project outright
CWD_BOOST = 4.0        # cwd is inside the project tree
RECENCY_BOOST = 0.15   # tiebreak only; must never outrank real evidence
GROUP_PENALTY = 0.5    # multiplicative on the evidence base; see below
WIDEN_RATIO = 0.85     # keep scopes scoring >= WIDEN_RATIO * top
MAX_SCOPES = 3
MIN_MATCHED = 4           # legacy count gate; see EVIDENCE_FLOOR
EVIDENCE_FLOOR = 7.6      # summed token evidence required to route at all
CONCENTRATED_SCOPES = 2   # a token held by at most this many scopes is "concentrated"
CONCENTRATED_EVIDENCE = 2.6

# The concentrated tier exists because every other gate asks "is there enough
# evidence for ONE scope", and a question about something two projects share
# splits its evidence between them by construction. Measured across five
# corpora, such questions returned both owners 0-17% of the time -- and on real
# repositories they did not merely rank badly, they ABSTAINED, so widening never
# got a chance to help.
#
# A short question like "how is curl handled?" carries two tokens. Its summed
# evidence is low simply because there are two of them; its matched count is
# below the count gate for the same reason; and DECISIVE_EVIDENCE is calibrated
# for tokens exclusive to a single scope, which a shared term can never be. It
# fell through every gate.
#
# What identifies it is CONCENTRATION: a token held by very few scopes and
# prominent inside them. Measured on questions the evidence gate actually
# decides -- deictic ones are refused earlier by grammar and never reach it --
# the bands separate cleanly:
#
#   contended questions      3.12 and above
#   nonsense and vague       2.05 and below
#
# The threshold sits in that gap. The tier also forces the holding scopes into
# the result, because returning one owner of a shared term is the failure being
# fixed, not a partial success.
# DECISIVE_EVIDENCE was removed. It admitted a question carrying one
# exceptionally discriminative token, and the concentrated tier below covers
# that case and more -- a token held by very few scopes is what "decisive"
# was reaching for, expressed in a way that also identifies WHICH scopes.
# Disabling it changed no metric on any of six beds or on the held-out
# hand-authored set, so it was deleted rather than kept as a second name for
# the same idea.

# EVIDENCE_FLOOR replaced MIN_MATCHED as the primary gate because a matched-token
# COUNT does not separate routable questions from unroutable ones. Measured
# against auto-labelled questions on a real corpus:
#
#   matched count    NOT separable, inverted: routable median 2, unroutable
#                    median 2 and maximum 4
#   max evidence     separable, 4.31 vs 3.75
#   TOTAL evidence   separable, widest margin, 8.59 vs 7.12
#   mean evidence    NOT separable, 3.07 vs 3.45
#
# The count gate worked on the corpus it was fitted to, and did so by accident.
# The floor here is the default for an uncalibrated install; `loci calibrate`
# fits it to the corpus actually present, which is the point -- the right value
# depends on how much vocabulary a user's projects share, and no shipped
# constant can know that.

# Fitted against two corpora of the same ten projects -- one indexed with code
# symbols, one prose-only with no graphs at all -- so these are not tuned to a
# single corpus shape. What the sweep showed:
#
#   SIZE_PRIOR   0.15 optimal in BOTH. 0.05 costs cwd routing (100% -> 94%/89%);
#                0.30 costs top-1 (85.7% -> 71.4%); 0.45 is catastrophic.
#   WIDEN_RATIO  0.85 strictly dominates 0.70 in both: identical top-1 and
#                hit@widen, consistently smaller result sets.
#   MIN_MATCHED  4 over 3 refuses one more plausible-but-absent question and
#                converts one confidently-wrong answer into an abstention, with
#                no loss of top-1. Both moves are toward abstention, which is
#                the direction this design prefers to be wrong in.
#   DECISIVE_EVIDENCE  nearly inert -- it changes 1 item in 18 and 0 in 80
#                deictic questions, and is flat from 3.67 to 5.90 on both
#                corpora. It is kept because the one item it changes is exactly
#                its purpose: a question matching two tokens where one of them
#                is exclusive to a scope and prominent inside it. Raising
#                MIN_MATCHED makes that escape hatch matter more, not less.
#
# A percentile-calibrated form was tried (DECISIVE = p99 of the corpus's own
# evidence distribution, which is 4.91 warm and 5.43 cold) and made no
# difference to any metric, because the threshold is not binding. Absolute is
# simpler and measured equivalent.

# A question that points at its subject instead of naming it ("this project",
# "the app", "here") cannot be routed by vocabulary, because the words that
# would identify the subject are exactly the words it declines to say. Measured
# across 80 generated deictic questions over 10 scopes: 100% correct WITH a
# working directory, and 6% without -- while abstaining only 37% of the time, so
# the other 56% were confident and wrong.
#
# No lexical statistic rescues this. Per-token evidence for "start" (2.51) and
# "services" (2.56) is indistinguishable from real evidence like "session"
# (2.53); the individual words are not generic, the question is. Deixis is a
# grammatical property, so it is detected grammatically -- a closed class of
# markers, not a tuned threshold.
# Bare pronouns count. "How does *it* start up?" and "What does *this* depend
# on?" point just as hard as "this project", and an earlier noun-anchored
# pattern missed both. False positives are cheap: deixis only forces abstention
# when neither cwd nor an alias has already identified the subject, so a
# question that says "it" *and* names a project is unaffected.
DEIXIS = re.compile(
    r"\b(?:this|these|it|its|here)\b"
    r"|\bthe\s+(?:project|repo|repository|codebase|code\s*base|app|application|"
    r"service|package|module|library|tool|system)\b",
    re.IGNORECASE,
)


def is_deictic(question: str) -> bool:
    """True when the question points at its subject instead of naming it."""
    return bool(DEIXIS.search(question))
SIZE_PRIOR = 0.15

# Why an ABSOLUTE matched-token floor rather than a margin test: measured on
# real corpora, margin does not separate routable from unroutable questions at
# all -- "How do I fix this bug?" produces margin 0.94 while a correctly-routed
# question produces 0.18. What separates them is how many query tokens landed:
# unroutable questions match 1-2, real ones match 6-8.
#
# Why a size prior: scope-IDF is confounded by vocabulary size. Ordinary English
# words appear in exactly one scope when that scope's index is 67x larger than
# the others, and get scored as maximally discriminative for it.
#
# Why the count gate needs a second tier: a raw count cannot tell two weak
# tokens from one decisive one. "which projects use wrangler and a D1 database"
# matches only 2 tokens against the right scope, but `wrangler` lives in exactly
# one scope out of ten and in 10% of its nodes -- that is not weak evidence, it
# is the strongest kind. DECISIVE_EVIDENCE is measured on the contribution
# normalized by the maximum possible IDF, which makes it scale-free: a token
# exclusive to one scope clears it at ~3% node prominence, a token shared by two
# scopes needs ~15%. Exclusivity and prominence trade off, as they should.

# GROUP_PENALTY is HAND-SET and has never been fitted. It is recorded here in
# the same terms as ALIAS_BOOST and CWD_BOOST, which is to say: the property
# that matters is the ORDERING, not the magnitude.
#
# It is multiplicative, and applied to the evidence base BEFORE the boosts.
# Measured on a real corpus, evidence scores occupy 0.1-1.5 while CWD_BOOST is
# 4.0:
#
#   why was the session cookie dropped on localhost?
#       urthreads 1.4279   odysseus 0.8369   Delroy 0.6980
#   how is this deployed?          (cwd = Delroy)
#       Delroy 4.8987 (cwd)        urthreads 0.1333
#
# An ADDITIVE penalty large enough to reorder that first row would drive most
# scopes negative and collapse `soft` into `hard`. A multiplicative one at 0.5
# moves urthreads to 0.7140 -- still just above Delroy -- and cannot remove the
# boost, because the boost is added afterwards.
#
# What that buys, exactly: a demoted scope holding cwd scores at least
# CWD_BOOST, at ANY penalty including 0, because the penalty scales only the
# evidence base. So no penalty can push it below a scope whose own base is under
# 4.0 -- which, measured, is every unboosted scope on a real corpus (0.1-1.5).
#
# It is NOT an unconditional guarantee, and an earlier version of this comment
# claimed it was. A scope with an unusually large base can still overtake a
# demoted cwd scope. Measured on a small, vocabulary-dense fixture:
#
#   penalty 1.0   Alpha 4.5888 (cwd)   Beta 4.2083
#   penalty 0.1   Alpha 4.0589 (cwd)   Beta 4.2083   <- inverted
#
# Sweep it across corpus shapes with the Phase 3 harness. What must hold is the
# floor: a demoted scope carrying cwd or an alias never scores below that
# scope's boost, so the signal survives every value of GROUP_PENALTY.


def _alias_hit(q_tokens: list[str], alias: str) -> bool:
    a = vtokens(alias)
    if not a:
        return False
    n = len(a)
    return any(q_tokens[i:i + n] == a for i in range(len(q_tokens) - n + 1))


def _recency_rank(scopes: dict) -> dict[str, float]:
    stamped = [(sid, m.get("updated_at") or "") for sid, m in scopes.items()]
    order = [sid for sid, _ in sorted(stamped, key=lambda x: x[1])]
    n = len(order)
    if n <= 1:
        return {sid: 1.0 for sid, _ in stamped}
    return {sid: i / (n - 1) for i, sid in enumerate(order)}


def _calibrated_floor() -> float:
    """The fitted floor for this corpus, or the shipped default."""
    try:
        from .calibrate import load
        cal = load()
    except Exception:
        cal = None
    return cal.evidence_floor if cal else EVIDENCE_FLOOR


def route(question: str, index: dict, *, cwd: str | Path | None = None,
          widen_ratio: float | None = None, max_scopes: int | None = None,
          min_matched: int | None = None,
          evidence_floor: float | None = None,
          concentrated_evidence: float | None = None,
          size_prior: float | None = None,
          eligible: set[str] | None = None,
          demoted: set[str] | None = None,
          group_penalty: float | None = None,
          strict_group: bool = False,
          group: str | None = None,
          mode: str | None = None) -> RouteResult:
    """Route a question to scope(s).

    `eligible` and `demoted` come from a group's mode. They filter SELECTION;
    they never change scoring, and `S` below stays the full corpus count --
    shrinking it would inflate scope-IDF for survivors and move every
    calibrated threshold silently.

    `group` and `mode` are carried through for reporting only; the router does
    not resolve policy.

    Every threshold defaults to None and is resolved from the module here, not
    in the signature. A default argument is evaluated once at import time, so
    `router.MIN_MATCHED = 6` had no effect on a caller using the default -- which
    silently turned several threshold sweeps into no-ops and made constants that
    matter look inert.
    """
    widen_ratio = WIDEN_RATIO if widen_ratio is None else widen_ratio
    max_scopes = MAX_SCOPES if max_scopes is None else max_scopes
    min_matched = MIN_MATCHED if min_matched is None else min_matched
    concentrated_evidence = (CONCENTRATED_EVIDENCE if concentrated_evidence is None
                             else concentrated_evidence)
    size_prior = SIZE_PRIOR if size_prior is None else size_prior
    group_penalty = GROUP_PENALTY if group_penalty is None else group_penalty
    scopes = index["scopes"]
    postings = index["postings"]
    S = len(scopes)

    q_all = vtokens(question)
    q = unique_tokens(question)
    recency = _recency_rank(scopes)
    cwd_path = Path(cwd).expanduser().resolve() if cwd else None

    scores: dict[str, float] = {}
    detail: dict[str, dict] = {}

    # Tokens held by very few scopes, and prominent in them.
    idf_max_global = math.log(1 + S) or 1.0
    concentrated_owners: set[str] = set()
    for t in q:
        post = postings.get(t)
        if not post or len(post) > CONCENTRATED_SCOPES:
            continue
        idf = math.log(1 + S / len(post))
        for sid, df in post.items():
            n = max(1, scopes[sid].get("node_count", 1))
            ev = idf * (1 + math.log1p(df / n * 1000)) / idf_max_global
            if ev >= concentrated_evidence:
                concentrated_owners.add(sid)

    for sid, meta in scopes.items():
        n_nodes = max(1, meta.get("node_count", 1))
        total = 0.0
        matched: list[tuple[str, float]] = []

        for t in q:
            post = postings.get(t)
            if not post or sid not in post:
                continue
            idf = math.log(1 + S / max(1, len(post)))
            contrib = idf * (1 + math.log1p(post[sid] / n_nodes * 1000))
            total += contrib
            matched.append((t, round(contrib, 3)))

        idf_max = math.log(1 + S) or 1.0
        best_evidence = max((c for _, c in matched), default=0.0) / idf_max
        total_evidence = sum(c for _, c in matched) / idf_max

        base = total / max(1, len(q))
        if size_prior:
            base /= max(1, meta.get("token_count", 1)) ** size_prior
        # Before the boosts, deliberately: a multiplicative penalty here cannot
        # invert a cwd or alias signal that is added below.
        if demoted and sid in demoted:
            base *= group_penalty

        signals: dict[str, str] = {}
        alias = next((a for a in meta.get("aliases", []) if _alias_hit(q_all, a)), None)
        if alias:
            base += ALIAS_BOOST
            signals["alias"] = alias
        if cwd_path is not None:
            root = Path(meta["root"])
            if cwd_path == root or root in cwd_path.parents:
                base += CWD_BOOST
                signals["cwd"] = str(root)
        base += RECENCY_BOOST * recency.get(sid, 0.0)

        scores[sid] = base
        matched.sort(key=lambda x: -x[1])
        detail[sid] = {
            "name": meta["name"], "score": round(base, 4),
            "matched": len(matched),
            "evidence": round(best_evidence, 3),
            "evidence_total": round(total_evidence, 3),
            "coverage": round(len(matched) / max(1, len(q)), 3),
            "top_tokens": matched[:6], "signals": signals,
        }

    ranked_all = sorted(scores, key=lambda s: -scores[s])
    top_all = ranked_all[0] if ranked_all else None
    ranked = ([s for s in ranked_all if s in eligible] if eligible is not None
              else ranked_all)

    top = ranked[0] if ranked else None
    top_score = scores.get(top, 0.0) if top else 0.0
    top_matched = detail[top]["matched"] if top else 0
    top_total = detail[top]["evidence_total"] if top else 0.0
    floor = _calibrated_floor() if evidence_floor is None else evidence_floor

    # An alias or cwd signal is direct evidence, never overruled by abstention.
    # `forced` means cwd or an explicit alias resolved the subject; deixis is
    # only a problem when nothing else identifies what "this" refers to.
    forced = bool(top and detail[top]["signals"])
    deictic = is_deictic(question)
    # A concentrated token held only by scopes outside the group must not force
    # routing. With `eligible` unset this intersection is a no-op, because
    # concentrated_owners is always a subset of the corpus.
    concentrated_here = concentrated_owners & set(ranked)
    # Three independent ways to have enough evidence, any one of which routes:
    # one exceptionally discriminative token, enough summed evidence, or enough
    # matched tokens. They fail on different question shapes -- a short question
    # about a rare symbol has high evidence and a low count, a long question
    # about a familiar subsystem has the reverse -- so requiring all three
    # abstains on both.
    enough = (top_total >= floor or top_matched >= min_matched
              or bool(concentrated_here))

    # The same three gates applied to the corpus-wide winner. Without them
    # `out_of_group` fires on a scope that matched nothing: when a question hits
    # no vocabulary at all, every scope scores ~0 and the winner is whoever took
    # the 0.15 recency tiebreak. Reporting "the answer was elsewhere" about a
    # scope scoring 0.15 made `out_of_group` swallow both other reasons for
    # every unroutable question under `hard`.
    top_all_total = detail[top_all]["evidence_total"] if top_all else 0.0
    top_all_matched = detail[top_all]["matched"] if top_all else 0
    answer_elsewhere = (top_all_total >= floor or top_all_matched >= min_matched
                        or top_all in concentrated_owners)

    reason: str | None = None
    if strict_group and eligible is not None and top_all is not None \
            and top_all not in eligible and answer_elsewhere:
        # The best answer is outside the group. Returning the in-group runner-up
        # would be a confident answer from the wrong project.
        abstain, reason = True, "out_of_group"
    elif not forced and deictic:
        abstain, reason = True, "deictic"
    elif not forced and not enough:
        abstain, reason = True, "no_evidence"
    else:
        abstain = False

    if abstain:
        selected = ranked            # caller should ask, not pick
    else:
        cutoff = top_score * widen_ratio
        selected = [s for s in ranked if scores[s] >= cutoff][:max_scopes]
        # Every scope holding a concentrated token belongs in the answer, in
        # rank order. Returning one owner of a shared term is the bug.
        if concentrated_here:
            merged = list(dict.fromkeys(
                selected + [s for s in ranked if s in concentrated_here]))
            selected = merged[:max_scopes]

    if top:
        detail[top]["deictic"] = deictic
    return RouteResult(
        question=question, query_tokens=q, ranked=ranked, selected=selected,
        abstain=abstain, top_score=top_score, top_matched=top_matched,
        detail=detail, group=group, mode=mode, abstain_reason=reason,
    )
