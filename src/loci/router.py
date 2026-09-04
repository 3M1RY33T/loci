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

# A token held by more than this share of the corpus is not a claim on the
# question, and a scope holding nothing else is not a candidate for it.
#
# It governs the shortlist an ABSTENTION hands back, never a routing decision.
# Before it existed the candidate line printed `ranked` -- every eligible scope,
# in score order, which is the entire registry on any abstention. Fourteen names
# with nothing to prefer between them is not a shortlist; it is the registry.
#
# Measured over 10 gold-bearing abstentions (the hand-authored eval set, plus
# ablations of the questions that route -- drop the winner's top token until it
# stops routing) and 22 questions that SHOULD abstain. Reproduce the table with
# `python evals/clarify.py --sweep`:
#
#   rule                       gold recall   |cand| gold   |cand| should-abstain
#   ranked, i.e. before            100.0%          14.0          14.0
#   holds any matched token         90.0%           7.8           6.0
#   held by <= S/2 scopes           90.0%           6.3           4.4
#   held by <= S/3 scopes           80.0%           4.0           2.5
#   held by <= 2 scopes             50.0%           2.4           1.0
#
# Half the corpus is the widest cut that costs no recall over "holds any token
# at all", and every tighter cut drops a real answer. It is a SHARE rather than
# a count because a count is a number that happened to suit fourteen scopes: at
# S=4 a threshold of 4 filters nothing, and the same table on a 200-scope corpus
# would be measuring something else entirely.
#
# Ten abstentions is not a fitted constant and is not presented as one -- the
# corpus has three questions that abstain outright, and the rest are weakened
# phrasings of ones that do not. What the table supports is the ORDERING, which
# is monotone and wide: recall falls at every tightening and the shortlist
# shrinks at every one, and 0.5 is the last point before recall moves at all.
#
# The one abstention no setting recovers is worth naming: "Can I drop
# photographs but keep vector graphics?" holds no token its gold scope indexes,
# so it is unreachable by any shortlist. That is a coverage problem, and
# `_advice` sends it to `doctor` rather than pretending a scope could be chosen.
CANDIDATE_SHARE = 0.5

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
#
# CWD_BOOST goes to the DEEPEST containing scope and to no other. It used to go
# to every scope whose root contains cwd, which on a flat corpus is the same
# thing and on a split monorepo is not: parent and child both took the identical
# 4.0, so cwd stopped discriminating between them entirely and the parent won on
# vocabulary, which it always does -- it is the larger index. Measured from
# inside `Delroy/glasses`, a deictic question scored Delroy 5.17 against
# Delroy/glasses 4.91 and routed to the parent.
#
# `scopes.scope_for_cwd` already resolved cwd to one scope by longest root, and
# says so; this makes the router agree with it. Measured twice, on two different
# indexes, and the two numbers are not interchangeable:
#
#   - a RECONSTRUCTED split-stage index, built to isolate this change from the
#     rest of the branch: deictic + cwd 88.9% -> 100.0% (36/36);
#   - then `loci eval` on the real split-stage index of the development corpus
#     (19 scopes registered, 18 indexed): deictic + cwd 100.0% (72/72), against
#     the 87.5% (63/72) evals/RESULTS.md records for that same index before this
#     change.
#
# Byte-identical on the pre-split stage, where no scope contains another and the
# deepest containing scope is the only containing scope. No constant moved --
# the change is what the signal MEANS, not how big it is.
#
# It does not fix the OTHER defect the split surfaced. A sub-project's bare
# directory name becomes a corpus-wide alias worth 6.0, which is why marker
# splitting is now off by default (scopes.WORKSPACE_MARKERS).
DEIXIS = re.compile(
    r"\b(?:this|these|it|its|here)\b"
    r"|\bthe\s+(?:project|repo|repository|codebase|code\s*base|app|application|"
    r"service|package|module|library|tool|system)\b",
    re.IGNORECASE,
)


def is_deictic(question: str) -> bool:
    """True when the question points at its subject instead of naming it."""
    return bool(DEIXIS.search(question))


# A question that asks for a SET is the exact inverse of the case every other
# gate is built for. Each gate asks "is there enough evidence for ONE scope",
# and enumeration splits its evidence across every owner by construction -- so
# the more projects genuinely share a term, the less likely all of them come
# back. Measured before this fix, `which of my projects use Cloudflare workers
# or D1?` returned one owner of two: 2.098 against 1.0867, where WIDEN_RATIO
# needed 1.783.
#
# The README's own quickstart returned both owners only by luck. `wrangler` sits
# in exactly two scopes and trips the concentrated tier at CONCENTRATED_SCOPES;
# `cloudflare` sits in two as well but its owners score 2.527 and 2.314 against
# CONCENTRATED_EVIDENCE of 2.6, so the tier declined them both by a hair. The
# mechanism inverts on the one question shape that wants a set.
#
# Enumeration is grammatical, like deixis, so it is detected grammatically -- a
# closed class of markers, not a tuned threshold.
ENUMERATIVE = re.compile(
    r"\bwhich\s+(?:of\s+)?(?:my|the|our|your)?\s*"
    r"(?:projects?|repos?|repositories|codebases?|code\s*bases?)\b"
    r"|\bwhat\s+(?:projects?|repos?|repositories|codebases?)\b"
    r"|\b(?:do\s+)?any\s+of\s+(?:my|the|our|your)\b"
    r"|\b(?:where|anywhere)\s+else\b"
    r"|\b(?:across|in)\s+(?:all\s+)?(?:my|the|our|your)\s+"
    r"(?:projects?|repos?|repositories|codebases?)\b"
    r"|\ball\s+(?:my|the|our|your)\s+(?:projects?|repos?|repositories)\b"
    r"|\bhave\s+i\s+ever\b",
    re.IGNORECASE,
)

# The frame nouns are scaffolding for the enumeration, not vocabulary that
# identifies anything -- the same insight as deixis, one level down. Leaving
# them in is not neutral: `projects` is held by exactly two scopes in the
# development corpus, at 88 and 7 occurrences, so it scored as discriminative
# evidence FOR those two and dragged both into the answer to every "which of my
# projects" question ever asked. Stripping them alone lifted negative-family
# abstention from 66.7% to 88.9%.
ENUM_FRAME = frozenset({
    "projects", "project", "repos", "repo", "repositories", "repository",
    "codebases", "codebase", "else", "ever",
})

# What set mode does with the floor. It is a RATIO of the routing floor rather
# than an absolute, so it inherits whatever `loci calibrate` fitted to the
# corpus actually present -- the same reason EVIDENCE_FLOOR is calibrated at
# all. A member of a set legitimately carries less evidence than a lone answer,
# because the question's evidence is split across the owners.
#
# Swept against the hand-authored eval on the development corpus. Confusable
# gold-coverage is FLAT at 83.3% from 0.6 to 0.8 and falls to 66.7% at 0.9;
# precision peaks at 0.8 (77.8%) and decays below 0.7 (72.2% at 0.7, 65.1% at
# 0.5) as the sets grow. 0.8 is the top of the flat region and the precision
# maximum, so it is the value that widens least to get the coverage. The
# acceptable band is 0.6-0.8, reported per the rule that a flat region beats a
# sharp optimum.
SET_FLOOR_RATIO = 0.8
# Enumeration over a corpus of ten to twenty projects should be allowed to
# return more than MAX_SCOPES, which is sized for "which ONE project". This is a
# runaway guard, not a fitted value: nothing in the eval reaches it.
MAX_SET_SCOPES = 8


def is_enumerative(question: str) -> bool:
    """True when the question asks which projects, not which project."""
    return bool(ENUMERATIVE.search(question))
SIZE_PRIOR = 0.15

# How much a token counts when it is NOT a scope's best evidence.
#
# SHIPPED INERT AT 1.0, which is a plain sum and exactly what the router did
# before this constant existed. It is here because the failure it addresses is
# real and measured, and the evidence to SETTLE it is not obtainable from the
# current test bed. Read the whole note before moving it.
#
# The failure. The score is a sum over matched tokens, and a sum rewards
# BREADTH. Measured on a 14-scope corpus where one scope holds 5,995 distinct
# tokens against urthreads' 427: "Why would a browser silently discard a login
# cookie that curl accepts?" routes to the big scope, which matches all seven
# query tokens on ordinary English -- `browser`, `accepts`, `discard`,
# `silently` -- while urthreads matches three and owns the two that carry the
# question, `cookie` at 5.47 and `login` at 4.53. Containing a word is not the
# same as being about it. This is the same mechanism the size prior was
# introduced for ("ordinary English words appear in exactly one scope when that
# scope's index is 67x larger"), at a skew the size prior no longer corrects:
# sweeping SIZE_PRIOR from 0.0 to 0.5 on this corpus never beats 0.15, so the
# knob is not the answer.
#
# What was tried. Five scoring families, swept against the hand-authored eval:
# share-weighting (contrib x its share of the token, exponent 0.5/1/2),
# owner-thresholding (count a token only for scopes within tau of its best
# owner, tau 0.4-0.85), power means (exponent 1.5-4), and this max-plus-
# discounted-rest blend. NONE beats the baseline's 28.6% behavior top-1. The
# power means and share-weightings are strictly worse. Reweighting lexical
# evidence cannot fix a case where the lexical evidence genuinely favours the
# wrong scope -- on `beh-ody-windows-clicks` the big scope matches `clicks` and
# `responding` and the gold scope matches neither. That needs a different
# signal, not a different weighting.
#
# What this blend DOES buy, at 0.55-0.90 on the real corpus: negative-family
# abstention 83.3% -> 100%, with behavior, confusable, cross and both taxonomy
# figures byte-identical to the baseline. The item it converts is
# `neg-plausible-absent` -- a technically specific question about Kubernetes,
# which no scope does -- from a confident wrong answer into an abstention. That
# is the "plausibility alone must not produce an answer" test, and abstention is
# the direction this design prefers to be wrong in.
#
# Why it is not enabled. One item, on one corpus, in the set it was measured
# against -- rules 1 and 2. The synthetic bed cannot arbitrate: swept across all
# thirteen standard shapes at 0.5/0.7/0.9/1.0, every metric reads 100% at every
# value, because `CorpusSpec.size_skew` scales file VOLUME while every scope
# still draws its filler from one fixed pool of shared English. The shape that
# breaks the real corpus -- one scope whose DISTINCT vocabulary approaches the
# whole corpus's -- is not generatable today. Give the generator a vocabulary-
# breadth dimension, reproduce the failure, and then this constant can be
# settled with evidence instead of with one item.
CORROBORATION_WEIGHT = 1.0

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
# evidence base. Recency is added to both sides afterwards, so the competitor a
# penalty cannot reach is one whose base stays under
#
#   CWD_BOOST + RECENCY_BOOST * (recency(demoted) - recency(other))
#
# i.e. under 3.85 in the worst case, where the demoted scope is the stalest in
# the corpus and its rival the freshest. Measured, unboosted bases occupy
# 0.1-1.5, so every real competitor is far below even that worst case.
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


# The routability predicate, asked twice per question: once of the in-group
# winner (`forced or enough`, which decides whether to answer) and once of the
# corpus-wide winner (`answer_elsewhere`, which decides whether a hard group is
# refusing something real). It was written out twice, and the copies drifted --
# a fix round dropped the `signals` disjunct from the second one, which turned a
# hard-group abstention into a confident answer from the wrong project.
#
# TWO helpers, not one, because the caller needs the halves apart: `forced`
# alone distinguishes `deictic` from `no_evidence`, and a single combined
# predicate would collapse those two abstention reasons into one.
_NO_SCOPE = {"evidence_total": 0.0, "matched": 0, "signals": {}}


def _signalled(d: dict) -> bool:
    """cwd or an explicit alias resolved the subject.

    Direct evidence, and it contributes nothing to `evidence_total` or
    `matched`: a scope winning purely on ALIAS_BOOST or CWD_BOOST reads as zero
    evidence to `_has_evidence`, which is why this is a separate disjunct at
    both call sites rather than something the counts can stand in for.
    """
    return bool(d["signals"])


def _has_evidence(d: dict, floor: float, min_matched: int,
                  concentrated: bool) -> bool:
    """Three independent ways to have enough evidence; any one routes.

    They fail on different question shapes -- a short question about a rare
    symbol has high evidence and a low count, a long question about a familiar
    subsystem has the reverse -- so requiring all three abstains on both.

    `concentrated` is the CALLER's concentration verdict, and the two callers
    ask deliberately different questions of it. The in-group gate asks whether
    ANY eligible scope holds a concentrated token, because a question about
    something two projects share splits its evidence between them and the tier
    then forces both owners into the result -- the top scope need not be one of
    them. `answer_elsewhere` asks whether THAT ONE scope holds one, because it
    is judging a single scope's answer and `bool(concentrated_owners)` there
    would fire on a token no scope in the comparison holds. Passing `sid in
    concentrated` at both sites reads tidier and is wrong: measured, it flips a
    contended question whose owner is the runner-up from a two-scope answer into
    a `no_evidence` abstention that drops the owner entirely.
    """
    return (d["evidence_total"] >= floor or d["matched"] >= min_matched
            or concentrated)


def route(question: str, index: dict, *, cwd: str | Path | None = None,
          widen_ratio: float | None = None, max_scopes: int | None = None,
          min_matched: int | None = None,
          evidence_floor: float | None = None,
          concentrated_evidence: float | None = None,
          size_prior: float | None = None,
          corroboration_weight: float | None = None,
          eligible: set[str] | None = None,
          demoted: set[str] | None = None,
          group_penalty: float | None = None,
          strict_group: bool = False,
          group: str | None = None,
          set_floor_ratio: float | None = None,
          max_set_scopes: int | None = None,
          mode: str | None = None) -> RouteResult:
    """Route a question to scope(s).

    `eligible` and `demoted` both come from a group's mode and do DIFFERENT
    things. `eligible` filters selection and never touches a score. `demoted`
    changes scoring: `base *= group_penalty` below, applied before the boosts so
    that it cannot invert a cwd or alias signal.

    Neither changes the scoring MODEL. `S` below stays the full corpus count --
    shrinking it would inflate scope-IDF for survivors and move every calibrated
    threshold silently.

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
    corroboration_weight = (CORROBORATION_WEIGHT if corroboration_weight is None
                            else corroboration_weight)
    group_penalty = GROUP_PENALTY if group_penalty is None else group_penalty
    set_floor_ratio = (SET_FLOOR_RATIO if set_floor_ratio is None
                       else set_floor_ratio)
    max_set_scopes = MAX_SET_SCOPES if max_set_scopes is None else max_set_scopes
    scopes = index["scopes"]
    postings = index["postings"]
    S = len(scopes)

    q_all = vtokens(question)
    q = unique_tokens(question)
    enumerative = is_enumerative(question)
    if enumerative:
        q = [t for t in q if t not in ENUM_FRAME]
    recency = _recency_rank(scopes)
    cwd_path = Path(cwd).expanduser().resolve() if cwd else None

    # The DEEPEST containing scope, and only it, takes CWD_BOOST. Resolved once
    # here rather than per scope below, on the same rule `scopes.scope_for_cwd`
    # already uses -- longest root wins -- so the router and the registry cannot
    # disagree about which scope you are standing in.
    cwd_owner: tuple[str, Path] | None = None
    if cwd_path is not None:
        for sid, meta in scopes.items():
            root = Path(meta["root"])
            if cwd_path == root or root in cwd_path.parents:
                if cwd_owner is None or len(str(root)) > len(str(cwd_owner[1])):
                    cwd_owner = (sid, root)

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

    # At most this many scopes may hold a token before holding it stops meaning
    # anything. Floored at 1 so a two- or three-scope corpus still asks for a
    # token that separates, rather than accepting one every scope shares.
    claim_max = max(1, int(S * CANDIDATE_SHARE))

    for sid, meta in scopes.items():
        n_nodes = max(1, meta.get("node_count", 1))
        total = 0.0
        matched: list[tuple[str, float]] = []
        claim_toks: set[str] = set()

        for t in q:
            post = postings.get(t)
            if not post or sid not in post:
                continue
            idf = math.log(1 + S / max(1, len(post)))
            contrib = idf * (1 + math.log1p(post[sid] / n_nodes * 1000))
            total += contrib
            matched.append((t, round(contrib, 3)))
            if len(post) <= claim_max:
                claim_toks.add(t)

        idf_max = math.log(1 + S) or 1.0
        best_evidence = max((c for _, c in matched), default=0.0) / idf_max
        total_evidence = sum(c for _, c in matched) / idf_max

        # Only the RANKING base is discounted. `evidence_total` above stays a
        # plain sum, so EVIDENCE_FLOOR and everything `loci calibrate` fitted
        # keep the meaning they were measured with -- this changes which scope
        # wins, never whether the question is routable at all.
        best = max((c for _, c in matched), default=0.0)
        ranked_total = best + corroboration_weight * (total - best)
        base = ranked_total / max(1, len(q))
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
        if cwd_owner is not None and sid == cwd_owner[0]:
            base += CWD_BOOST
            signals["cwd"] = str(cwd_owner[1])
        base += RECENCY_BOOST * recency.get(sid, 0.0)

        scores[sid] = base
        matched.sort(key=lambda x: -x[1])
        detail[sid] = {
            "name": meta["name"], "score": round(base, 4),
            "matched": len(matched),
            # Ordered by contribution, not by where they fell in the question:
            # these are printed to a reader who has to choose a scope, and the
            # first one should be the term that most separates this scope.
            "claims": [t for t, _ in matched if t in claim_toks],
            "evidence": round(best_evidence, 3),
            "evidence_total": round(total_evidence, 3),
            "coverage": round(len(matched) / max(1, len(q)), 3),
            "top_tokens": matched[:6], "signals": signals,
        }

    ranked_all = sorted(scores, key=lambda s: -scores[s])
    top_all = ranked_all[0] if ranked_all else None
    ranked = ([s for s in ranked_all if s in eligible] if eligible is not None
              else ranked_all)

    # In `ranked`'s order, which is the SCORE's. Re-sorting the shortlist by
    # evidence reads as the more principled choice and measures worse: on the
    # same 10 abstentions, the gold scope came first 70% of the time under the
    # score and 30% under evidence_total. The score carries the cwd and alias
    # boosts and the size prior; raw evidence favours whichever scope is
    # biggest, which is the failure SIZE_PRIOR exists to correct.
    candidates = [s for s in ranked if detail[s]["claims"]]

    top = ranked[0] if ranked else None
    top_score = scores.get(top, 0.0) if top else 0.0
    top_d = detail[top] if top else _NO_SCOPE
    top_all_d = detail[top_all] if top_all else _NO_SCOPE
    top_matched = top_d["matched"]
    floor = _calibrated_floor() if evidence_floor is None else evidence_floor

    # An alias or cwd signal is direct evidence, never overruled by abstention.
    # `forced` means cwd or an explicit alias resolved the subject; deixis is
    # only a problem when nothing else identifies what "this" refers to.
    forced = _signalled(top_d)
    # Enumeration outranks deixis. "where else does this pattern appear?" points
    # at its subject AND asks across the corpus; the deixis rule exists because
    # a pointing question gives no way to pick ONE scope, and an enumerative one
    # is not picking one. The evidence floor still guards it, so a question that
    # points and says nothing else abstains on `no_evidence` as before.
    deictic = is_deictic(question) and not enumerative
    # A concentrated token held only by scopes outside the group must not force
    # routing. With `eligible` unset this intersection is a no-op, because
    # concentrated_owners is always a subset of the corpus.
    concentrated_here = concentrated_owners & set(ranked)
    enough = _has_evidence(top_d, floor, min_matched, bool(concentrated_here))

    # `forced or enough` asked of the corpus-wide winner instead of the in-group
    # one -- the same predicate, through the same two helpers, which is the
    # point: written out twice it drifted, and the copy here lost the `signals`
    # disjunct. That let the in-group runner-up's own evidence satisfy `enough`
    # above, turning a hard-group abstention into a confident answer from the
    # wrong project -- naming a project outside the group, or asking from inside
    # its tree, produced exactly the failure hard mode exists to prevent.
    #
    # Without a gate here at all, `out_of_group` fired on a scope that matched
    # nothing: when a question hits no vocabulary, every scope scores ~0 and the
    # winner is whoever took the 0.15 recency tiebreak, so `out_of_group`
    # swallowed both other reasons for every unroutable question under `hard`.
    answer_elsewhere = (_signalled(top_all_d)
                        or _has_evidence(top_all_d, floor, min_matched,
                                         top_all in concentrated_owners))

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
    elif enumerative:
        # Every scope that clears the floor ON ITS OWN, in rank order -- not
        # every scope within a ratio of the winner. The ratio cutoff is what
        # inverts here: it measures distance from the TOP scope, and the top
        # scope of an enumeration is merely the owner that happens to say the
        # term most often. `_has_evidence` is asked of each scope in turn, the
        # same predicate the top scope faces, against the discounted floor.
        set_floor = floor * set_floor_ratio
        selected = [s for s in ranked
                    if _signalled(detail[s])
                    or _has_evidence(detail[s], set_floor, min_matched,
                                     s in concentrated_owners)][:max_set_scopes]
        if not selected:
            # The corpus-wide gate passed on the top scope's own evidence, but
            # no scope clears the per-scope floor. Naming one owner of a set is
            # the failure this branch exists to prevent, so abstain instead.
            abstain, reason = True, "no_evidence"
            selected = ranked
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
        enumerative=enumerative, candidates=candidates,
    )
