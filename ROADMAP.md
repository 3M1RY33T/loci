# Roadmap: from "works on one corpus" to "works on yours"

Everything in loci was fitted and measured against ten repositories belonging to
one person. The architecture came from mechanisms and should generalise; the
numbers came from one machine and demonstrably do not. This is the plan to close
that, in the order the evidence says to close it.

## Where things actually stand

| | count | status |
|---|---|---|
| Calibrated per corpus | 1 | `EVIDENCE_FLOOR` |
| Swept, but on one corpus | 5 | `WIDEN_RATIO` `SIZE_PRIOR` `MIN_MATCHED` `DECISIVE_EVIDENCE` `SEMANTIC_FLOOR` |
| Hand-set, no evidence at all | 8 | fusion weights, `SCORE_FLOOR`, `MIN_GROUNDED`, `LENGTH_SATURATION`, `ALIAS_BOOST`, `CWD_BOOST` |

The third row is worse than the second. A corpus-fitted constant is at least
fitted to *something*; those eight were chosen because they seemed reasonable
and no alternative was ever tested.

---

## Rules this plan runs under

Every one of these was learned by violating it during development. They are
listed first because the roadmap is worthless without them.

**1. Never fit a threshold to a best-case sample.** The first calibration drew
signature terms from each scope's rarest words, producing a "routable" band
starting at 8.59 when real questions score 6.5–8.8. The fitted floor cut real
questions in half and dropped cross-scope routing from 100% to 33%.

**2. Validate on held-out questions, always.** Calibration scored 100% on its
own families while degrading held-out performance. A benchmark that shares
questions with the thing it tunes measures nothing.

**3. Prefer a flat region to a sharp optimum.** A constant that peaks at one
value on one corpus and falls away steeply is fitted to noise. A constant that
is flat across a wide band on several corpora is a real setting. Report the
*width* of the acceptable range, not just the argmax.

**4. Ruling out the parameter is not ruling out the corpus.** A routing drop was
diagnosed as single-item noise, and a threshold re-sweep correctly cleared the
threshold — while the real cause was a glob bug silently dropping 28 files. When
a number moves, diff the inputs before tuning the knobs.

**5. A benchmark that penalises correct behaviour is worse than none.** One
generated template contained the pronoun "it", the router correctly abstained on
it, and the benchmark scored that as failure — halving the reported score and
nearly shipping a diagnostic telling every user their projects were
indistinguishable.

**6. Report abstention separately from error.** They are different outcomes with
different costs, and averaging them hides which one is happening.

**7. Synthetic corpora test robustness, not optimality.** Two synthetic-corpus
bugs during development produced confidently wrong measurements. Use generated
projects to sweep a constant across corpus *shapes*; use real projects to decide
whether the answer is any good.

---

## Phase 1 — The test bed

Nothing downstream is trustworthy without this, and it is mostly built already
(`evals/scale.py` generates scopes with controlled vocabulary overlap).

**1.1 Generalise the corpus generator.** Today it varies scope count and
vocabulary overlap. Add the dimensions that plausibly change the right answer:

- **size skew** — the real corpus spans 82 to 6,469 tokens, a 79× range
- **prose-to-code ratio** — a docs monorepo and a C library sit at opposite ends
- **doc density** — projects with no README at all, versus heavily documented
- **naming convention** — `snake_case`, `camelCase`, `kebab-case`, single-word
- **language mix** — the tokenizer splits on ASCII word characters; a corpus of
  CJK identifiers or accented Latin is untested
- **shared-dependency noise** — vendored code, generated files, lockfiles

**1.2 Assemble a real held-out corpus.** Twenty public repositories chosen for
variety, not convenience: at least one large monorepo, one C/C++ project, one
documentation-only repo, one with non-English docs, one with almost no prose.
This is the set that decides whether a fitted constant is any good; the
synthetic set only decides whether it is *stable*.

**1.3 Make `loci eval` scriptable across corpora.** A harness that builds a
corpus, indexes it, runs `eval`, and emits one row of JSON. Everything after
this phase is a loop over that.

**Done when:** one command sweeps a named constant across ≥8 corpus shapes and
prints accuracy per shape.

---

## Phase 2 — Self-calibrate what is genuinely corpus-dependent

A constant should only become a fitted parameter if its right value provably
depends on the corpus. Otherwise it is a constant and should be defended as one.

**2.1 `SEMANTIC_FLOOR` (highest value).** Currently 0.57, chosen from the gap
between "real" (0.61–0.80) and "nonsense" (0.46–0.52) cosines on one corpus with
one embedding model. Cosine scale is a property of the *model*, and the gap is a
property of the *corpus*, so this is the most obviously mis-generalising
constant in the codebase.

The measurement needs no labels: encode random or shuffled text and take the max
cosine against the corpus. That is the "no relationship" baseline for this
corpus and this model. The floor sits a fixed margin above it. Validate that the
fitted value reproduces 0.57 on the development corpus and moves sensibly on a
corpus with a different model.

**2.2 `SCORE_FLOOR`.** Same technique, same justification. Hand-set at 0.12 with
no evidence.

**2.3 Decide whether `SIZE_PRIOR` is corpus-dependent.** It was optimal at 0.15
on both a symbol-indexed and a prose-only version of the same corpus, which is
weak evidence for stability. Sweep it across Phase 1's size-skew dimension. If
it is flat, freeze it and say so; if it tracks skew, fit it.

**Done when:** every constant is either fitted from a labelled-free measurement,
or documented with the sweep showing it is flat across corpus shapes.

---

## Phase 3 — Fit the eight that were never fitted

These have no evidence at all, and two of them are load-bearing.

**3.1 Fusion weights** (`BM25_WEIGHT` 0.40, `CHAR_WEIGHT` 0.22, `EMBED_WEIGHT`
0.38). Grid-search on the simplex across corpus shapes. Expect the answer to
depend on prose-to-code ratio: character n-grams matter more for identifier-
heavy corpora, embeddings more for prose-heavy ones. If so, this is a fitted
parameter, not a constant.

**3.2 `ALIAS_BOOST` / `CWD_BOOST`.** Both are large enough to dominate, which is
intended — but "6.0 and 4.0" was a guess. The only property that matters is that
they outrank vocabulary evidence; verify the *ordering* holds across shapes
rather than tuning the magnitudes.

**3.3 `MIN_GROUNDED`, `LENGTH_SATURATION`, `MIN_MATCHED`.** `MIN_MATCHED` is
already known not to separate routable from unroutable questions — it survives
only as one of three OR'd gates. Test whether removing it entirely costs
anything across shapes. Deleting a constant is the best possible outcome.

**Done when:** each constant is fitted, frozen-with-evidence, or deleted.

---

## Phase 4 — The known failures

These are diagnosed and unfixed, and none is a tuning problem.

**4.1 Structure store surfaces the answering symbol in 4 of 7 probes.** The
misses are real: one seeds on helper modules instead of the resolver, one has an
82-token symbol vocabulary that the query expansion cannot match at all. Likely
needs query expansion against the graph's own vocabulary — the same trick the
episode store already uses — rather than passing raw tokens to `graphify query`.

**4.2 Large prose-only scopes over-attract.** A scope with thousands of chunks
and no code graph absorbs questions belonging elsewhere. The size prior corrects
for vocabulary size but not for the *kind* of vocabulary. Candidate: normalise
prose-derived and symbol-derived vocabularies separately before comparing.

**4.3 `loci eval` and `loci calibrate` share question families.** A perfect
`eval` score straight after calibrating is partly circular. Split the generated
families into a fit half and a report half.

---

## Phase 5 — Platform and packaging

**5.1 CI on Linux and Windows.** Never executed on either. Two platform bugs
were already found by reading — `os.kill(pid, 0)` terminating on Windows, and
backslash paths defeating glob matching — which is evidence that reading finds
bugs, not that it finds all of them.

**5.2 Python 3.10 through 3.13** in the matrix. The floor is claimed and only
3.14 has been run.

**5.3 First publish to PyPI as `loci-mem`,** with the README's claims restated
as "measured on the author's corpus; run `loci eval` for yours".

---

## Phase 6 — The only thing that actually settles it

**Someone else runs `loci eval` on a corpus that is not the author's and reports
what it says.**

Every phase above tightens the fit against corpora chosen by the same person who
wrote the router. That is worth doing — it is how the fitted constants stop
being one machine's habits — but it cannot detect a blind spot shared by the
author and the test bed. Phase 6 can start the day Phase 5 ships and should not
wait for Phases 1–4 to finish.

---

## Sequencing

```
Phase 1  test bed            ── everything depends on it
   ├── Phase 2  self-calibrate      (highest value: SEMANTIC_FLOOR)
   ├── Phase 3  fit the unfitted    (can run in parallel with 2)
   └── Phase 4  known failures      (independent of 2 and 3)
Phase 5  platform + publish   ── unblocks Phase 6
Phase 6  external validation  ── the only phase that settles anything
```

The honest expected outcome is not "perfect". It is: every constant carries
either a fitted value or a sweep showing it does not matter, the tool measures
itself on whatever corpus it is pointed at, and the remaining error is
attributable to a property of the corpus rather than to a guess made in
August 2026.
