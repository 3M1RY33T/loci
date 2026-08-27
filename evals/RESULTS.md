# Eval results

98 items over 10 scopes. Run: `python3 evals/run_eval.py --misses --retrieval`

## Method

Three families, and the reason there are three is that they can prove different
things and should never be averaged together.

**A — taxonomy (80 items, generated).** Eight generic developer questions
applied mechanically to all ten scopes. Written without reference to any corpus,
and not selected per scope, so there is no opportunity to pick questions that
happen to work. All are *deictic* — they say "this project", never a name.

**B — behavior (7 items, hand-authored).** Written by reading **code bodies**,
which neither store indexes: the structure graph holds labels and paths, the
episode store holds prose. Questions are phrased in domain terms that avoid the
identifier, so answering requires bridging intent to implementation. Five of
seven are marked `contamination: none`; two derive from indexed prose and are
marked `prose`, because those can only demonstrate ranking, not recall.

**C — cross / confusable / negative (11 items).** Questions that legitimately
span scopes (`urthreads` and `MyBlog` both use D1; `TSRC` and `zim-compress`
both handle ZIM archives), plus six that no scope can answer — two nonsense, two
vague, and two that are technically specific but describe work nobody here does
(Kubernetes ingress, Stripe webhooks). The last pair matters: plausibility alone
must not produce an answer.

## Routing

```
taxonomy (deictic, 80 items)
  with cwd      top-1 100.0%    abstained   0.0%
  without cwd   top-1   1.2%    abstained  87.5%   <- abstaining is CORRECT here

family            n   top-1  hit@widen  goldcov   set
behavior          7   85.7%      85.7%    78.6%  1.43   (uncontaminated only)
confusable        2  100.0%     100.0%    50.0%  1.50
cross             3  100.0%     100.0%    83.3%  2.67
negative          6   66.7%      66.7%    66.7%  7.33
```

**The finding this set exists to have produced.** At 80 deictic items the
picture is far worse than a 4-item spot check suggested. Before the fix below,
deictic questions without a working directory abstained only **37.5%** of the
time — meaning **56% were confident and wrong**, which is worse than useless
because it is indistinguishable from correct from the outside.

No lexical statistic rescues this. Per-token evidence for `start` (2.51) and
`services` (2.56) is indistinguishable from real evidence like `session` (2.53).
The individual words are not generic; the *question* is. Deixis is a grammatical
property, so `router.is_deictic` detects it grammatically — a closed class of
markers, not a tuned threshold — and abstains unless cwd or an alias has already
identified the subject. That moved abstention **37.5% → 87.5%** with cwd routing
unchanged at 100%.

### Known costs and remaining misses

- **One false abstention** from that fix: `beh-del-biggest` contains an
  anaphoric "it" that refers to something inside the sentence, not the project.
  Broadening the pattern to bare pronouns bought +50 points on 80 items and cost
  this one; the trade is recorded, not hidden.
- **Two negatives still route** (`Kubernetes ingress`, `Stripe webhook`). Both
  are specific enough to match scattered vocabulary in a large prose scope.
- **`odysseus` is a vocabulary sponge.** With 3,848 prose tokens it absorbs
  questions belonging elsewhere — it captured both a `zim-compress` and an
  `urthreads` question. Large prose-only scopes need a stronger size prior than
  graph-backed ones.
- Two taxonomy phrasings (`How do I run the tests?`, `How do I set up a
  development environment?`) carry no deictic marker at all and depend on the
  count gate alone.

## Retrieval

Scope forced to gold, so this measures the store, not the router.

```
evidence returned              12/12
answer content found @1        10/12
answer content found @3        11/12
  ...uncontaminated only  @1    5/7      @3   6/7
```

The single miss is `beh-ody-windows-clicks`, and it is the most useful result in
the set.

## The sharpest finding: nobody indexes docstrings

The question *"Why would the interface load fine but stop responding to clicks
on Windows?"* has a precise written answer — a five-line docstring on
`register_static_mime_types()` in `odysseus/app.py`, explaining that stale
Windows registry mappings serve `.js`/`.mjs` with a non-JS `Content-Type`.

Neither store can see it. The structure graph indexes the *label*
`register_static_mime_types()` and its path; the episode store indexes README,
docs and commits. **The docstring — the actual explanation — is invisible to
both.** Retrieval returned commits about API-key encryption instead.

This was surfaced by a mistake in the eval itself: the first version of the
labels named code files as expected sources, which no episode store could ever
return. Fixing the metric exposed the real gap rather than hiding it.

Docstrings and comments are where a large share of "why" is written down, and
they sit in the seam between the two stores. Closing that seam — a third
collector that extracts docstrings and comment blocks as episode chunks with
their symbol as heading — is the highest-value next change to loci.

## After adding the docstring collector

`backends/docstrings.py` mines docstrings (Python via `ast`, C-family via block
and line comments) and attributes each to its enclosing symbol, so a hit cites
`app.py > register_static_mime_types()`.

Coverage moved most where `doctor` had been complaining loudest:

```
hlep_davay      15 -> 1379 chunks      (its README is a framework template)
beacon           2 ->   20 chunks
odysseus       542 -> 5948 chunks
Delroy        1979 -> 8320 chunks
```

Retrieval improved, and the eval's own failing item was fixed:

```
answer content @1        10/12 -> 11/12       uncontaminated 5/7 -> 6/7
beh-ody-network-drive    miss  -> rank 1, from code:app.py (the comment block)
beh-urth-cookie          git commit -> code:src/worker-security.mjs (the fix itself)
```

**And routing collapsed.** Folding docstring vocabulary into the routing index
more than doubled one scope's vocabulary, made it top-ranked for questions
belonging elsewhere, and dropped cross-scope routing from 100% to 33.3%;
uncontaminated top-1 fell 85.7% -> 71.4% and mean set size went 1.43 -> 3.29.

The fix is a units argument, not a tuning one. A scope's *symbol* vocabulary is
bounded by its code; its *docstring prose* is not. Mixing them makes scopes
incomparable and the size prior can no longer correct for it. `ROUTING_KINDS`
therefore excludes `docstring` — the chunks stay fully searchable for retrieval,
where more prose is unambiguously better, and contribute nothing to routing,
where the unit mismatch is fatal.

With that split, both wins hold at once: routing back to 100% cross / 85.7%
uncontaminated, retrieval at 11/12.

The last remaining miss stopped being a coverage problem and became a ranking
one: odysseus contains *two* Windows-specific explanations (a MIME-type fix and
a symlink fix) and the retriever returns the wrong one. `--rerank` finds it at
rank 3, taking the behavior family to `@3 7/7` (uncontaminated 5/5) while
costing one position at rank 1 — the same trade reranking has shown throughout.

## Cold start: the same eval with no code graphs

Built in a sandbox with graphify disabled and no notes of any kind — git
history, READMEs and docstrings only. 10 scopes, 16,015 chunks, 41s to index
plus 49s to embed.

**Read this as an onboarding result, not a head-to-head.** An earlier version of
this section reported cold gold-coverage (92.9% vs 78.6%) and cold hit@widen as
improvements. They are not: both rise mechanically when a router returns 3.29
scopes instead of 1.43. Compared at matched set size, warm wins everywhere and
the gap does not close:

| config | top-1 | scopes returned |
|---|---|---|
| **warm (graphify)** | **85.7%** | 2.50 |
| cold (no graphify) | 57.1% | 3.58 |
| cold, narrowed to warm's set size | 57.1% | 2.75 |
| warm, top-1 only | **85.7%** | — |
| cold, top-1 only | 57.1% | — |

**28 points of top-1 routing accuracy is what the structure store buys**, and
that is only its contribution to *routing*. Its actual product — call-graph
traversal with `file:line` citations — is scored by the `structure` family and
by nothing else:

| | warm | cold |
|---|---|---|
| answering symbol surfaced | **4/7** | 0/7 (no graph to traverse) |

The retrieval rows are identical warm and cold (11/12, uncontaminated 6/7) for a
structural reason rather than an interesting one: retrieval is measured with the
scope forced and `with_structure=False`, so graphify is never invoked in that
measurement at all. It is not a comparison.

What cold start *does* establish is that a new user is not handed an empty tool.
On day one, with no graphs and no notes, they get cwd routing at 100%,
abstention on deictic questions at 87.5%, negatives refused at 66.7%, the
correct scope inside the returned set 85.7% of the time, and a fully functional
episode store. What they do not get is precision, or any structural answer at
all.

One second-order effect: cold evidence scores run much higher (4.2–4.4 against
1.8–2.8 warm). With no symbol vocabulary, scope vocabularies are smaller and
ordinary prose tokens look more distinctive, pushing more questions past
`DECISIVE_EVIDENCE`. That is the mechanism behind the wider sets.

`loci graphs` closes the gap in seconds per repo, for free.

## Threshold re-fit

The cold run suggested the thresholds were fitted to a symbol-rich corpus and
would need re-fitting for a prose-dominant one. They were swept against both
corpora at once — the same ten projects, indexed with and without code symbols —
so the result is not tuned to a single corpus shape.

**No corpus-specific constants were needed.** One set is optimal for both:

| constant | was | now | evidence |
|---|---|---|---|
| `SIZE_PRIOR` | 0.15 | **0.15** | optimal in both. 0.05 costs cwd routing (100% → 94%/89%); 0.30 costs top-1 (85.7% → 71.4%); 0.45 is catastrophic |
| `WIDEN_RATIO` | 0.70 | **0.85** | strictly dominates in both — identical top-1 and hit@widen, consistently smaller sets |
| `MIN_MATCHED` | 3 | **4** | refuses one more plausible-but-absent question; converts one confidently-wrong answer into an abstention; no top-1 loss |
| `DECISIVE_EVIDENCE` | 4.5 | **4.5** | see below |

Result, both corpora:

| | warm before | warm after | cold before | cold after |
|---|---|---|---|---|
| deictic **with cwd** | 100% | **100%** | 100% | **100%** |
| deictic without cwd → abstains | 87.5% | **100%** | 87.5% | **100%** |
| negatives refused | 66.7% | **83.3%** | 66.7% | **83.3%** |
| uncontaminated top-1 | 85.7% | 85.7% | 57.1% | 57.1% |
| uncontaminated hit@widen | 85.7% | 85.7% | 85.7% | 85.7% |
| scopes returned when routed | 2.50 | **1.67–2.43** | 3.58 | **1.70** |

Every gain is in *abstention* and in *set width*; no accuracy was traded for it.
That is the direction this design prefers to be wrong in.

### Two constants that were not what they looked like

`DECISIVE_EVIDENCE` is nearly inert. Swept from 3.67 to 5.90 it changes **no
metric on either corpus**, and disabling it entirely changes 1 item in 18 and
0 of 80 deictic questions. The hypothesis it was tested under — that cold's
higher evidence scores (4.2–4.4 against 1.8–2.8 warm) were pushing questions
past it and widening the sets — was simply wrong; it is not binding at all.

It is kept because the single item it does change is exactly its purpose: a
question matching only two tokens where one of them is exclusive to a scope and
prominent inside it. Raising `MIN_MATCHED` makes that escape hatch matter more,
not less.

A percentile-calibrated form was also tried — `DECISIVE = p99` of each corpus's
own evidence distribution, which is 4.91 warm and 5.43 cold. It made no
difference to any metric, for the same reason. Absolute is simpler and measured
equivalent.

## Rejected: conditional docstring routing

Docstrings are excluded from routing vocabulary. The obvious objection is that a
cold machine has no code graphs, so thin scopes route on almost nothing — one
project here has 114 prose tokens and 3,237 once docstrings count, and its
README is an unmodified framework template. Making the exclusion conditional on
whether a scope has symbols was implemented and measured:

| variant | cwd | clean top-1 | cross | negatives | set |
|---|---|---|---|---|---|
| warm, fallback off | 100% | **85.7%** | **100%** | 67% | 2.50 |
| warm, fallback on | 100% | 71.4% | 67% | 67% | 3.83 |
| cold, docstrings off | 100% | **57.1%** | **67%** | **67%** | 3.58 |
| cold, docstrings on (uniform) | 100% | 28.6% | 33% | 50% | 3.08 |

Warm, the fallback fired on exactly one scope (+478 tokens) and cost 14 points,
which suggested the damage was *asymmetry* — adding vocabulary to some scopes
and not others distorts a comparison that is between scopes. Applying it
uniformly on a cold machine falsified that hypothesis: accuracy halved there too.

The actual reason is what docstrings are made of. They are high-volume generic
English — "returns", "the value", "default", "configuration", "path", "error" —
which is maximally non-discriminative, and at thousands of chunks per scope it
drowns the signal that symbol names and commit subjects carry. Docstrings are
excellent **retrieval** material and poor **routing** material, and having no
graph does not change that.

The fix for a cold scope is `loci graphs`, which is free and takes seconds.

## A drop I misdiagnosed as noise

Pruning vendored directories out of traversal moved uncontaminated top-1 from
85.7% to 71.4% and cross-scope routing from 100% to 66.7%. I attributed that to
single-item variance on a seven-item family, re-swept `MIN_MATCHED` to confirm
the threshold was not the cause, and recorded it as noise.

It was a bug. The replacement traversal matched globs with `fnmatch`, which has
no notion of `**`, so `docs/**/*.md` -- one of the three default episode globs --
compiled to something requiring an intervening directory and silently skipped
every `docs/guide.md`. **28 documentation files were dropped**, 25 of them from
one project. Fixing the matcher to use pathlib's semantics (`**` matches *zero*
or more directories) restored both numbers exactly:

| | before pruning | after pruning (bug) | after fix |
|---|---|---|---|
| uncontaminated top-1 | 85.7% | 71.4% | **85.7%** |
| cross-scope | 100% | 66.7% | **100%** |

The lesson is not about globs. Re-sweeping the threshold was the right instinct
and it correctly cleared `MIN_MATCHED` -- but ruling out the *parameter* is not
the same as ruling out the *corpus*, and "small sample, call it noise" is
exactly how a real regression gets filed away. The cheap check that would have
caught it immediately -- compare the file list before and after a traversal
change -- now runs as a test.

## Scale

Synthetic scopes, each with its own domain vocabulary plus a shared pool of
ordinary engineering English. Overlap is the fraction of a scope's domain terms
that its same-domain peers also use.

| scopes | overlap | index | route | abstains | top-1 of routed | set |
|---|---|---|---|---|---|---|
| 25 | 0.0–0.4 | 0.7s | 0.1ms | 0% | **100%** | 1.00 |
| 25 | 0.6 | 0.7s | 0.2ms | 0% | 88% | 1.12 |
| 25 | 0.8 | 0.7s | 0.1ms | 0% | **36%** | 2.48 |
| 50 | 0.0–0.4 | 1.4s | 0.3ms | 0% | **100%** | 1.00 |
| 50 | 0.6 | 1.4s | 0.3ms | 0% | 90% | 1.20 |
| 50 | 0.8 | 1.4s | 0.3ms | **90%** | — | 3.00 |
| 100 | 0.0–0.4 | 2.8s | 0.5ms | 0% | **100%** | 1.00 |
| 100 | 0.6 | 2.8s | 0.6ms | 0% | 90% | 1.20 |
| 100 | 0.8 | 2.8s | 0.5ms | **90%** | — | 3.00 |

**Scope count is not the limit.** From 25 to 100 scopes accuracy is identical at
every overlap level; index time and routing latency both scale linearly and
routing stays under a millisecond.

**Vocabulary overlap is the limit**, and the interesting result is that the
failure gets *safer* as scope count rises. At 0.8 overlap with 25 scopes the
router answers, and is wrong 64% of the time. With 50 or 100 scopes the same
questions abstain 90% of the time, because more scopes sharing a term means less
evidence for any one of them and the gate fires. A small corpus of highly
similar projects is the dangerous configuration -- just enough evidence to look
confident.

A first attempt at this measurement cycled ten fixed domains, so at fifty scopes
five shared identical vocabulary and the gold label was genuinely ambiguous.
That measured the generator, not the router; terms are now unique per scope.

## Honest limits## Honest limits

- One machine, ten scopes, one author. The author had read parts of these
  corpora before writing family B, which is why `contamination` is a field and
  why uncontaminated results are always reported separately.
- Family A is unbiased by construction; families B and C are not, and their
  sample sizes (7, 11) are small enough that single items move percentages a lot.
- `negative` at 66.7% is 4 of 6 — do not read that as a rate.
