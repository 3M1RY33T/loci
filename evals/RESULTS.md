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

## Phase 1: the test bed

`evals/corpus.py` generates corpora with controlled shape; `evals/sweep.py` runs
`loci eval` across them and can sweep any constant. Dimensions: scope count,
vocabulary overlap, pair-shared terms, size skew, prose-to-code ratio, doc
density, naming convention, character set, vendored noise.

```
shape            cwd  nocwd  nons  signat  contend   set  secs
baseline        100%   100%  100%    100%      86%  1.86   1.8
many-scopes     100%   100%  100%    100%      83%  1.83   3.1
size-skew       100%   100%  100%    100%     100%  2.00   3.3
undocumented    100%   100%  100%    100%      43%  1.43   1.2
cjk             100%   100%  100%    100%     100%  2.00   1.3
vendored        100%   100%  100%    100%     100%  2.00   1.4
```

### Three real bugs it found

**Mixed-script identifiers lost their non-Latin half.** `invoice設定` tokenized
to `['invoice']`. The camelCase pattern matches ASCII Latin only, and applying it
to a mixed chunk silently discarded everything else. Tokens now split into
per-script runs first; only Latin runs are case-split. Pure CJK survives
unsegmented -- imperfect for languages written without spaces, but not lost.

**A length floor tuned for English hid a whole writing system.** `signature_terms`
filtered `len(t) < 4`, excluding every two-character CJK token. The benchmark
reported a CJK corpus as unroutable without ever having looked at its
vocabulary. The floor is now script-aware, as is the tokenizer's.

**Scopes with no distinctive vocabulary beyond their own name were scored as
failures.** A scope's name is excluded from signature terms on purpose -- a
question containing it would route on the alias boost and test nothing -- but a
scope with nothing else was then counted as a miss. `loci eval` now reports
those as unmeasurable and names them.

### A new family: `contended`

Terms held by exactly two scopes, asked as `"how is X handled?"`, gold being both
owners. It is the only family where more than one scope has a real claim, and
therefore the only one where widening does anything.

It immediately found a genuine weakness on the real corpus: **8.3%**. For terms
like `glasses` (shared by Delroy and G2-claude-companion) loci returns both
owners once in twelve tries. This is the multi-scope recall problem that was
suspected from cross-scope gold coverage but never had a metric.

### Five artifacts in the bed itself, and what they cost

Every one produced a confidently wrong number before being caught:

1. **Cycled domains** -- ten domains over fifty scopes made five identical, so
   the gold label was ambiguous and routing "collapsed" at scale.
2. **Best-case term sampling** -- drawing only a scope's rarest words put the
   fitted evidence band two points too high.
3. **Domain-named scopes** -- naming a scope after its own vocabulary made that
   vocabulary an alias, and aliases are excluded, so three shapes scored 0%.
4. **Overlap shared within a domain only** -- with more domains than scopes each
   scope held a unique domain and the parameter did nothing. The entire bed was
   insensitive: `SIZE_PRIOR` from 0.0 to 1.2 moved no metric at all.
5. **Shared terms concatenated first** -- the README uses `terms[0..3]`, so past
   50% overlap the prose held nothing distinctive while unique terms sat in
   docstrings, which routing excludes. The metric fell off a cliff instead of
   degrading.

The pattern is worth stating plainly: **a synthetic benchmark fails silently and
looks like a finding.** Four of these five presented as a defect in loci.

### What the bed cannot do

`WIDEN_RATIO` and `SIZE_PRIOR` remain unmeasurable here. Swept across their full
range on every shape, neither moves any metric. Synthetic pair-shared terms are
equally prominent in both owners, so the scores tie and any widen ratio keeps
both; real corpora have a term that is central to one project and marginal in
another, which is exactly when widening matters.

**Phase 3 must fit those two against real repositories, not generated ones.**
That is roadmap rule 7 -- synthetic corpora test robustness, not optimality --
arriving as a measurement rather than a principle.

## Phase 1.2: real held-out repositories

`evals/real_corpus.py` holds a manifest of twenty public repositories chosen for
shape variety rather than convenience, with a resumable shallow fetcher and
named subsets. Nothing in it is the author's, and it is deliberately not all
Python -- a tokenizer and a set of thresholds fitted on one language's naming
conventions is exactly what this set exists to detect.

Seven repositories spanning C, Rust, Go, Java, Python, JavaScript and Ruby
(cJSON, ripgrep, hugo, guava, flask, express, rails; 149MB, 11s to index):

```
7 scopes | random guessing would score 14.3%

family                  n  correct  scopes
deictic + cwd          56  100.0%       -
deictic, no cwd         8  100.0%       -
unanswerable            6  100.0%       -
signature              14  100.0%     1.0
contended              12    0.0%       -
```

**Four families out of five transfer intact to code nobody involved has read.**
That is the strongest generalisation evidence the project has: cwd routing,
abstention on deictic questions, refusal of nonsense, and scope
distinguishability all hold on seven languages at once.

### `contended` fails everywhere, and that is the finding

Zero percent here, zero on a four-repo subset, 8.3% on the development corpus.
Three independent corpora agree: **loci does not return both owners for a term
two projects genuinely share.** Cross-project questions under-report, and the
fix is not a threshold -- see below.

### What the real corpus exposed that generated ones could not

**A floor fitted elsewhere is simply wrong.** Run uncalibrated, the real corpus
was routed with `EVIDENCE_FLOOR = 7.6` -- fitted to the author's repositories --
while its own contended questions score around 6.0, so every one abstained. This
is the generalisation problem happening live rather than argued about.

**Calibration had a blind spot.** Its routable sample was drawn only from
signature questions, which are structurally high-evidence. Contended questions
occupy a lower band entirely, so the fitted floor sat above the range they live
in and refused all of them. They are now part of the sample. On this corpus it
did not move the floor -- the sweep still prefers refusing 14 unroutable
questions to admitting 12 contended ones -- which says the fix belongs in
ranking, not in the threshold.

**Parser warnings leaked to the user.** Indexing somebody else's repository
printed `SyntaxWarning: "\*" is an invalid escape sequence` from `ast.parse`.
A user can do nothing about a warning in code they did not write.

**The verdict line contradicted its own table.** `contended` sat at 0% across
three corpora while the summary read "routing looks healthy on this corpus".
A summary that disagrees with the numbers above it is worse than no summary.

### Prose-only and non-English subsets

Both were fetched on the theory that they were the most likely to break
something. Routing held: four repositories of pure documentation
(public-apis, free-programming-books, rust-lang/rfcs, awesome-python) and three
mixed Chinese/English documentation repositories each scored 100% on the four
families that transfer, with `contended` failing as everywhere else.

**What the Chinese corpus exposed is a retrieval defect, not a routing one.**
CJK reached the index -- 55% of the routing vocabulary -- but unsegmented, one
token per phrase rather than per word:

```
usable as search terms (<= 4 chars)   821 / 2373   (35%)
effectively unmatchable (> 8 chars)   597 / 2373
longest token                          30 characters, a whole sentence
```

A thirty-character token is reachable only by a query containing that exact
sentence. Character bigrams are the standard answer where no segmenter is
available, and the whole run is still kept so an exact phrase matches exactly:

| | before | after |
|---|---|---|
| CJK tokens usable as search terms | 35% | **79%** |
| 30-char phrase reachable by a 2-char query | no | **yes** |
| index size | 231KB | 339KB |

This is the clearest example of why the held-out set exists. Every English
corpus in the project scores identically with and without the fix; the defect is
invisible unless the corpus is non-English, and it had been shipped.

**One manifest entry was wrong.** `vuejs/docs` was labelled "i18n, mixed
scripts" and contains no CJK at all -- it is the English documentation site. It
is kept as an explicit control, and a genuinely Chinese-prose repository
(`doocs/advanced-java`) was added. With the corrected set, `contended` rose from
0% to 17% -- the first non-zero score on any real corpus.

### Cost

```
smoke        4 repos    12MB    2s to index
polyglot     7 repos   149MB   11s to index
prose        4 repos    20MB
nonenglish   3 repos    41MB
```

Subsets are named so the expensive ones stay opt-in.

## Phase 3: fitting the constants that were never fitted

`evals/fit.py` sweeps a constant across both beds at once -- four synthetic
shapes and the real subsets -- because Phase 1 established that neither alone
is sufficient.

### `CWD_BOOST = 4.0` is validated, and only the real bed could do it

| value | synthetic beds | real repos |
|---|---|---|
| 0.5 | 100% | **61%** |
| 1.0 | 100% | 96% |
| 1.5 – 20.0 | 100% | **100%** |

The constraint is a floor at about 1.5, not a peak, and above it the band is flat
to at least 20. The shipped 4.0 sits comfortably inside it. Every synthetic shape
scores 100% at every value, because generated scopes have vocabulary distinctive
enough that the working directory never has to break a tie -- exactly the
prediction Phase 1 made about what synthetic corpora cannot measure.

### `MIN_MATCHED = 4` is load-bearing, and this benchmark cannot see that

Removing it costs 14 points of uncontaminated top-1 and **67 points of
cross-scope accuracy** on the held-out hand-authored set. Yet swept across every
bed, `loci eval` reports it as completely inert.

The two questions that need it are specific:

```
cross-cloudflare   matched=4  evidence 6.70  strongest single token 2.14
cross-macos-app    matched=4  evidence 6.47  strongest single token 2.34
                                             (evidence floor: 7.17)
```

Both are cross-scope questions made of four *ordinary* tokens, none individually
decisive. `MIN_MATCHED`'s real job is letting breadth compensate for depth, and
it operates only in that narrow band.

Three attempts to generate that shape auto-labelled all failed:

1. **A `detailed` family using four rare terms** -- overwhelming evidence, sails
   past every gate, 100% at every value.
2. **The same family drawn from ranks 4-8** -- now discriminates between beds
   (42% to 100%) but still flat across `MIN_MATCHED`.
3. **Lengthening the `contended` questions** to reach the matched-token range --
   dropped that family to 0% on every bed, because the added ordinary words
   pulled the top scope away from the two owning the term. Reverted.

The honest conclusion is that `loci eval` cannot measure this constant, not that
the constant is inert. That bounds what the self-service benchmark can tell a
user, and it is worth stating plainly: **a user who tunes on `loci eval` alone
would find no reason to keep a setting worth 67 points.**

### `ALIAS_BOOST` is unmeasurable by construction

Flat from 0.5 to 20.0 on every bed. The benchmark deliberately excludes scope
names from its questions -- a question containing one would route on the alias
boost and test nothing -- so it can never produce a case where that boost is
marginal. Not a gap to close; a consequence of the design.

### The retrieval metric

`evals/retrieval.py` asks a section's own **heading** and looks for its body,
with the heading stripped from what is indexed. A heading is written by a human
as a compressed description of the section below it, so it overlaps its answer
partly in wording and partly only in meaning -- the mix a real question has.

The obvious alternative, building a query from a chunk's own words, rewards
lexical overlap by construction and would drive `EMBED_WEIGHT` to zero as an
artefact of the metric. So the bias is **measured rather than assumed**: every
query is also scored under each ranker alone.

```
208 heading queries over bodies that never contain them
  both rankers find it       78  (61%)
  only lexical finds it      35  (28%)
  only semantic finds it     14  (11%)
```

Eleven percent need the semantic ranker and nothing else, so the metric is not
secretly lexical and weights fitted on it can be trusted. If that number were
zero the report says so and tells the reader to treat `EMBED_WEIGHT` as unfitted.

Both rankers are blinded, not just one. The shipped vectors were built over
`heading + text`; reusing them would let the semantic side match a query against
its own heading while the lexical side could not, which is worse than no
blinding at all. Bodies are re-encoded without headings for this measurement.

### Fusion weights, fitted

Two independent corpora -- ten repositories and seven, 208 and 251 queries:

| weights | corpus 1 MRR | corpus 2 MRR |
|---|---|---|
| bm25 only | 0.374 | 0.461 |
| char n-grams only | 0.324 | -- |
| embeddings only | 0.346 | 0.481 |
| 0.40 / 0.22 / 0.38 *(previous)* | 0.395 | 0.495 |
| **0.20 / 0.20 / 0.60** *(fitted)* | **0.406** | **0.526** |

**Fusion beats every single ranker on both corpora.** That is the first direct
evidence that the three-ranker design earns its complexity rather than being
assumed to. Both corpora peak at the same point independently, and the band is
flat for `EMBED_WEIGHT` between roughly 0.4 and 0.7, falling away at 0.8 -- so
0.6 is the middle of a real region, not a sharp peak.

A trap on the way: the second corpus was first swept without its vectors built,
so every "embed weight" row silently renormalised to a lexical-only mix. The
embeddings-only baseline scoring **0.000** is what exposed it. A sweep without a
solo-ranker baseline would have reported those numbers as fitted.

### `LENGTH_SATURATION = 260` validated

| value | 80 | 160 | **260** | 400 | 800 |
|---|---|---|---|---|---|
| MRR | 0.354 | 0.364 | **0.384** | 0.342 | 0.267 |

A clear peak with falloff on both sides. This one was hand-set and happens to be
right, which is worth saying plainly: it was a guess, and it is now a
measurement.

### `MIN_GROUNDED` is still unfitted

It reads flat across 0-3, but that is the harness rather than the constant: the
retrieval metric disables the grounding gate so that every query produces a
measurable rank. A constant that decides whether to return *anything* cannot be
fitted by a metric that always demands a ranking. It needs a measurement that
scores refusals, which does not exist yet.

## Phase 4: the known failures

### The sweep harness was broken, and several Phase 3 conclusions were wrong

`route()` bound its thresholds as **default arguments**, which Python evaluates
once when the function is defined. Setting `router.MIN_MATCHED = 6` therefore had
no effect on any caller using the default -- so every threshold sweep run
through `evals/fit.py` and `evals/sweep.py` was a no-op, and each one reported
"flat" because nothing had changed.

`CWD_BOOST` and `ALIAS_BOOST` were unaffected, because they are read from the
module inside the function body rather than bound in the signature. That is why
`CWD_BOOST` showed a real effect while everything else looked inert, and the
inconsistency should have been the clue.

All thresholds now resolve from the module at call time, with a test that fails
if any of them regresses to a default argument. The corrected sweeps:

| constant | corrected finding |
|---|---|
| `MIN_MATCHED` | **not** inert. At 2, real-corpus abstention falls to 75% and nonsense refusal to 67%. 4 is correct. |
| `SIZE_PRIOR` | safe band [0.15, 0.35]; 0.6 collapses signature to 43%. |
| `WIDEN_RATIO` | genuinely flat 0.55-0.95, now measured rather than assumed. |
| `DECISIVE_EVIDENCE` | redundant -- see below. |

The Phase 3 claim that "`loci eval` cannot measure `MIN_MATCHED`" was an artefact
of this bug. Three attempts to build a family that could measure it were made
against a harness that could not have detected any change at all.

### `contended`: 8.3% to 100%

The consistent failure across five corpora. Two distinct causes, and the smaller
one was the obvious one:

```
term       owner A                  owner B              ratio  outcome
software   G2-claude-companion 2.97 3M1RY33T 2.79         0.94  ABSTAIN
curl       G2-claude-companion 2.84 brewery 2.18          0.77  ABSTAIN
glasses    G2-claude-companion 3.65 Delroy 1.52           0.42  one owner only
venv       Delroy 1.96              odysseus 1.77         0.90  both
```

Five of eight **abstained outright**, so widening never got a chance -- and on
real repositories that was every single one. Widening was the visible symptom;
the gate was the cause.

A question about a term two projects share splits its evidence between them by
construction. "How is curl handled?" carries two tokens, so its summed evidence
is low simply because there are two; its matched count is below the count gate
for the same reason; and the old `DECISIVE_EVIDENCE` was calibrated for tokens
exclusive to a single scope, which a shared term can never be. It fell through
every gate.

What identifies such a question is **concentration**: a token held by very few
scopes and prominent inside them. Measured only on questions the evidence gate
actually decides -- deictic ones are refused earlier by grammar and never reach
it -- the bands separate cleanly:

```
contended questions      3.12 and above
nonsense and vague       2.05 and below
```

The new tier admits them and forces every holding scope into the result, because
returning one owner of a shared term is the bug being fixed. `contended` went to
**100% on all six beds**, mean set 2.0, with every other family unchanged.

### `DECISIVE_EVIDENCE` deleted

It admitted a question carrying one exceptionally discriminative token. The
concentrated tier covers that and more: a token held by very few scopes is what
"decisive" was reaching for, expressed in a way that also identifies *which*
scopes. Disabling it changed no metric on any of six beds or on the held-out
set, so it was deleted rather than kept as a second name for the same idea.

### The trade this made, stated plainly

`CONCENTRATED_EVIDENCE` is a real trade-off with no dominant side:

| value | held-out negatives | contended |
|---|---|---|
| **2.6** *(shipped)* | 67% (4 of 6) | **100%** |
| 3.0 | 83% (5 of 6) | 67% |
| disabled | 83% | 8% |

At 2.6 one question leaks: *"Where is the Stripe webhook signature verified?"*
routes to the two scopes containing `webhook` (evidence 2.96). No project here
uses Stripe, so the labelled answer is to abstain -- though returning projects
that genuinely mention webhooks is a near miss rather than a fabrication.

2.6 is shipped because it costs one near-miss and buys four genuine cross-project
questions. A stricter 3.0 is available and the cost is recorded here rather than
hidden.## Phase 1: the test bed

`evals/corpus.py` generates corpora with controlled shape; `evals/sweep.py` runs
`loci eval` across them and can sweep any constant. Dimensions: scope count,
vocabulary overlap, pair-shared terms, size skew, prose-to-code ratio, doc
density, naming convention, character set, vendored noise.

```
shape            cwd  nocwd  nons  signat  contend   set  secs
baseline        100%   100%  100%    100%      86%  1.86   1.8
many-scopes     100%   100%  100%    100%      83%  1.83   3.1
size-skew       100%   100%  100%    100%     100%  2.00   3.3
undocumented    100%   100%  100%    100%      43%  1.43   1.2
cjk             100%   100%  100%    100%     100%  2.00   1.3
vendored        100%   100%  100%    100%     100%  2.00   1.4
```

### 4.1 Structure store: 4/7 to 6/7

graphify seeds a traversal by lexical similarity to a node label, so a question
phrased in *behaviour* reaches code named for its *implementation* only by
accident:

```
"what parses the command line arguments?"      expansion empty, no traversal ran
"what removes stored data from the browser?"   seeded getStoredWorkerUrl(),
                                               wanted clearStorage()
```

Passing raw tokens instead does not help -- `resolve_link` contains none of
"short", "code" or "looked". These are semantic gaps, not expansion bugs.

The fix embeds each scope's **symbol labels** at index time, matches the question
against them, and hands graphify the tokens of the nearest ones. Width matters
more than the idea:

| nearest labels used | structure probes |
|---|---|
| 1 | 6/7 |
| **2** | **6/7** |
| 3 | 5/7 |
| 4 | 5/7 |

Past two, the added tokens are generic -- "logic", "admin", "mutation" -- and
dilute queries the lexical expansion had already aimed correctly. The first
attempt used four and *regressed* a probe that previously passed.

The remaining miss is `beacon-resolve`, whose gold label is arguable: asked
"what happens when a short code is looked up?", the traversal returns
`shortcode.py` and its helpers, which answers the question.

**Cost:** the vector cache grew from 20MB to 105MB. Capped at 30,000 symbols per
scope; absent vectors degrade to lexical seeding silently.

### 4.2 The prose-scope sponge: investigated, not fixed

Three corrections were tried and all three failed.

**Unit-free prominence.** Prominence is `df / node_count`, and `node_count` means
542 *chunks* for a prose scope but 732 *symbol nodes* for a graph-backed one.
Replacing it with `df / max_df` should have removed the mismatch; the
`node_count / max_df` ratio turned out to be 1.3-2.1 everywhere, so nothing moved.

**A stronger size prior.** Strictly worse, and revealingly -- the question did not
go to the right scope, it went to smaller *wrong* ones:

| SIZE_PRIOR | clean top-1 | where the question went |
|---|---|---|
| 0.15 | 85.7% | odysseus |
| 0.25 | 71.4% | G2-claude-companion |
| 0.35 | 57.1% | brewery |

**Prose vocabulary for every scope.** The real asymmetry: a scope with a code
graph routes on *symbols*, one without routes on *prose*, so a documented
project's own words are invisible to routing.

```
question: browser silently discard login cookie curl accepts

urthreads  routes on symbols | routing vocab: browser, login, cookie
                             | its prose:     browser, silently, login, cookie, curl
odysseus   routes on prose   | routing vocab: all six
```

odysseus wins not because it is bigger but because it is allowed to use its
prose. Folding prose into every scope did not fix the item and regressed
`contended` from 100% to 67%. Reverted.

Recorded as open. The failing item may also not be a defect: odysseus has
genuine login and cookie handling and the question names no project.

### 4.3 The eval/calibrate circularity: closed

`loci calibrate` fitted its threshold on the same families `loci eval` reported
on, so a perfect score straight after calibrating was partly the fit reading its
own training data back. The generated families now split into disjoint halves --
deterministic, nothing in both, together the whole -- with calibration fitting on
one and evaluation reporting on the other. The score is unchanged at 100%, which
is the point: it now means something it did not before.

## `SEMANTIC_FLOOR` self-calibrated

The last hand-picked threshold, flagged three times as most likely to
mis-generalise. It was read off one corpus with one embedding model: real
questions scored 0.61-0.80 and unrelated ones 0.46-0.52, so 0.57 sat in the gap.
Cosine scale belongs to the *model* and the size of that gap to the *corpus*, so
a number from one pairing describes neither.

Both bands are label-free: a section's own heading is related to its body by
construction, and a fixed set of questions about penguins, sourdough and the
1966 World Cup final is related to no software corpus at all.

**Pooling the scopes destroyed it.** The first attempt fitted one floor across
the corpus and reported the bands separating by **-0.002** over 170 samples.
Per scope instead:

| scope | chunks | fitted floor |
|---|---|---|
| delroy | 8,320 | 0.678 |
| odysseus | 5,939 | 0.665 |
| urthreads | 119 | 0.643 |
| g2-claude-companion | 179 | 0.579 |

Mean separation **+0.109**. The floor tracks corpus size because it is compared
against a *maximum over a scope's chunks*, and the expected maximum of an
unrelated query grows with how many chunks you take it over. One pooled number
could not describe both -- the same units mistake as 4.2, one layer down.

Nothing regressed: retrieval MRR 0.406, evidence returned 12/12, uncontaminated
answer content @1 6/7, every `loci eval` family at 100%.

## A note on how this document was damaged

Sections were appended by string-replacing an anchor heading. When that heading
stopped existing the replacement silently did nothing, and when it appeared
twice the replacement inserted twice -- leaving this file at 1,158 lines with
every Phase 1.2 through Phase 4 section duplicated, while four sections that were
"written" had never landed at all.

Three silent no-op edits happened this session by the same mechanism: a string
replacement whose target had drifted. None of them raised anything. Appending is
now used instead of anchor replacement.

## Scope groups

Measured on the real corpus (`~/Documents/GitHub`, 11 git repositories plus
three nested one level deeper) rather than a fixture, because the thing this
feature changes -- how many scopes there are and how alike they look -- is the
one property a synthetic bed cannot supply.

Every command ran against a scratch `LOCI_HOME`. The working install at
`~/.loci` was not touched (10 scopes before, 10 scopes after, `updated_at`
unchanged), nothing was written into any scanned repository, and `loci graphs`
was not run, so every scope routes on the structure graph it already had on
disk. Those scratch `LOCI_HOME`s are session-local and will not outlive this
work, so the output blocks below are the artifact -- every one was produced by
re-running against the stage it names, not transcribed from memory. `loci embed` was skipped: routing does not consult embeddings, and the
consequence is only that `loci calibrate` reports `semantic floor 0.57
(default; no embeddings built)` at every stage.

### The four stages

| stage | code | registry |
|---|---|---|
| **S0 baseline** | `507c04c`, the commit before this feature | 14 scopes, no groups, no split |
| **S1 inert** | `HEAD`, marker split suppressed, groups stripped | 14 scopes, no groups |
| **S2 provenance** | `HEAD`, marker split suppressed | 14 scopes, provenance groups |
| **S3 split** | `HEAD`, unmodified | 19 registered, 18 indexed |

S1 exists because S0 changes two things at once -- the feature and 24 commits of
code. Suppressing only `WORKSPACE_MARKERS` isolates them.

`Delroy/client` carries no workspace marker, so it splits only with a
`.loci.json` written into a repository that is not this one. It was left out;
the split measured here is marker-only, and `client/` is unmeasured.

### Per-family rates

```
family              n   S0      S1      S2      S3
deictic + cwd      56/72  100.0%  100.0%  100.0%   87.5%
deictic, no cwd     4     100.0%  100.0%  100.0%  100.0%
unanswerable        3     100.0%  100.0%  100.0%  100.0%
signature          14/18   92.9%   92.9%   92.9%   88.9%
detailed           28/36   92.9%   92.9%   92.9%   88.9%
contended          12      91.7%   91.7%   91.7%   58.3%

evidence_floor            6.151   6.151   6.151   4.317
routable band from        2.144   2.144   2.144   1.997
unroutable band to        5.998   5.998   5.998   6.607
bands                     OVERLAP OVERLAP OVERLAP separated
self-classification        77.1%   77.1%   77.1%   94.0%
```

**Read the S3 column with two corrections.** `deictic + cwd` 100.0% -> 87.5% is
a real regression and the headline of this section: every one of the nine misses
is a new sub-scope, and the mechanism is `CWD_BOOST` landing on both members of
a nesting pair. The `contended` and `detailed` columns are **not comparable as
printed**: both families derive their questions from the index they score, and
the split rewrote most of them. Scored on the subset whose questions survived
unchanged, both are flat -- `detailed` 14/14 -> 14/14 over seven scopes,
`contended` 4/4 -> 4/4 over four terms. `signature` is flat on every scope that
existed before the split. Details below.

S0, S1 and S2 are identical line for line, including the fitted floor to three
decimals. Two things follow.

**The code drift is inert.** Twenty-four commits, including a group-aware
`route`, changed no number on a corpus with no groups. That is what the feature
was supposed to do when nothing asks for it.

**`loci eval` cannot see provenance at all, and that is a property of the
benchmark, not a result.** `eval.run` calls `route(q, index, cwd=...)` and
passes no `eligible`, no `demoted`, no `strict_group`. Group policy reaches
routing only through `groups.confinement`, which `loci route` and `loci ask`
call and `loci eval` does not. Labelling `odysseus` as
`vendor:pewdiepie-archdaemon` and `beacon` as `vendor:unknown` therefore cannot
move a single `loci eval` figure, whatever it does to real questions.

That argument is the load-bearing one, and it is checkable against
`eval.py`/`cli.py` by anyone. It is corroborated by a throwaway wrapper that
re-ran the same families with confinement resolved per question, which
reproduced the table exactly under the default `soft` mode and under
`loci group set me --mode hard` alike -- but a wrapper that silently failed to
*apply* confinement would produce that same null, so the run was instrumented to
count what it actually passed and to re-route each question unconfined as a
control:

```
soft:  route calls 145 | with cwd 72 | eligible 0  | demoted 72 | strict 0
hard:  route calls 145 | with cwd 72 | eligible 64 | demoted 8  | strict 64
both:  calls where confinement was non-empty                    72
       calls where the confined result DIFFERED from unconfined  0
```

Confinement reached `route` on every one of the 72 cwd-bearing calls and changed
nothing. The reason is the shape of the question set: confinement is anchored on
cwd, only `deictic + cwd` supplies a cwd, and that family was already at
ceiling. The wrapper itself is a scratch script and is not in this repository;
the claim rests on the mechanism, with the run as corroboration.

### The split hurt, and here is exactly where

`deictic + cwd` fell 100.0% -> 87.5%. All nine misses are the new sub-scopes:

```
Delroy/glasses:             What is the entry point and how does it start up?
Delroy/glasses:             What are the known risks, gaps or limitations here?
Delroy/glasses:             What configuration or environment variables does it need?
Delroy/n8n-nodes-delroy:    How do I run the tests?
Delroy/n8n-nodes-delroy:    What is the entry point and how does it start up?
Delroy/n8n-nodes-delroy:    What are the known risks, gaps or limitations here?
Delroy/n8n-runtime-adapter: How do I run the tests?
Delroy/n8n-runtime-adapter: What is the entry point and how does it start up?
Delroy/n8n-runtime-adapter: What are the known risks, gaps or limitations here?
```

The denominator moved with the corpus: four new scopes were indexed, and the
family applies the same four evaluation-half questions to every scope, so
56 -> 72. Sixteen pairs were added and nine of them fail.

Restricted to the fourteen scopes that existed before the split, the same family
scores **56/56 = 100.0%** on the split index. *In this family*, the split cost
nothing that already worked. The other three families each say the same thing
where they are licensed to speak, and are silent where the split rewrote their
questions. The decomposition below separates the two for each.

The mechanism is `CWD_BOOST`, and it is structural rather than tunable. A
sub-scope's root is *inside* its parent's root, so both satisfy
`root in cwd_path.parents` and both collect `CWD_BOOST` (4.0). The tie is then
broken by the evidence base, and the parent's is larger -- `Delroy` carries
6,404 routing tokens against `Delroy/glasses`'s 396, a 16x vocabulary. The
decisive margin is much smaller than that, because `SIZE_PRIOR = 0.15` already
discounts the parent: subtracting the shared 4.0 leaves **1.1733 vs 0.9106**,
a factor of 1.29.

```
loci route --cwd ~/Documents/GitHub/Delroy/glasses --explain \
  "What is the entry point and how does it start up?"
-> Delroy, Delroy/glasses
 * Delroy          score=5.1733  matched=3  signals={'cwd': '.../Delroy'}
 * Delroy/glasses  score=4.9106  matched=2  signals={'cwd': '.../Delroy/glasses'}
```

### Does a question about `glasses` reach `delroy-glasses`?

Two different questions, two different answers.

**Named -- yes, decisively.** `ALIAS_BOOST` (6.0) outranks `CWD_BOOST` (4.0),
and the sub-scope owns the alias:

```
loci route --no-cwd --explain "how does the glasses app talk to the runtime?"
-> Delroy/glasses
 * Delroy/glasses  score=7.6255  matched=2  signals={'alias': 'glasses'}
   G2-claude-companion score=1.3585  matched=2  signals=-
   Delroy          score=1.2839  matched=4  signals=-
```

(`Delroy` is third there, not runner-up. An earlier draft of this section cut
the middle line.)

**Deictic, from inside `glasses/` -- mostly no.** Over the four evaluation-half
taxonomy questions applied to the parent and its three indexed sub-scopes:

```
top-1 is the right scope                7/16   43.8%
right scope present in the widened set 14/16   87.5%
```

(A second 7/16 appears later, in the `### Reading` section. The denominators are
different -- this one is the `Delroy` family, parent plus its three indexed
sub-scopes; that one is the four *new* scopes, three of them `Delroy`'s plus
`_site`. They agree by coincidence.)

The parent scores 4/4; each sub-scope scores 1/4. So the split is worth having
when the user names the sub-project and costs accuracy when they do not -- and
"do not" is the case `cwd` exists to serve.

### The `contended` collapse is not comparable, and saying so is the finding

91.7% -> 58.3% looks like the worst number here. It is not a number at all:
`contended_terms` selects terms held by *exactly two* scopes, so partitioning
`Delroy` and `3M1RY33T.github.io` rewrites the question set. Eight of the twelve
terms changed. Four survive into both stages, and scored at both -- the same
licensed-subset standard `detailed` gets below -- all four pass at both:

```
LOCI_HOME=<scratch>/home-s<n> python3 contended_surv.py swift evals kiwix embedding

--- S1 ---                                                      --- S3 ---
PASS swift     owners=brewery, hlep_davay    -> brewery, hlep_davay        (identical)
PASS evals     owners=Delroy, loci           -> loci, Delroy               (identical)
PASS kiwix     owners=tensor-serve, TSRC     -> tensor-serve, TSRC         (identical)
PASS embedding owners=odysseus, tensor-serve -> odysseus, tensor-serve     (identical)

licensed subset: 4 terms -> S1 4/4   S3 4/4
```
 Three of the new terms (`portfolio`,
`statements`, `moderating`) are shared between `3M1RY33T.github.io` and its own
`_site` build directory -- a scope and a copy of itself, which is not a
cross-project question. Four of the five failures are `no_evidence` abstentions
on terms too rare to clear any floor.

Diffing the inputs before the outputs, per ROADMAP rule 4: **the family is
derived from the index it scores and cannot be compared across a change that
alters the index's scope partition.** `deictic + cwd` is the only family here
that is genuinely like-for-like across every scope, because its questions come
from a fixed list *and* it scores routing rather than abstention. The other two
fixed-list families, `deictic no cwd` and `unanswerable`, are equally
comparable but measure only whether the router declines to answer.

### `signature` and `detailed`, decomposed the same way

`deictic + cwd` was the only family decomposed old-vs-new in the first draft of
this section, and "the split cost nothing that already worked" was then written
as though it covered all of them. Both derived families were re-run, split into
the fourteen pre-existing scopes and the four new ones:

```
              S1 (pre)        S3 (pre)        S3 (new)
signature     13/14  92.9%    13/14  92.9%    3/4   75.0%
detailed      26/28  92.9%    24/28  85.7%    8/8  100.0%
```

An intermediate draft of this section reported that `detailed` line as a
regression. **It is not one, and reporting it as one broke this section's own
rule two paragraphs up.** Both halves of that -- whether the comparison is
licensed at all, and what the misses actually are -- were then measured.

**`signature` is comparable; `detailed` is not.** The two families draw
different terms: `signature` uses `signature_terms(k=2)`, `detailed` uses
`pool[4:9]` -- ranks 5 to 9 (eval.py:245-246). Only the top-2 were checked in
that draft, which licenses `signature` and says nothing about `detailed`. Both
ranks, all fourteen pre-existing scopes, S1 against S3:

```
python3 termstab.py <stage>.json ; python3 termdiff.py s1.json s3.json   (scratch harness)

identical top-2   : 14/14
identical rank 5-9: 4/14

CHANGED 3m1ry33t
  5-9  S1 ['platforms', 'interesting', 'frontend', 'blog', 'star']
       S3 ['href', 'platforms', 'frontend', 'star', 'span']
CHANGED 3m1ry33t-github-io
  5-9  S1 ['website', 'workers', 'worker', 'posts', 'statements']
       S3 ['posts', 'portfolio', 'website', 'tensor', 'worker']
CHANGED beacon
  5-9  S1 ['amount', 'alphabet', 'metrics', 'bucket', 'validating']
       S3 ['refill', 'amount', 'alphabet', 'metrics', 'bucket']
CHANGED brewery
  5-9  S1 ['core', 'sources', 'homebrew', 'formula', 'catalog']
       S3 ['sources', 'core', 'homebrew', 'formula', 'catalog']
CHANGED delroy
  5-9  S1 ['jupyter', 'vendor', 'venv', 'share', 'harnessbench']
       S3 ['jupyter', 'venv', 'vendor', 'harnessbench', 'computer']
CHANGED g2-claude-companion
  5-9  S1 ['tsconfig', 'headless', 'hook', 'permissions', 'injector']
       S3 ['headless', 'glasses', 'hook', 'protocol', 'injector']
CHANGED myblog
  5-9  S1 ['relationships', 'doing', 'delroy', 'says', 'modifying']
       S3 ['relationships', 'doing', 'says', 'modifying', 'knowledge']
CHANGED tensor-serve
  5-9  S1 ['reranker', 'bash', 'reranking', 'please', 'faiss']
       S3 ['reranker', 'bash', 'faiss', 'reranking', 'please']
CHANGED urthreads
  5-9  S1 ['likes', 'worker', 'admin', 'moderation', 'setup']
       S3 ['admin', 'likes', 'moderation', 'worker', 'readline']
CHANGED zim-compress
  5-9  S1 ['article', 'temporary', 'mmap', 'archives', 'image']
       S3 ['article', 'temporary', 'mmap', 'archives', 'strip']
```

Ten of fourteen pre-existing scopes drew different rank-5-9 terms once four
scopes were added, so `detailed` asks ten of them different questions at S3 than
at S1. The rule stands as written: **a family derived from the index it scores
cannot be compared across a change that alters the index's scope partition.**
`deictic + cwd` remains the only like-for-like family; `signature` joins it only
because its top-2 terms were measured not to move.

**And every miss on both sides is a generator artifact, not a routing loss.**

```
S1 pre-existing misses (2):
  MyBlog -> Delroy: when relationships is created, how do doing and delroy affect the says that gets written?
  MyBlog -> Delroy: what connects relationships to doing, and where do delroy and says come into that?

S3 pre-existing misses (4), as `loci eval --misses` prints them:
  3M1RY33T.github.io -> 3M1RY33T.github.io/_site: when posts is created, how do portfolio and website affect the tensor that gets written?
  3M1RY33T.github.io -> 3M1RY33T.github.io/_site: what connects posts to portfolio, and where do website and tensor come into that?
  G2-claude-companion -> Delroy/glasses: when headless is created, how do glasses and hook affect the protocol that gets written?
  G2-claude-companion -> Delroy/glasses: what connects headless to glasses, and where do hook and protocol come into that?
```

*The two `_site` misses are unsatisfiable.* Parent and `_site` have set-equal
postings, so `signature_terms` draws the same pool for both and the generator
emits **byte-identical question strings** with two different gold labels:

```
python3 detailed_qs.py                                        (scratch harness)

[0] parent: when posts is created, how do portfolio and website affect the tensor that gets written?
[0] _site : when posts is created, how do portfolio and website affect the tensor that gets written?
[0] IDENTICAL STRING: True
[1] parent: what connects posts to portfolio, and where do website and tensor come into that?
[1] _site : what connects posts to portfolio, and where do website and tensor come into that?
[1] IDENTICAL STRING: True

pool[4:9] parent: ['posts', 'portfolio', 'website', 'tensor', 'worker']
pool[4:9] _site : ['posts', 'portfolio', 'website', 'tensor', 'worker']

scope groups sharing an identical detailed question set: [['3M1RY33T.github.io', '3M1RY33T.github.io/_site']]
```

No router can score better than 50% on that pair. The parent lost both and
`_site` scored 8/8, which is what an unsatisfiable pair looks like from the
outside.

*The other four are another project's alias.* `signature_terms` bans a scope's
own name and aliases (built at eval.py:132-133) and nothing else, so a term that happens
to be a *different* project's name is eligible to be drawn as "distinctive
vocabulary". The question then names that project outright, and `ALIAS_BOOST`
(6.0) does exactly what it is for:

```
S3, G2-claude-companion's own generated question:
  "when headless is created, how do glasses and hook affect the protocol ...?"
 * Delroy/glasses       score=6.6574  matched=1  signals={'alias': 'glasses'}
   G2-claude-companion  score=1.9490  matched=4  signals=-

S1, MyBlog's own generated question:
  "when relationships is created, how do doing and delroy affect the says ...?"
 * Delroy               score=7.1298  matched=6  signals={'alias': 'delroy'}
 * MyBlog               score=3.2725  matched=4  signals=-
```

`glasses` is an alias only because the split created `Delroy/glasses`; `delroy`
was one all along. So the split did not lose `G2-claude-companion` -- it gave a
sub-project a name, and the generator then put that name inside another scope's
question. Symmetrically, the two S1 `MyBlog` misses "fixed" by the split were
never routing failures either: `delroy` simply dropped out of MyBlog's rank-5-9
pool.

**`detailed` does have a licensed subset, and it is flat.** "Not comparable"
was applied to the whole family when it is only true of half of it -- the same
escape the `contended` paragraph above refuses when it scores the four terms
that survive both stages. Seven of the fourteen pre-existing scopes ask a
*routing-identical* `detailed` question at S1 and S3:

```
python3 licensed_render.py s1-det.json s3-det.json            (scratch harness)

scope                  same string  S1    S3    the four terms drawn (S1 -> S3)
brewery                reordered    2/2   2/2   core,sources,homebrew,formula  ->  sources,core,homebrew,formula
hlep-davay             yes          2/2   2/2   firestore,boring,chttp,absl
loci                   yes          2/2   2/2   backends,deictic,doctor,graphify
odysseus               yes          2/2   2/2   docx,gallery,hwfit,research
tsrc                   yes          2/2   2/2   vite,function,tensor,preload
urthreads              reordered    2/2   2/2   likes,worker,admin,moderation  ->  admin,likes,moderation,worker
zim-compress           yes          2/2   2/2   article,temporary,mmap,archives

licensed subset: 7 scopes, 14 items -> S1 14/14   S3 14/14
```

`S1`/`S3` are that scope's two `DETAILED_TEMPLATES` items scored at each stage.

Four of the seven kept their rank-5-9 terms outright. `zim-compress` differs
only at rank 9 and the template formats `a,b,c,d` from `terms[0:4]`
(eval.py:250), so `terms[4]` is never used and the string is byte-identical.
`brewery` and `urthreads` draw the same four terms in a different order; `route`
scores a token multiset, and the one order-sensitive test is `_alias_hit`'s
contiguous run, which none of this corpus's multi-token aliases (`g2 claude
companion`, `hlep davay`, `tensor serve`, `zim compress`, and three sub-scope
names) can match inside those four tokens. That licence is load-bearing rather
than a formality -- after stopwording, all four terms land contiguous, so a two-
or three-token alias genuinely could match a run that a swap destroys. None
does, and both stages were scored directly rather than argued: 2/2 and 2/2 for
each of the seven.

**Same question, changed index, 14/14 -> 14/14.** That is the like-for-like the
split was to be scored on, and it says the split cost `detailed` nothing where
`detailed` can speak. Every one of the six misses falls on one of the other
seven scopes -- the ones whose questions the split rewrote.

**Verdict.** In `deictic + cwd` the split cost nothing that already worked
(56/56). In `signature` it cost nothing on pre-existing scopes (13/14 both
stages, the same single miss). In `detailed` it cost nothing on the seven scopes
whose questions did not change (14/14 both stages); on the other seven there is
no measurement, because the questions are not the same questions.

**The generator defect is not a footnote, and it did move numbers.** An earlier
draft of this paragraph said it "is not the reason any routing number moved".
That is false, and the check that would have caught it was not run. Recording
`signature`'s misses in full:

```
LOCI_HOME=<scratch>/home-s<n> python3 sigmiss.py              (scratch harness)

S1 signature (13/14), the one miss:
  3M1RY33T.github.io -> urthreads: what happens to dashboard during urthreads processing?

S3 signature (16/18), both misses:
  3M1RY33T.github.io       -> urthreads: what happens to dashboard during urthreads processing?
  3M1RY33T.github.io/_site -> urthreads: what happens to dashboard during urthreads processing?

identical signature questions across scopes: [['3M1RY33T.github.io', '3M1RY33T.github.io/_site']]
```

Both are the same two defects again. The question contains `urthreads`, which is
`urthreads`'s own alias:

```
 * urthreads              score=7.7517  matched=2  signals={'alias': 'urthreads'}
   3M1RY33T.github.io/_site score=2.4942  matched=2  signals=-
   3M1RY33T.github.io     score=2.4324  matched=2  signals=-
```

Subtract the 6.0 and `urthreads` scores 1.7517 against the parent's 2.4324 --
the alias is decisive there too. And the second S3 miss exists only because
parent and `_site` collide onto one string, the `detailed` collision repeated.
So `signature`'s pre-existing figure is genuinely flat, but its single miss was
never a routing failure at either stage, and its new-scope miss is a collision.

The leak is wider than `signature_terms`, which at least bans the scope's own
name and aliases (eval.py:132-133): `contended_terms` (eval.py:150-168) has no
ban list at all. Only `deictic + cwd`, `deictic no cwd` and `unanswerable` draw
from fixed lists and are immune. The fix -- ban every *registered* alias, not
only the scope's own -- is out of scope here, and its consequences for the
router are a defect in their own right, recorded below.

### The floor moved, and it explains none of it

`evidence_floor` fell 6.151 -> 4.317 and the bands stopped overlapping
(self-classification 77.1% -> 94.0%). Not because the parent shrank -- `Delroy`
went from 6,469 routing tokens to 6,404, which is nothing. Calibration fits on
its own generated half, and that half grew from 146 routable samples to 175 as
the scope count went 14 -> 18, which also moves every IDF in the corpus. Which
of the two drove the floor was not isolated.

Per ROADMAP rule 1, the threshold was not re-fitted to recover anything. The
opposite check was run: the split corpus was re-evaluated with S1's floor forced
back to 6.151, and every family reported the identical rate. The regression is
the corpus shape, not the constant.

### `loci group set vendor:... --mode explicit` does not do what the plan said

The plan's Step 3 called for `loci group set vendor:pewdiepie-archdaemon --mode
explicit` to take a foreign repository out of the competition. Run on the split
corpus, it does not:

```
loci route --cwd ~/Documents/GitHub/loci --explain \
  "how is mammoth handled together with cookbook?"
-> loci, odysseus
 * loci      score=4.1147  matched=0  signals={'cwd': '.../loci'}
 * odysseus  score=1.2339  matched=4  signals=-
```

A mode governs what a group does to questions asked from *inside* it, and
`explicit` is the mode that confines least. `loci doctor` says so unprompted --
"`loci group set me --mode hard` ... keeps them out of every question asked from
inside it" -- and the doctor is right. Worth noting separately: `loci` wins that
line with **zero matched tokens**, purely on `CWD_BOOST`, while `odysseus`
matches four high-evidence terms and is still returned second.

### cwd-anchored `hard` mode barely fires -- confirmed, quantified

`out_of_group` requires the *corpus-wide* winner to sit outside the group. cwd
adds 4.0 to a scope that is in the group by construction, and measured evidence
bases occupy 0.1-1.5. With `me` set to `hard` on the split corpus, over six
in-group anchors x six probes built from the out-of-group scopes' own most
discriminative vocabulary:

```
out_of_group fired                                     12/36
  of which, probes that NAME the outside project       12/12
  probes carrying only its distinctive vocabulary       0/24
same probes with --group me and no cwd                  6/6
```

Every firing came from `ALIAS_BOOST` (6.0) beating `CWD_BOOST` (4.0). The
twenty-four evidence-only probes -- questions whose true answer is a vendor
repository -- all routed to whatever scope the user happened to be standing in.
Hard mode is `--group`-driven and alias-driven in practice, exactly as the
review claimed. It is not useless; it is not cwd-anchored protection.

### Three defects the split surfaced

**`Delroy/extension` is registered, stored, and unreachable.** Nineteen scopes
register; eighteen index. `extension/` contains no markdown of any kind, so its
343 episode chunks are all `kind="docstring"`, and `ROUTING_KINDS` excludes
docstrings. The guard meant to catch this is

```python
if not nodes and len(prose_vocab) < DOCSTRING_FALLBACK_VOCAB:   # index.py:164
```

with `DOCSTRING_FALLBACK_VOCAB = 0` (index.py:53), so `0 < 0` is false and the
fallback never runs. Folding the docstrings in by hand yields 1,724 vocabulary
tokens -- the scope is not empty, it is discarded. The constant is deliberate
and test-pinned (`test_docstring_routing_fallback_stays_disabled`: warm it costs
14 points of top-1, cold it halves accuracy), so the defect is not the value.
The defect is that the comment above it promises the fallback keeps a
prose-only scope from having its "chunks sit indexed and unreachable", and that
is precisely what happens: the scope is absent from the routing index, no
question can reach it, and `loci ask --scope delroy-extension` answers `error:
unknown scope`. Before the split those 343 chunks were reachable through
`Delroy`. Sub-projects with no README and no graph are the shape this bites, and
splitting a monorepo manufactures them.

**Three mitigations, which change how bad that is.** The loss is not silent:
`loci index` prints `- Delroy/extension  nothing indexable, skipped`, and
`loci doctor` reports the registered-but-unindexed scope by design. And it is
not permanent -- the scope has no structure graph, and building one gives it
`nodes`, which populates `counts` and makes it routable without touching the
docstring fallback at all. `index.py:50` says so in as many words: "The fix for
a cold scope is `loci graphs`, not this." **`loci graphs` is exactly what this
measurement's constraints forbade** (it writes `graphify-out/` into every
repository it scans, and the corpus is read-only here). So `Delroy/extension`
is unreachable *in this configuration*, not unreachable in principle. What
remains a real defect is the comment's promise, which describes a fallback that
cannot fire.

**`_site` became a scope.** `3M1RY33T.github.io/_site` is a Jekyll build
output. It carries a `package.json` and its own `graphify-out/`, it is not in
`SKIP_DIRS`, and it does not start with a dot, so the depth-1 marker rule
registered it. Its routing vocabulary is not merely the same size as its
parent's, it is the same vocabulary:

```
LOCI_HOME=<scratch>/home-s3 python3 -c 'from loci.index import load_index ...'

parent vocab 221  _site vocab 221  shared 221  jaccard 1.000
identical vocabulary: True
```

It contributed three of the twelve `contended` terms
(`portfolio`, `statements`, `moderating`) and beat its own parent on both
`detailed` items. Generated directories need excluding by name the way
`node_modules` already is.

**The split minted `glasses` as a corpus-wide alias, and it now outranks real
evidence.** `_aliases_for` (scopes.py:33-35) promotes `root.name` to an alias,
so splitting `Delroy` gave `Delroy/glasses` the alias `glasses` -- a common
English noun -- worth `ALIAS_BOOST` 6.0, matched by `_alias_hit` as a contiguous
token run anywhere in a question. This is not confined to the benchmark's
generated questions. A hand-written question about the *other* project that
works on glasses:

```
loci route --no-cwd --explain "how does the companion daemon drive the glasses display?"

S1 (before the split)
-> G2-claude-companion, Delroy
 * G2-claude-companion  score=3.0632  matched=5  signals=-
      [('daemon', 11.738), ('glasses', 9.619), ('drive', 6.934), ...]
 * Delroy               score=1.1239  matched=5  signals=-

S3 (after the split)
-> Delroy/glasses
 * Delroy/glasses       score=7.2453  matched=2  signals={'alias': 'glasses'}
      [('glasses', 10.476), ('display', 3.282)]
   G2-claude-companion  score=3.2241  matched=5  signals=-
```

Five matched tokens and the highest evidence in the corpus lose to two matched
tokens and a name collision. (Only the S1 `->` line lists two scopes: the S1
cutoff is 3.0632 x 0.85 = 2.6037 and `Delroy` scores 1.1239, so it enters solely
through the concentrated-token merge at router.py:466-471. The S3 line lists
one. The ranking is what matters here either way.)

One probe does not establish a rule, so seven more hand-written
`G2-claude-companion` questions containing the word were scored at both stages,
and then re-scored against the **unchanged S3 index with `ALIAS_BOOST = 0.0`** --
a counterfactual, so that "the alias did it" is demonstrated rather than
inferred from the margin:

```
LOCI_HOME=<scratch>/home-s<n> python3 glasses_probe.py           (scratch harness)
LOCI_HOME=<scratch>/home-s3  python3 glasses_counterfactual.py   (scratch harness)

#  question (abridged)              S1 top-1     S3 top-1        S3, alias=0     G2    glasses
1  ...daemon drive the glasses      G2 (3.0632)  glasses(7.2453) G2-claude-comp  3.2241  1.2453
   display?
2  what protocol does the           G2 (2.1680)  glasses(6.9777) G2-claude-comp  2.2183  0.9777
   companion use to talk...?
3  where is the glasses pairing     G2 (1.5066)  glasses(7.9081) Delroy/glasses  1.4761  1.9081
   handshake implemented?
4  which spikes cover glasses       G2 (2.3724)  glasses(7.6486) G2-claude-comp  2.4286  1.6486
   connectivity?
5  how do the injector and hook     G2 (2.8674)  glasses(7.4584) G2-claude-comp  2.9466  1.4584
   set up glasses permissions?
6  does the headless daemon need    ABSTAIN      glasses(7.4114) ABSTAIN         2.9282  1.4114
   glasses hardware...?
7  what happens when glasses        G2 (1.3568)  glasses(7.4285) Delroy/glasses  1.3711  1.4285
   firmware reports...?
8  how is the glasses session       G2 (2.1321)  glasses(7.1946) G2-claude-comp  2.1578  1.1946
   resumed...?

reached G2-claude-companion:  S1 7/8   S3 0/8
alias-caused captures:        6/8      (probes 3 and 7 win on evidence)
```

**Six of the eight are caused by the alias.** Removing the 6.0 reverts five to
`G2-claude-companion` and returns probe 6 to its abstention; probes 3 and 7 still
land on `Delroy/glasses` without it, on their own evidence (1.9081 and 1.4285
against G2's 1.4761 and 1.3711). An earlier draft said `Delroy/glasses` "wins
each on the alias alone", which the last two columns disprove -- they are the
columns that draft omitted.

*And on those two the gold label is contestable*, which a reader who opens the
corpus will discover, so it is said here first. `Delroy/glasses` is itself an
Even Realities **G2** smart-glasses app ("# Delroy Lens / An Even Realities
**G2** smart-glasses companion", README line 1-3) and it contains
`src/pairing/pair.ts`. So the two projects are not merely name-colliding on
`glasses`, they collide on `G2` as well, and "where is the glasses pairing
handshake implemented?" arguably belongs to the sub-scope. The split did not
invent that evidence: at S1 the runner-up on that question was `Delroy` itself,
on `pairing` (4.073), and the split moved it to the sub-project that owns it.
Probe 7 is stronger still -- `firmware` appears in 31 files of `Delroy/glasses`
and none of `G2-claude-companion` (it carries no routing weight either way; the
token is not in the routing index at all).

So the bounded claim, which is the defensible one: **six of eight hand-written
questions about `G2-claude-companion` were captured by `Delroy/glasses` purely
because the split minted `glasses` as a 6.0-point alias.** The remaining two
were arguably routed correctly, and the split is what made them routable at all.

Probe 6 is the worst case in the set and its cause is proven rather than
inferred: with `ALIAS_BOOST` at 0 it abstains again. `route` sets
`forced = _signalled(top_d)` (router.py:424), and `forced` suppresses the deixis
guard -- so an alias hit does not merely outscore the right answer, it converts
a correct abstention into a confident answer from the wrong project.
router.py:432-438 names the same failure shape under hard groups, in its own
comment.

(Both citations were wrong when first written and are corrected here. The line
range read 346-349, which is a blank line and a comment; the assignment was at
350. The quote was `forced = bool(detail[top]["signals"])`, silently dropping
the `top and` guard the source carried. The code has since moved behind the
`_signalled` helper -- one predicate for two call sites, no change in
behaviour -- and both citations name that final state.)
The same mechanism produced two of the six `detailed` misses; subtract the 6.0
there and `G2-claude-companion` wins 1.9490 to 0.6574.

The generator's missing ban list made this visible; it is not the cause. The
cause is that a sub-project's directory name becomes a corpus-wide 6.0-point
alias with no check on whether it is a distinctive name or an ordinary word.
Splitting a monorepo is exactly what mints them, four at a time.

The chunk-partition invariant does hold. Parent against each sub-scope, and all
six sub-scope pairs:

```
LOCI_HOME=<scratch>/home-s3 python3 -c 'from loci.index import load_episodes ...'

delroy & delroy-extension:            0 shared text(s) (sub=343)
delroy & delroy-glasses:              0 shared text(s) (sub=838)
delroy & delroy-n8n-nodes-delroy:     0 shared text(s) (sub=13)
delroy & delroy-n8n-runtime-adapter:  0 shared text(s) (sub=5)

delroy-extension & delroy-glasses:                0
delroy-extension & delroy-n8n-nodes-delroy:       0
delroy-extension & delroy-n8n-runtime-adapter:    0
delroy-glasses & delroy-n8n-nodes-delroy:         0
delroy-glasses & delroy-n8n-runtime-adapter:      0
delroy-n8n-nodes-delroy & delroy-n8n-runtime-adapter: 0

texts in S1 delroy but not in S3 delroy:            1046
  of those, recovered by some sub-scope:            1046
  NOT recovered anywhere in the delroy family:         0
```

Pairwise disjointness makes "recovered by exactly one sub-scope" true as
written. Nothing was double-counted and nothing left the store -- 343 of them
merely left the reachable set.

### What this measurement does not cover

- `Delroy/client`, which needs a `.loci.json` in a repository this task was not
  authorised to write to.
- What the split did to `detailed` and `contended` **on the scopes whose
  questions it rewrote**. Both families regenerate their questions from the
  index. Where the question survived unchanged both were scored -- `detailed`
  14/14 -> 14/14 on seven scopes, `contended` 4/4 -> 4/4 on four terms -- and
  where it did not, there is no before/after number and none was invented.
- Any effect of provenance labels on real questions, in a metric. `loci eval`
  is structurally blind to group policy; the `--group me` probes above are a
  spot check, not a benchmark.
- Semantic ranking. No embeddings were built, so `semantic_floor` is the 0.57
  default at every stage and `loci ask`'s episode ranking was not exercised.
- Missing structure graphs were left missing rather than built, because
  `loci graphs` writes `graphify-out/` into every repository it scans and the
  corpus is read-only here. `MyBlog`, `tensor-serve`,
  `Delroy/n8n-nodes-delroy` and `Delroy/n8n-runtime-adapter` route on prose
  alone; `Delroy/extension` routes on nothing. Whether a graph would restore
  `Delroy/extension` to the routable set -- `index.py:50` says it should -- is
  therefore untested here.

### Reading

**The split cost `deictic + cwd` 100.0% -> 87.5%, and that is the finding.**
Nine of seventy-two items fail, all of them new sub-scopes, and the family is
one of only three families whose questions come from a fixed list rather than
from the index they score, and the only one of those three that scores routing
rather than abstention. Every scope that existed before the split still routes at 100%, which
makes this additive damage rather than a corruption of what worked -- but it is
damage: standing inside a `Delroy` sub-project reaches it 1 time in 4, and the
fifth new scope, `Delroy/extension`, never entered the routing index at all and
so is absent from the 72-item denominator rather than scored badly in it.

Against that, the split does what it was built to do -- a named sub-project now
reaches its own scope instead of a 6,469-token parent that would have won on
size. Both things are true, and the second does not pay for the first: naming a
project is the case that already worked, and `cwd` exists for the case where the
user names nothing.

It manufactures the failure the README's scale table warns about, in the one
place the table predicted: nested scopes whose cwd signal is ambiguous by
construction, because a sub-scope's root is inside its parent's.

`CWD_BOOST` is the constant at fault, and its *magnitude* provably cannot fix
this: the identical 4.0 is added to both members of a nesting pair, so it
cancels out of their comparison at every value. The tie therefore falls to the
size-discounted evidence base -- **not to parenthood**, and the corpus contains
a clean demonstration of the difference. Across all four new scopes, sixteen
`deictic + cwd` items:

```
LOCI_HOME=<scratch>/home-s3 python3 newscopes.py              (scratch harness)

3M1RY33T.github.io/_site     4/4
Delroy/glasses               1/4
Delroy/n8n-nodes-delroy      1/4
Delroy/n8n-runtime-adapter   1/4
new scopes overall: 7/16 = 43.8%
```

The child wins every time in the one pair whose vocabularies are set-equal, and
loses three times in four where the parent is 16x larger. `_site` is not an
exception to the thesis, it is the control for it: with nothing to separate the
evidence bases, the pair falls to the 0.15 recency tiebreak.

```
loci route --cwd .../3M1RY33T.github.io/_site --explain "How do I run the tests?"
 * 3M1RY33T.github.io/_site score=4.7461  matched=1  signals={'cwd': '.../_site'}
 * 3M1RY33T.github.io       score=4.6843  matched=1  signals={'cwd': '...'}
```

**And that tiebreak is not measuring recency.** A sub-scope directory has no
`.git` of its own -- verified for all five here -- so `_git_updated_at` returns
`""` (scopes.py:38-40) and `make_scope` falls back to `datetime.now()`
(scopes.py:133). `updated_at` is not in `PRESERVED_FIELDS` (scopes.py:232), so
every scan re-stamps it. The five sub-scopes are therefore the five freshest
scopes in the registry *by construction*, permanently:

```
myblog                       2026-08-26T19:12:36-04:00
delroy                       2026-08-27T00:45:18-04:00
loci                         2026-08-27T20:42:53-04:00
delroy-extension             2026-08-28T00:44:56.415189+00:00   <- scan time
delroy-glasses               2026-08-28T00:44:56.415343+00:00   <- scan time
delroy-n8n-nodes-delroy      2026-08-28T00:44:56.415385+00:00   <- scan time
delroy-n8n-runtime-adapter   2026-08-28T00:44:56.415418+00:00   <- scan time
3m1ry33t-github-io-site      2026-08-28T00:44:56.415470+00:00   <- scan time
```

`_site` would win that tiebreak even if its parent had been committed to five
minutes earlier. The three Delroy sub-scopes lose *despite* the same free
freshness, not for want of it: 5.1733 - 4.9106 = 0.2627 is a gap between
**final scores**, and `Delroy/glasses` is scan-stamped while `Delroy` is
git-stamped, so the child has already been paid its recency advantage and is
still 0.2627 behind. The 0.15 is spent, not available. So the nesting pair is decided by an evidence base when the gap is wide
enough and by a timestamp that means nothing when it is not -- never by
containment, which is the one fact that actually answers the question.

The boost is also asymmetric, which is the same defect from the other side.
Standing in the *parent* gives the child nothing, because `root in
cwd_path.parents` is false in that direction -- so the parent scores 4.6843
alone and wins 4/4 as well. Both members of the pair win when you stand in them,
for reasons that have nothing to do with which one you are in.

`SIZE_PRIOR` is the one knob that could in principle flip a parent/child tie,
and the Phase 3 sweep already closed both directions (router.py:88-90): at 0.30
top-1 falls 85.7% -> 71.4%, at 0.45 it is "catastrophic", and at 0.05 cwd
routing itself falls 100% -> 94%/89%. 0.15 was optimal on both corpus shapes
tested. Tuning is not the road.

The fix is available inside the current design and is not a new mechanism.
`scope_for_cwd` already resolves a working directory to the *deepest* scope
containing it; `route` simply does not use it, and instead tests
`root in cwd_path.parents` independently per scope. Giving `CWD_BOOST` to the
deepest containing scope only -- the answer `scope_for_cwd` already computes --
makes the nesting pair decidable without touching any constant. That is a
change to what the signal means, not to its size. Recorded, not fixed, and not
fitted around.
