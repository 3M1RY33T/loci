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
hidden.

- One machine, ten scopes, one author. The author had read parts of these
  corpora before writing family B, which is why `contamination` is a field and
  why uncontaminated results are always reported separately.
- Family A is unbiased by construction; families B and C are not, and their
  sample sizes (7, 11) are small enough that single items move percentages a lot.
- `negative` at 66.7% is 4 of 6 — do not read that as a rate.
