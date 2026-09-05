# Changelog

## 0.3.0 — 2026-09-05

**No reindex.** `INDEX_VERSION` stays at 2. The shortlist below is computed at
query time from postings already on disk, and nothing changed about what a token
is, so upgrading is `pipx upgrade loci-mem` and nothing else.

### An abstention's candidates are a shortlist, not the registry

The candidate line printed `ranked` — every eligible scope, in score order.
On a fourteen-scope corpus that is fourteen names with nothing to prefer between
them, which is the registry rather than a shortlist:

```
ABSTAINED - not enough of the question exists in any project.
  candidates: Delroy, odysseus, tensor-serve, loci, MyBlog, 3M1RY33T.github.io,
              urthreads, beacon, hlep_davay, G2-claude-companion, TSRC, brewery,
              3M1RY33T, zim-compress
```

A scope now earns its place by holding a term **at most half the corpus holds**,
and the line says which terms those are:

```
ABSTAINED - not enough of the question exists in any project.
  candidates: Delroy (handled, caching), odysseus (handled, caching), tensor-serve (caching)
```

Holding a query token is not itself a claim. `change`, `handled` and `work` sit
in nearly every project, so matching one is a coincidence — filtering on
"matched anything" keeps 8.4 scopes of 14 and manufactures plausible candidates
for questions that should return none. Measured over 24 gold-bearing abstentions
and 25 questions that should abstain, via the local eval harness (see
**Status** in the README for why the corpus is not published):

| rule | gold recall | \|cand\| gold | \|cand\| should-abstain |
|---|---|---|---|
| `ranked`, i.e. before | 100.0% | 14.0 | 14.0 |
| holds any matched token | 95.8% | 8.4 | 6.1 |
| **held by ≤ S/2 scopes** | **91.7%** | **6.5** | **4.6** |
| held by ≤ S/3 scopes | 66.7% | 4.5 | 2.7 |
| held by ≤ 2 scopes | 20.8% | 2.1 | 1.0 |

What picks 0.5 is the cliff under it, not a flat optimum above it: halving the
corpus costs 4.1 points of recall for two scopes off the list, the next
tightening costs 25 more, and the one after that gives up three quarters of the
answers. It is a share rather than a count so it scales with the corpus.

When nothing clears the bar there is no shortlist to print, and saying so is the
answer — the subject is indexed nowhere, and `--scope` would only relocate the
guess:

```
ABSTAINED - not enough of the question exists in any project.
  no project holds a distinctive term from this question -- `loci doctor` shows what is not indexed.
```

`route --json` gains `candidates` and a per-scope `claims` list. The MCP `ask`
tool returns the same rendered text it always did, so an agent client picks this
up with no change.

### A question set big enough to decide something

The hand-authored eval set went from 31 questions to 61, covering 13 of 14
scopes rather than 10, and a new `clarify` harness reports what an abstention's
shortlist is worth and where the remaining error actually lives. Both stay out
of the repository: they name six private or local-only projects at `file:line`.

| family | n | top-1 | gold-coverage | | contamination | n | top-1 |
|---|---|---|---|---|---|---|---|
| behavior | 22 | 27.3% | 63.6% | | none (from code) | 25 | 32.0% |
| confusable | 7 | 42.9% | 76.2% | | prose (from docs) | 14 | 57.1% |
| cross | 5 | 40.0% | 70.0% | | | | |
| enumerative | 5 | 100.0% | 100.0% | | | | |
| negative | 12 | 91.7% | 91.7% | | | | |

Questions derived from code bodies route at roughly half the rate of ones
derived from indexed prose. The set can now say that on 25 items against 14
rather than 24 against 7, and six questions were discarded or reworded during
authoring because the gold scope indexed none of their tokens — they measured
the corpus, not the router.

It immediately falsified a number in this release. The first table above was
measured on 10 abstentions and reported 0.5 as costing *no* recall; on 2.4× the
questions it costs 4.1 points. The shape of the curve survived, the number did
not, which is why the table now names its sample size and ships the command that
rebuilds it.

### Rejected on measurement

**Ordering the shortlist by evidence.** It reads as the more principled choice
and measures worse: the gold scope comes first 41.7% of the time in score order
and 12.5% in evidence order. The score carries the cwd and alias boosts and the
size prior; raw evidence favours whichever scope is biggest.

**Generating a question that splits the shortlist.** The postings are an
object × attribute matrix, so information gain over them is available and cheap —
and the terms it selects are properties of the corpus, not of the question.
Across the twelve abstentions where the gold scope is present but not first, the
top splitters were the same handful of terms (`dart`, `brewery`, `minigames`)
regardless of whether the question was about rate limiting, chunk overlap or a
blog post. Not one was a term the asker could have answered from their own
subject.

## 0.2.0 — 2026-08-30

**Upgrading requires one command.** `INDEX_VERSION` is now 2 and a v0.1.0 index
is rejected on load with the reason. Run:

```bash
loci index && loci calibrate
```

The tokenizer changed what counts as a word, so the old vocabulary is genuinely
incompatible rather than merely older. `loci update` does the whole chain.

### Enumerative questions return the whole set

`which of my projects use Cloudflare workers or D1?` returned one owner of two.
Every gate in the router asks *is there enough evidence for ONE scope*, and a
question about something several projects share splits its evidence across them
by construction — so the more projects genuinely shared a term, the less likely
all of them came back. The README's own quickstart returned both owners only by
luck.

Enumeration is now detected grammatically, like deixis — a closed class of
markers, no threshold — and switches selection from "within 0.85 of the top
scope" to "every scope that clears the floor on its own". The frame nouns
(`projects`, `repos`, …) are stripped from the query, because they are
scaffolding rather than vocabulary.

| | before | after |
|---|---|---|
| confusable top-1 | 50.0% | **100.0%** |
| confusable gold-coverage | 41.7% | **66.7%** |
| cross gold-coverage | 66.7% | **83.3%** |
| negative (abstains correctly) | 66.7% | **88.9%** |

`behavior` is byte-identical, and taxonomy holds at 100% with cwd / 100%
abstained without. A new `enumerative` eval family scores 100% on all three
metrics. `SET_FLOOR_RATIO` is a ratio of the *calibrated* floor, so it inherits
whatever `loci calibrate` fitted; it is flat across 0.6–0.8 and ships at 0.8.

Rejected on measurement: folding singular/plural at query time. It looked
obviously right — `urthreads` holds `worker` 158 times and `workers` twice — and
measured worse (behavior gold-coverage 42.9% → 28.6%, negatives 66.7% → 55.6%).

### The tokenizer no longer deletes `D1`

`[^\W\d_]+` treated every digit as a separator *and* discarded it, so `D1`
became `d` and died on the length floor. Across 53 markdown files in three
repositories: 258 distinct alphanumeric terms deleted outright over 1,246
occurrences — `s3` (102), `n8n` (70), `d1` (64) — plus `base64` → `base`,
`sha256` → `sha`.

`d1` appears in this project's README as a routing example. It could never have
worked.

It also killed aliases: a scope named `3M1RY33T` tokenized its only alias to
`[]`, so `ALIAS_BOOST` — the strongest signal in the router — could never fire
for it, and naming the project outright abstained.

The new output is a **superset**: `base64` yields both `base64` and `base`, so
no query that matched before can stop matching. Bare numbers are dropped, and a
deliberately narrow git-hash guard drops runs that are ≥6 characters, entirely
hexadecimal, and mix digits with letters — narrow because dropping the digit
requirement would delete `decade`, `facade` and `deface`.

### A tokenizer change no longer leaves the index stale

Shipping the above, `loci index` reported 12 of 14 scopes "unchanged, reused
without re-parsing" and kept the old vocabulary — the fingerprint is a content
signature and no file's mtime had moved. The routing index and the tokenizer
disagreed about what a word is, and nothing said so.

`fingerprint()` is now seeded with `text.rules_signature()`, so any future
change to the tokenizing rules invalidates the cache by construction rather than
when somebody remembers to bump a constant.

### Fixed: a cold multi-scope `ask` could crash the process

Found by running the new feature's own example. `ask` fans out across the
selected scopes in a thread pool, and every lazy initializer in the episode
backend — the embedding model, the vector cache, the reranker — was an
unguarded `if X is None: X = build()`. Two threads building a
sentence-transformer concurrently do not merely duplicate the work: the same
question returned **SIGSEGV, SIGABRT and an indefinite hang** on different runs,
and always succeeded once the model was already warm.

Two changes, because one alone is not enough:

- Each lazy initializer is now double-checked under its own lock, so concurrent
  construction is safe wherever it happens.
- `ask` calls `warm_up` on one thread *before* the fan-out when more than one
  scope is selected, so it does not happen at all. `mcp_server` already did this
  at boot for the same reason; a one-shot CLI invocation had no equivalent.

The race is not new — any question routing to two scopes on a cold process could
reach it — but enumerative set mode selects two or more scopes far more often,
which turned a rare crash into the first thing a user would run.

### MCP is documented as installable

The server was built and reachable by nothing. The README now carries the
`claude mcp add` line and the `claude_desktop_config.json` block, with the note
that version-managed installs need an absolute path.

### Honest reporting

- **The `behavior` family fell from 85.7% to 28.6%** when the corpus grew from
  10 scopes to 14. Not a threshold: `SIZE_PRIOR` swept 0.0–0.5 never beats the
  shipped 0.15. Five alternative scoring families were swept and none beat the
  plain sum. The one strict improvement ships **inert** as
  `router.CORROBORATION_WEIGHT = 1.0`; enabling it on one item from one corpus
  would be exactly the mistake the roadmap's rules exist to prevent.
- **The synthetic test bed cannot reproduce that failure.** `CorpusSpec.size_skew`
  varies file volume, not vocabulary breadth, so all thirteen shapes read 100%
  at every value. Fixing the bed is now roadmap Phase 5.5 and it blocks the
  router change.
- **Two eval golds were stale**, one of which marked a *correct* three-scope
  answer down as a precision failure.
- `evals/RESULTS.md` keeps the superseded numbers in place rather than editing
  them out, with a new section recording what moved and why.

## 0.1.0 — 2026-08-28

First public release. Router, structure store, episode store, groups, MCP
server, calibration, eval harness. CI across Linux, Windows and macOS on Python
3.10–3.13.
