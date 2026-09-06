# Changelog

## 0.5.0 — 2026-09-06

**No reindex.** `INDEX_VERSION` stays at 3 and nothing about what a token is
changed. Upgrading is `pipx upgrade loci-mem`, then one command to populate the
new table:

```bash
loci update          # refreshes signboards, then collects edges
```

`loci update` now refreshes every scope's signboard whether or not it scanned,
because an install that predates this release has none — and an edge can only
name a project that has one.

### A question the router could not be asked

*"In which project am I using another one of my projects?"* Asked verbatim in
three sessions on 2026-09-06, all three abandoned:

```
ABSTAINED - not enough of the question exists in any project.
  candidates: loci (another), tensor-serve (using), Delroy (another, using)
```

The abstention was correct. Every content-bearing part of that question names a
**relation between scopes**, and the evidence model is
`token -> {scope: node_df}`, so there is nothing to look up — the shortlist is
what `CANDIDATE_SHARE` admits when the only terms available are `another` and
`using`. The fact the question wants is not in either store. It is in the
registry, and reaching it took three pieces.

**Signboards.** What a project is called from the outside — repository,
distribution, import, command — read from git and root manifests at
registration. loci is the case that forces the shape: `loci-mem` on PyPI,
`loci` to import, `loci` to run, `3M1RY33T/loci` on GitHub. Four names, one
project, and the only one any edge in the corpus mentions is the third.

**Edges.** What a project reaches for, collected two ways, because a real
corpus needs both:

```
3m1ry33t-github-io -> urthreads   "urthreads": "^1.2.0"   package.json:10
delroy             -> loci        shutil.which("loci")    client/loci_memory.py:36
```

There is no `.gitmodules` anywhere in the 15-project corpus, no path dependency
and no git-URL dependency — so the declared half rests entirely on the plain
registry name — and `Delroy -> loci` is a binary resolved on `PATH`, declared
in no manifest, lockfile or submodule. **Either collector alone has a recall of
0.5.**

**A relational path**, ahead of routing. Without it the capability is
unreachable: `route` can only choose scopes, and this answer is not a scope.

```
loci ask "in which project am I using another one of my projects?"

USES -> 2 cross-project edge(s)
  3M1RY33T.github.io -> urthreads   (depends on `urthreads`)  package.json:10
  Delroy -> loci                    (runs `loci`)  client/loci_memory.py:36
```

`loci uses` prints the same thing on demand; `loci doctor` reports the coverage.

### An empty scan never answers "none"

The one regression that would have made this a net loss. *"None of your
projects use OAuth"* — false, and the statement this whole line of work started
from — is what a confident empty scan produces. An empty edge table is
indistinguishable from an uncollected one, so it never terminates the question:

```
no cross-project edge answers this: 0 edge(s) from 0 outbound reference(s),
  collected from 0 of 1 project(s); nothing to collect from: Alpha ...

ABSTAINED - not enough of the question exists in any project.
```

Coverage has two independent holes and both are reported, because either one
shrinks the answer silently: a scope nothing could be collected **from** can
never be the source of an edge, and a scope with no signboard can never be the
**target** of one.

### Routing did not move

The relational path runs before `route`, and the guard on it is the registry: a
named object is taken only when the name **is** a registered project, so
*"which of my projects use wrangler?"* stays with the enumerative path that
answers it.

| family | 0.4.0 | 0.5.0 |
|---|---|---|
| behavior | 27.3% | *unchanged* |
| confusable | 28.6% | *unchanged* |
| cross | 25.0% | *unchanged* |
| enumerative | 85.7% (7) | 87.5% (8) |
| negative | 92.9% (14) | 93.3% (15) |

Taxonomy holds at 100% top-1 with cwd and 100% abstention without. The two
moves are the two new items passing.

### Rejected on measurement

**Trusting a `src/` layout as packaging intent.** A project's importable
packages are part of its signboard, and reading them off the filesystem is the
only way that works across setuptools, hatchling, poetry and flit. Both loose
rules put fabricated identities on signboards:

| rule | what it claimed |
|---|---|
| every root dir with `__init__.py` | `api`, `cli`, `search`, `agent_tools` |
| trust `src/` on its own | `search`, `agent_tools` |

One real project keeps application code in `src/` under a pyproject declaring
only `[tool.pytest.ini_options]`. An edge naming `search` resolving to it is a
fabricated dependency, which is worse than the missing edge it replaces. Import
identity now requires a declared distribution in either layout.

### Known, and not fixed here

Vendored or copied code carrying no name, and *"I reused the approach from X"*.
Those are similarity questions rather than identity questions, and nothing here
reaches them. The function-word shortlist is also untouched: the relational path
routes **around** it, and every other question that abstains still gets a
candidate list built the same way.

## 0.4.0 — 2026-09-05

**Upgrading requires one command.** `INDEX_VERSION` is now 3 and a v2 index is
rejected on load with the reason. Run:

```bash
loci update          # or: loci index && loci calibrate
```

The tokenizer changed what counts as a word again, so the old vocabulary is
genuinely incompatible rather than merely older.

### `OAuth` was indexed as `auth`, and could not find itself

The camel splitter emits `O` + `Auth`, the solitary `O` dies on the length
floor, and the term the writer typed survived only as the ordinary word `auth`.
Lowercase occurrences in code (`oauth_config`, `oauth_tokens`) index as `oauth`,
so the two halves of a corpus disagreed: one real project held `oauth` 50 times
and no naturally-written question could ever reach it.

A split that discards a piece now keeps the whole run as well:

| written | 0.3.0 | 0.4.0 |
|---|---|---|
| `OAuth` | `auth` | `oauth`, `auth` |
| `macOS` | `mac` | `macos`, `mac` |
| `OpenID` | `open` | `openid`, `open` |
| `GraphQL` | `graph` | `graphql`, `graph` |
| `IOError` | `error` | `ioerror`, `error` |
| `GlassesBridge` | `glasses`, `bridge` | *unchanged* |

Conditioned on a piece actually being lost, so this is not a blanket "keep the
run too" — a name that splits cleanly gains nothing, which is what stops the
vocabulary doubling. The output stays a **superset**, the same rule 0.2.0
followed for `base64`, so no query that matched before can stop matching.

### A project named as the tool is not the subject

*"Can you use loci to tell me in which project am I using another one of my
projects?"* returned the `loci` scope, alone. The user addressed the tool; the
router read it as the subject.

Scope names used in the instrumental position are now stripped from the query
before it is tokenized — the same treatment `ENUM_FRAME` already gives
`projects` and `repos`, for the same stated reason: scaffolding, not vocabulary.
Detection is grammatical, a closed class of verbs that take a tool as their
object, with no threshold to fit. Naming a project anywhere else is untouched:
*"how does Alpha handle the widget"* still routes to Alpha.

### Measured

Both changes against the same index and the same 67-question hand-authored set,
so this isolates the code from a corpus repair that landed in the same release:

| | top-1 | gold-coverage | negatives correctly abstained |
|---|---|---|---|
| 0.3.0 | 42.6% | 45.1% | 92.3% |
| **0.4.0** | **46.3%** | **48.8%** | **92.3%** |

| family | 0.3.0 | 0.4.0 |
|---|---|---|
| cross | 28.6% | **42.9%** |
| enumerative | 71.4% | **85.7%** |
| behavior, confusable, negative, structure | — | *unchanged* |

Nothing regressed, and negative abstention did not move: neither change buys
coverage by answering questions it cannot know.

### Rejected on measurement

**Suppressing `ALIAS_BOOST` for an instrumentally-named scope.** The obvious fix
for the second bug, and it changes nothing at all: a scope's own name is
ordinary vocabulary inside it — `loci` sits in 407 nodes of the loci scope
against 1 elsewhere — so the scope still wins on evidence with the boost gone.
The name has to leave the query, not just the alias test. There is a test
pinning this, because it is the fix a reader will propose again.

### Known, and not fixed here

A large scope can still lose a term it genuinely owns. *"…in which one of my
projects am I using oauth?"* now abstains rather than answering confidently
wrong, but the right project is third on the shortlist instead of first: 50
occurrences inside a 16,793-node scope is 0.30% prominence, below the
concentrated tier's threshold, while generic vocabulary elsewhere outweighs it.
Opening that tier wide lifts coverage to 54.3% and collapses negative abstention
from 92.3% to 53.8%, which is the trade this design refuses. It needs a fitted
threshold, not a patch.

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
