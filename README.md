# loci

**Scoped memory for coding agents.** A router in front of two stores.

```
question ──▶ router ──▶ ┌── structure store   what calls what
             (no LLM)   └── episode store     what happened and why
                        └─▶ merged, cited answer   │
                            or ABSTAIN ────────────┘
```

Existing memory tools give you one of these. Knowledge graphs hold structure but
no prose, so they cannot answer *why did the cookie get dropped*. Verbatim
recall systems hold prose but no call graphs, so they cannot answer *what calls
`run_agent_turn`*. Both make you name the namespace when you **write**.

loci decides the scope when you **read**. One question path serves *"how does
auth work here"* and *"have I solved this in any project"* — only the size of
the scope set changes.

## Install

The distribution is **`loci-mem`**; the command, the import and the project are
all `loci`. PyPI's `loci` is an unrelated outlier-detection package abandoned in
2018 — same split as `python-dateutil` installing as `dateutil`.

```bash
pipx install loci-mem                    # routing + lexical episode search
pipx install 'loci-mem[all]'             # + graphify, embeddings, MCP server
```

## Use

```bash
loci scan ~/code            # register every git repo it finds
loci graphs                 # optional: add code symbols (free, no model calls)
loci index                  # build the routing index + episode store
loci embed                  # optional: local vectors for semantic recall
loci doctor                 # what is missing, and the command to fix it

loci ask "why was the session cookie dropped on localhost?"
loci ask "which projects use wrangler and D1?"
loci route "how do I run the tests?" --explain
```

`ask` uses your working directory by default, and should. Questions that name no
project — *"how is this deployed?"*, *"how do I run the tests?"* — route
correctly **12/12 with cwd and 1/4 without**. cwd is not a tiebreaker signal
here, it is the primary one.

## First run, from nothing

loci needs no pre-existing index of any kind. A cold machine has git history,
READMEs and docstrings, and that is enough — measured on 11 repos with no
graphify graphs and no notes of any kind:

```
11 repos discovered           0.2s
indexed                      44.1s      16,091 chunks, 7.4MB
                                        docstrings 13,373 · docs 1,841 · commits 877
```

Cold routing is usable, and **the path that dominates real use is unaffected**:

| | with code symbols | cold, prose only |
|---|---|---|
| deictic question, **with cwd** | 100% | **100%** |
| deictic question, no cwd → abstains | 87.5% | **87.5%** |
| cross-scope | 100% | 66.7% |
| negatives correctly refused | 66.7% | 50.0% |
| mean scopes returned | 1.43 | 3.43 |

What degrades cold is *discrimination* — wider result sets, more plausible-but-
absent questions slipping through — because without symbol vocabularies, scopes
are told apart only by prose, which is thinner and more alike across projects.

`loci graphs` closes that in one command, per scope or all at once. It shells
out to `graphify update`, which is AST-only: no model calls, no API cost,
seconds per repo.

```
$ loci graphs zim-compress loci
building structure graphs for 2 scope(s) (AST-only, no model calls, no API cost)
  zim-compress ... 78 symbols
  loci ... 268 symbols
```

`loci scan` and `loci index` both name what is missing and the command that
fixes it, so this is not something to discover from a diagnostic you had no
reason to run.

## MCP

```bash
loci mcp        # stdio server
```

Three tools — `ask`, `scopes`, `doctor` — not several dozen. Agents are the
primary consumer, and a wide tool surface pushes orchestration onto the model
and burns a turn per hop. Pass `cwd` to `ask` whenever the client knows it.

## Why it is not a graphify fork

loci **depends** on [graphify](https://github.com/safishamsi/graphify) through
one adapter, `backends/graphify.py`, which reads its `graph.json` data contract
and shells out to `graphify query --graph`. Nothing above `backends/` knows which
structure store is in use.

Forking would mean inheriting a 767KB extractor and ~18 tree-sitter grammars in
order to change code that never needed changing. The scoping gap that motivated
loci — graphify tags merged-graph nodes with a `repo` attribute and exposes no
way to filter on it — is solved *above* graphify by keeping one graph per scope
and pointing `--graph` at the right file. Both projects are MIT, so forking stays
available; it just is not necessary.

The same applies to episode storage: `backends/base.py` defines the contract,
the builtin backend is the default, and an adapter for another store is one file.

## Design rules, and the measurements behind them

Each of these is something a reasonable implementation gets wrong.

**1. Never merge scopes into one index.** Against a merged graph, the same
questions returned **2%** and **18%** on-topic nodes — the largest project won
regardless of the question. It looks excellent whenever the answer happens to
live in the biggest corpus, and collapses when it does not. Same failure in both
directions; visible in only one.

**2. Abstention is a feature at both layers.** The router abstains below
`MIN_MATCHED` grounded tokens. The episode store abstains unless a question is
lexically grounded **or** semantically confident. Measured 10/10 on probes
including deliberate nonsense.

**3. Margin and ratio tests do not measure confidence.** Both were tried and
both invert on real data. *"How do I fix this bug?"* produces routing margin
0.94 while a correctly-routed question produces 0.18. *"The airspeed velocity of
an unladen swallow"* scored 0.608 against a large scope with a **higher**
top-to-p90 ratio than a genuine question against a small one. What separates
them is whether the question's words exist in the corpus at all.

**4. Normalize for scope size, twice.** Scope-IDF is confounded by vocabulary
size: with one scope 67x larger, ordinary English words appear in exactly one
scope and score as maximally discriminative for it. `SIZE_PRIOR` was worth ~9
points of top-1 accuracy on its own.

**5. Coverage is the binding constraint, not ranking.** This surfaced in every
experiment. Running a free AST-only index over three un-indexed repos moved
routing 69.6% → 82.6% *while adding two more scopes to compete against*. Hence
`doctor` as a first-class command: an empty answer with a reason beats a
confident answer from the wrong project.

**6. Nothing leaves the machine.** No LLM in the query path; routing is a dict
lookup per token. Embeddings, when enabled, run locally — the alternative is
shipping every private note and commit message to a third party, which is a
strange thing to do to a tool whose premise is holding your working history.

## Ranking

Episode search fuses three signals, renormalizing over whichever produced one:

| ranker | catches |
|---|---|
| BM25 | exact terminology, rare identifiers |
| char 3–5 grams | morphology and casing — `samesite` vs `SameSite=None` |
| embeddings (bge-small, local, optional) | meaning without shared words |

Cross-encoder reranking is available via `--rerank` and **off by default**: it
measured 2/6 → 3/6 precision@1 at ~96ms/query with two small regressions, on six
cases whose labels are themselves arguable. Turn it on when it wins on questions
you wrote.

## Status

Alpha. Measured on a 98-item set over 10 scopes — see `evals/RESULTS.md` for
method, per-family numbers, and known costs.

```
deictic questions, with cwd          top-1 100.0%
deictic questions, without cwd       abstains 100.0%   (abstaining is correct)
plausible-but-absent questions       refused  83.3%
behavior questions, uncontaminated   top-1     85.7%
structure store, answering symbol    4/7
retrieval, answer content @3         11/12
```

Thresholds were fitted against two corpora at once — the same projects indexed
with and without code symbols — and one set is optimal for both, so they are not
tuned to a single corpus shape.

80 of those items are generated from a fixed taxonomy applied to every scope, so
that portion is unbiased by construction. The rest were hand-authored by the
same person who built the router; each carries a `contamination` field recording
what the author had seen, and uncontaminated results are reported separately.
Independent replication is still the gating item before any public claim.

**Closed since:** docstrings and code comments are now mined as episode chunks
(`backends/docstrings.py`), keyed by their enclosing symbol. They are searchable
but deliberately excluded from *routing* vocabulary — symbol vocabulary is
bounded by the code and docstring prose is not, so mixing the units breaks the
size prior. Including them cost 67 points of cross-scope routing accuracy before
that split; see `evals/RESULTS.md`.

## License

MIT.
