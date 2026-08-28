---
name: loci
description: "Scoped memory across every project you have registered: what a project does, why a decision was made, and whether you have solved something before. Use when the user types /loci (any subcommand), asks about work in another project, asks why something was done a particular way, wonders whether a problem has been solved before, or asks a question about the current project that reading the working tree cannot answer. Also refreshes and diagnoses that memory: /loci update, /loci doctor."
---

# /loci

Scoped memory for coding agents. A router in front of two stores: a **structure
store** (what calls what, with `file:line` citations) and an **episode store**
(what happened and why — READMEs, docs, commit bodies, comment blocks). One
question path serves *"how does auth work here"* and *"have I solved this in any
project"*; only the size of the scope set changes.

Two properties shape everything below. **Scopes are never merged** — a question
is routed to at most three projects and each is searched alone. **Both layers
can refuse** — the router can decline to pick a project, and a project it picked
can hand back nothing. A refusal is an answer.

## Usage

```
/loci "<question>"          ask memory; routes on your working directory
/loci update                refresh everything registered
/loci update <root>         ...and scan there for projects added since
/loci doctor                coverage gaps per project, and the fix for each
/loci scopes                what is registered, and the group each is in
/loci route "<question>"    where a question routes, and why it abstained
/loci add <path>            register one project (`/loci add .` for this repo)
/loci setup [dirs…]         first run: scan, graph, index, embed, calibrate
/loci <anything else>       passed through to the `loci` CLI verbatim
```

## What to do when invoked

If the user typed `/loci --help` or `/loci -h` with nothing else, print the
Usage block above verbatim and stop.

If `command -v loci` finds nothing, say so once and stop: the fix is
`pipx install 'loci-mem[all]'`, and no subcommand can work without it.

Then dispatch on the first word. A first word that is not a subcommand below and
not a flag is a **question** — send it to `ask`, do not treat it as a path.
Anything that names a real `loci` subcommand not documented here (`index`,
`embed`, `graphs`, `calibrate`, `eval`, `groups`, `group`, `scan`, `mcp`) is
passed through unchanged.

**Prefer the MCP server when it is connected.** If tools named `ask`, `scopes`
and `doctor` are available from a `loci` MCP server, call those instead of
shelling out, and pass `cwd` explicitly — the server holds the embedding model
and the largest projects' rankers in memory, so a call costs ~0.1s against the
~4.4s a cold CLI invocation pays in imports and model construction. Everything
else (`update`, `route`, `add`, `setup`) has no tool; use the CLI.

## Asking

```bash
loci ask "<the user's question>"
```

Run it from the project root and let it read the working directory. **cwd is the
primary routing signal, not a tiebreaker**: questions that name no project —
*"how is this deployed?"*, *"how do I run the tests?"* — route correctly 100% of
the time with it and are unanswerable without it. Never pass `--no-cwd` unless
the user is deliberately asking across projects from an unrelated directory.

- `--scope <name>` when the user names a project. It skips routing entirely and
  drops the episode gate with it, so use it whenever the target is not in doubt.
- `--group <name>` to confine to a labelled set of projects.
- `--fast` skips semantic ranking: ~5s faster per invocation, lower recall. For
  a quick check, not for a question the user is waiting on an answer to.
- `--json` when you need to parse the result rather than read it.

## Reading what comes back

Three shapes, and they mean different things.

**`ROUTED -> <projects>`** followed by one block per project. Answer only from
what those blocks contain. Quote `file:line` from structure hits and
`kind:source` from episode chunks; the citations are the point.

**`no evidence in this scope`** — the router picked that project and both of its
stores came back empty. Say that plainly. If you then answer from the working
tree instead, say that is what you did.

**`ABSTAINED - <reason>`**, with the candidates it considered and one suggested
flag. **This is an outcome, not an error, and not a prompt to try again.**
Rephrasing and re-asking is the wrong move — abstention is about which
vocabulary exists in which project, not about wording. Act on the reason:

| reason | what it means | what to do |
|---|---|---|
| `the question points at its subject without naming it` | the question said "this" or "it" and nothing identified the subject | name the project with `--scope`, or ask the user which one |
| `not enough of the question exists in any project` | memory does not hold this | say so, and fall back to ordinary tools |
| `best match was outside group X` | the real answer is outside the group's policy | `--group` or `--scope` to override deliberately |
| `no indexed project is in group X` | almost always a typo in a group name | `loci groups` lists the real ones |

## /loci update

```bash
loci update                 # everything registered
loci update ~/code          # ...and scan there, remembering the root
```

Nothing in loci updates itself — there is no watcher and no installed hook — so
this is the whole refresh: it rescans wherever the last scan was pointed,
rebuilds every structure graph, reindexes the projects whose fingerprint moved,
re-encodes vectors if the user already had them, refits the routing floor if it
was already fitted, and ends with `doctor`.

A registry created before roots were recorded has none, so the first run may
report *"no scan root recorded"*. Name the directory once (`loci update ~/code`)
and every run after it needs no arguments.

It prompts for nothing, so it is safe to run unattended. Suggest it after a
session that changed a lot of code, and whenever `doctor` reports staleness.

## /loci doctor, /loci scopes

`loci doctor` reports coverage gaps per project and names the command that fixes
each, plus group policy and stale embeddings. `loci scopes` lists what is
registered. Both are cheap and read-only; run them rather than guessing at the
state of the corpus.

## /loci route

```bash
loci route "<question>" --explain
```

For *"why did it answer from that project?"* and *"why did it not answer?"*.
Routing is deterministic and involves no model call — an explicit project name
scores 6.0, the working directory 4.0, vocabulary evidence 0.1–1.5, recency
0.15 — so `--explain` gives a real account rather than a guess. Reach for it
whenever a user doubts an answer's provenance.

## /loci add

```bash
loci add <path>        # `.` for the repo you are standing in
loci update            # build its graph and index it
```

`add` only registers. Until something indexes it the new project is in the
registry and unqueryable, so always chain the second command and say you did.

## /loci setup

First run only. Run it from a project root with the directories that hold the
user's repositories: `loci setup ~/code`.

Every prompt takes its default when stdin is not a terminal, which is the case
for you — and one of those defaults is a one-time ~130MB embedding-model
download. If the user has not asked for semantic search, pass `--no-embed`, and
say which flags you chose either way.

## When to reach for this without being asked

Ask memory **before** broad `grep`/`read` exploration in a registered project,
and whenever:

- the user refers to another project, or to how they did something elsewhere
- the user asks *why* something is the way it is, and the answer would live in a
  commit body or a design doc rather than in the code
- the user wonders whether they have solved a problem before

Do **not** reach for it when reading a file answers the question, when the work
is in a directory no scope covers, or when the user is asking you to write code
rather than to recall anything. A memory lookup that returns
`not enough of the question exists in any project` cost the user a turn.
