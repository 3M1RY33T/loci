"""Index construction: the routing index, the episode store, the vectors.

Routing must be O(query tokens). Loading every scope's graph to answer "which
project is this about" defeats the entire design -- one real graph here is 36MB.
So the scope index is an inverted `token -> {scope: node_df}` map, built once.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .backends import get_episode_backend, get_structure_backend
from .paths import (BuildLock, atomic_write, atomic_write_via, embeddings_file,
                    ensure_home, episode_store_file, scope_index_file)
from .types import Chunk, Scope

INDEX_VERSION = 1
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Chunk kinds whose vocabulary feeds ROUTING by default. Docstrings are held
# back, and it is a units problem rather than a quality one: a scope's symbol
# vocabulary is bounded by its code, while its docstring prose is not, so mixing
# the two makes scopes incomparable and the size prior stops correcting for it.
# Measured -- folding docstrings in unconditionally more than doubled one
# scope's vocabulary, made it top-ranked for questions belonging elsewhere, and
# dropped cross-scope routing from 100% to 33%.
ROUTING_KINDS = {"note", "doc", "session", "commit"}

# Making that conditional on whether a scope has symbols was tried and measured,
# because a cold machine has no graphs and thin scopes then route on almost
# nothing. It is worse in BOTH regimes, so the exclusion stays unconditional:
#
#   warm, fallback off      cwd 100%   clean 85.7%   cross 100%   set 2.50
#   warm, fallback on       cwd 100%   clean 71.4%   cross  67%   set 3.83
#   cold, docstrings off    cwd 100%   clean 57.1%   cross  67%   set 3.58
#   cold, docstrings on     cwd 100%   clean 28.6%   cross  33%   set 3.08
#
# Warm it fired on ONE scope (+478 tokens) and cost 14 points, which suggested
# the damage was asymmetry -- adding vocabulary to some scopes and not others
# distorts a comparison that is between scopes. Applying it uniformly on a cold
# machine falsified that: it halved accuracy there too.
#
# The real reason is what docstrings are made of. They are high-volume generic
# English -- "returns", "the value", "default", "configuration", "path", "error"
# -- which is maximally non-discriminative, and at 6,000 chunks per scope it
# drowns the signal that symbol names and commit subjects carry. Docstrings are
# excellent RETRIEVAL material and poor ROUTING material, and having no graph
# does not change that. The fix for a cold scope is `loci graphs`, not this.
#
# Set above zero only to reproduce the experiment; nothing in loci enables it.
DOCSTRING_FALLBACK_VOCAB = 0


def fingerprint(scope: Scope, sb, *, exclude=()) -> str:
    """Cheap content signature: which files exist and when they last changed.

    Stat-ing every candidate file is fast; parsing them is not. Reindexing an
    unchanged scope re-parses thousands of files to produce byte-identical
    chunks, so the signature is what makes `loci index` idempotent in practice
    rather than only in result.

    `exclude` must match what `collect` was given: a signature covering files a
    sub-scope owns would move whenever that sub-scope did, and the parent would
    re-parse its whole tree to rebuild identical chunks.
    """
    import hashlib

    from .walk import iter_files

    h = hashlib.sha256()
    globs = list(scope.episode_globs or []) + list(scope.code_globs or [])
    for f in iter_files(scope.root, globs, exclude=exclude):
        try:
            h.update(str(f).encode("utf-8", "replace"))
            h.update(str(int(f.stat().st_mtime)).encode())
        except OSError:
            continue
    for g in sb.graph_paths(scope) if hasattr(sb, "graph_paths") else []:
        try:
            h.update(f"{g}:{int(g.stat().st_mtime)}".encode())
        except OSError:
            continue
    # git HEAD moves independently of any working-tree file
    head = scope.root / ".git" / "HEAD"
    try:
        h.update(head.read_bytes())
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            rp = scope.root / ".git" / ref[5:]
            if rp.is_file():
                h.update(rp.read_bytes())
    except OSError:
        pass
    return h.hexdigest()[:16]


def build(scopes: list[Scope], *, structure: str = "graphify",
          episodes: bool = True, episode_vocab: bool = False,
          verbose: bool = True, force: bool = False) -> dict:
    """Build and persist the scope index and (optionally) the episode store.

    `episode_vocab` folds episode prose into the ROUTING vocabulary. It measured
    a large gain, but on a question set authored from that same prose -- so it
    is off by default and the honest number is the structure-only one.
    """
    sb = get_structure_backend(structure)
    eb = get_episode_backend()

    postings: dict[str, dict[str, int]] = defaultdict(dict)
    meta: dict[str, dict] = {}
    store: dict[str, list[dict]] = {}

    prev_index: dict = {}
    prev_store: dict = {}
    if not force:
        try:
            prev_index = load_index()
            prev_store = load_episodes()
        except (FileNotFoundError, ValueError):
            pass
    reused = 0
    redacted: dict[str, dict[str, int]] = {}

    from .scopes import nested_roots
    # Exclusion is a property of the registry, so it is computed here and passed
    # down rather than stored on any scope. Without it a monorepo parent
    # collects the very files its sub-scopes own: two vocabularies inflated with
    # identical tokens, and a size prior corrupted for both.
    excludes = {sc.id: nested_roots(sc, scopes) for sc in scopes}

    for sc in scopes:
        ex = excludes[sc.id]
        counts = sb.vocabulary(sc, exclude=ex) if sb.available() else {}
        nodes = sb.node_count(sc, exclude=ex) if sb.available() else 0
        srcs = sb.sources(sc) if sb.available() else []

        fp = fingerprint(sc, sb, exclude=ex)
        prev = prev_index.get("scopes", {}).get(sc.id, {})
        unchanged = bool(prev) and prev.get("fingerprint") == fp \
            and sc.id in prev_store.get("chunks", {})

        chunks: list[Chunk] = []
        used_docstrings = False
        if episodes:
            if unchanged:
                chunks = [Chunk.from_json(c) for c in prev_store["chunks"][sc.id]]
                reused += 1
            else:
                chunks = eb.collect(sc, exclude=ex)
                red = eb.redactions() if hasattr(eb, "redactions") else {}
                if red:
                    redacted[sc.id] = red
            store[sc.id] = [c.to_json() for c in chunks]
            # `episode_vocab` augments structure vocabulary and is off by
            # default. But a scope with NO structure graph has nothing to
            # augment, and dropping it would make a docs-only or notes-only
            # project unroutable while its chunks sit indexed and unreachable.
            # Prose is then the only vocabulary there is, so use it.
            if episode_vocab or not counts:
                routable = [c for c in chunks if c.kind in ROUTING_KINDS]
                prose_vocab = eb.vocabulary(routable)
                if not nodes and len(prose_vocab) < DOCSTRING_FALLBACK_VOCAB:
                    routable += [c for c in chunks if c.kind == "docstring"]
                    prose_vocab = eb.vocabulary(routable)
                    used_docstrings = True
                for t, n in prose_vocab.items():
                    counts[t] = counts.get(t, 0) + n

        if chunks and not unchanged:
            eb.save_rankers(sc.id, chunks)

        if not counts:
            if verbose:
                print(f"  - {sc.name:<18} nothing indexable, skipped")
            continue

        for t, c in counts.items():
            postings[t][sc.id] = c

        meta[sc.id] = {
            "name": sc.name, "root": str(sc.root), "aliases": sc.aliases,
            # Size normalization needs a unit comparable across scopes. For a
            # graph-backed scope that is nodes; for a prose-only scope it is
            # chunks. Falling back to 1 would make every token maximally
            # prominent and the scope would win every query it touched.
            "node_count": max(nodes or sum(1 for c in chunks
                                           if c.kind in ROUTING_KINDS) or len(chunks), 1),
            "structure_nodes": nodes,
            "fingerprint": fp,
            "docstring_routing": used_docstrings,
            "token_count": len(counts),
            "chunk_count": len(chunks), "sources": srcs,
            "updated_at": sc.updated_at,
        }
        if verbose:
            kinds = ",".join(s["kind"] for s in srcs) or "-"
            tag = " (unchanged)" if unchanged else ""
            print(f"  {'=' if unchanged else '+'} {sc.name:<18} {nodes:>7} nodes  "
                  f"{len(counts):>6} tokens  {len(chunks):>5} chunks  [{kinds}]{tag}")

    if redacted:
        total = sum(sum(v.values()) for v in redacted.values())
        kinds = Counter()
        for v in redacted.values():
            kinds.update(v)
        if verbose:
            print(f"\n  redacted {total} credential(s) before indexing: "
                  f"{', '.join(f'{k}x{n}' for k, n in kinds.most_common())}")
            for sid, v in redacted.items():
                print(f"    {meta.get(sid, {}).get('name', sid)}: "
                      f"{', '.join(f'{k}x{n}' for k, n in v.items())}")
    if verbose and reused:
        print(f"\n  {reused}/{len(scopes)} scope(s) unchanged, reused without re-parsing")
    index = {
        "version": INDEX_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "structure_backend": structure,
        "episode_vocab": episode_vocab,
        "scopes": meta,
        "postings": dict(postings),
    }
    ensure_home()
    # Store first, index second: the index is what readers gate on, so a crash
    # between the two leaves a stale index pointing at a store that is a
    # superset of it, never an index promising chunks that do not exist.
    if episodes:
        atomic_write(episode_store_file(), json.dumps({
            "version": INDEX_VERSION,
            "built_at": index["built_at"],
            "scopes": {sid: m["name"] for sid, m in meta.items()},
            "chunks": store,
        }))
    atomic_write(scope_index_file(), json.dumps(index))
    return index


def load_index() -> dict:
    f = scope_index_file()
    if not f.is_file():
        raise FileNotFoundError(
            f"no scope index at {f}. Run `loci index` first.")
    try:
        idx = json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"scope index at {f} is corrupt ({exc}). Run `loci index` to rebuild."
        ) from exc
    if idx.get("version") != INDEX_VERSION:
        raise ValueError(
            f"scope index at {f} is version {idx.get('version')}, "
            f"expected {INDEX_VERSION}. Run `loci index` to rebuild.")
    return idx


def load_episodes() -> dict:
    f = episode_store_file()
    if not f.is_file():
        return {"version": INDEX_VERSION, "scopes": {}, "chunks": {}}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"episode store at {f} is corrupt ({exc}). Run `loci index` to rebuild."
        ) from exc


def chunks_for(store: dict, scope_id: str) -> list[Chunk]:
    return [Chunk.from_json(d) for d in (store.get("chunks", {}).get(scope_id) or [])]


def embeddings_status() -> list[str] | None:
    """Scopes whose vectors no longer match the store. None when absent entirely.

    Reindexing without re-embedding leaves per-scope arrays the wrong length.
    `episodes._semantic` correctly refuses to use them -- but silently, so the
    tool quietly degrades to lexical-only and nothing says why. Surfacing it is
    the whole point of a tool that claims coverage gaps are a feature.
    """
    import numpy as np

    f = embeddings_file()
    if not f.is_file():
        return None
    store = load_episodes()
    try:
        z = np.load(f, allow_pickle=False)
    except Exception:
        return sorted(store.get("chunks", {}))
    names = store.get("scopes", {})
    stale = []
    for sid, chunks in (store.get("chunks") or {}).items():
        if not chunks:
            continue
        if sid not in z.files or z[sid].shape[0] != len(chunks):
            stale.append(names.get(sid, sid))
    return stale


SYMBOL_PREFIX = "sym::"
MAX_SYMBOLS = 30000


def build_embeddings(model_name: str = DEFAULT_EMBED_MODEL,
                     verbose: bool = True) -> Path:
    """Encode every chunk once, and every symbol label too.

    Symbol labels are encoded so a question can be matched to code
    SEMANTICALLY before being handed to the structure store. graphify seeds a
    traversal by lexical similarity to a node label, which cannot connect "what
    removes stored data from the browser?" to `clearStorage()` -- measured, it
    seeded on `getStoredWorkerUrl()` instead -- nor "what parses the command
    line arguments?" to `build_parser()`, where the expansion came out empty and
    no traversal ran at all. Nearest-label search finds both.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    store = load_episodes()
    model = SentenceTransformer(model_name)
    arrays: dict[str, "np.ndarray"] = {}

    from .backends import get_structure_backend
    from .scopes import load_scopes, nested_roots
    sb = get_structure_backend()
    registry = load_scopes()
    by_id = {s.id: s for s in registry}
    for sid in store.get("chunks", {}):
        scope = by_id.get(sid)
        if scope is None or not hasattr(sb, "labels"):
            continue
        # Same exclusion as indexing: otherwise a parent seeds its semantic
        # symbol search from labels that belong to its sub-scopes, and a
        # question about the sub-project matches in both.
        labels = sb.labels(scope, exclude=nested_roots(scope, registry))[:MAX_SYMBOLS]
        if not labels:
            continue
        arrays[f"{SYMBOL_PREFIX}{sid}"] = np.asarray(
            model.encode(labels, normalize_embeddings=True, batch_size=256,
                         show_progress_bar=False), dtype="float32")
        Path(embeddings_file().parent / f".symbols-{sid}.json").write_text(
            json.dumps(labels), encoding="utf-8")
        if verbose:
            print(f"  {store['scopes'].get(sid, sid):<18} {len(labels):>5} symbols")

    for sid, raw in store.get("chunks", {}).items():
        if not raw:
            continue
        texts = [f"{c.get('heading','')} {c['text']}" for c in raw]
        arrays[sid] = np.asarray(
            model.encode(texts, normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False), dtype="float32")
        if verbose:
            print(f"  {store['scopes'].get(sid, sid):<18} {len(texts):>5} chunks "
                  f"-> {arrays[sid].shape}")
    out = embeddings_file()
    ensure_home()
    atomic_write_via(out, lambda tmp: np.savez_compressed(
        tmp, _model=np.array([model_name]), **arrays))
    return out
