"""Episode backend: verbatim prose, chunked, ranked by three fused signals.

Structure stores hold `{label, source_file, source_location}` and no text, so
they cannot answer "why did the cookie get dropped" or "what did we decide about
lanes" -- the words those questions turn on are absent from every node label,
and no amount of ranking fixes an absent token. That gap is what this store
exists to close.

Ranking fuses three signals because each fails where the others work:

    BM25 (word)          exact terminology, rare identifiers
    char 3-5 grams       morphology and casing: "samesite" vs "SameSite=None"
    embeddings           meaning without shared words: "external services"
                         against "Deployment: Cloudflare Workers"

Embeddings are optional. Without a vector cache the weights renormalize over the
two lexical rankers and everything still works.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..paths import (LEX_SUFFIX, atomic_write_via, embeddings_file,
                     rankers_dir)
from ..redact import is_sensitive_file, redact
from ..text import tokens as vtokens
from ..text import token_set
from ..types import Chunk, EpisodeHit, Scope

# -- chunking --------------------------------------------------------------
TARGET_CHUNK_CHARS = 512   # a cross-encoder window is 512 tokens; oversized
HARD_MAX_CHARS = 900       # chunks get truncated there and dilute an embedding
OVERLAP_CHARS = 80         # carry a tail forward so a split fact survives
MIN_CHUNK_CHARS = 40
MIN_CONTENT_WORDS = 8
NAV_LINK_RATIO = 0.6
MAX_COMMITS = 400
MAX_CODE_FILES = 1500      # a hard stop, so one huge repo cannot dominate

# -- ranking ---------------------------------------------------------------
# Fitted, at last, on a retrieval metric rather than chosen because they seemed
# reasonable. `evals/retrieval.py` asks a section's own heading and looks for its
# body, with the heading stripped from what is indexed -- a human-written heading
# is a compressed paraphrase of the section below it, so the query overlaps its
# answer partly in wording and partly only in meaning.
#
# Measured on two independent corpora, ten repositories and seven, 208 and 251
# queries. Both peak at the same point and both beat the previous values:
#
#                      corpus 1   corpus 2
#   bm25 only            0.374      0.461
#   char only            0.324        --
#   embeddings only      0.346      0.481
#   0.40 / 0.22 / 0.38   0.395      0.495     (previous)
#   0.20 / 0.20 / 0.60   0.406      0.526     (fitted)
#
# Fusion beats every single ranker on both, which is the first direct evidence
# that the three-ranker design earns its complexity. The band is flat for
# EMBED_WEIGHT between roughly 0.4 and 0.7 and falls away at 0.8, so 0.6 is the
# middle of a real region rather than a sharp peak.
BM25_WEIGHT = 0.20
CHAR_WEIGHT = 0.20
EMBED_WEIGHT = 0.60
RECENCY_WEIGHT = 0.05
LENGTH_SATURATION = 260
SCORE_FLOOR = 0.12
MIN_GROUNDED = 2
MIN_GROUNDED_FRAC = 0.25
SEMANTIC_FLOOR = 0.57
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 20

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
STUB_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\[\[.*\]\]|#[\w/]+|Category:.*|\|.*\|)\s*$")
ANY_LINK = re.compile(r"\[\[[^\]]+\]\]|\[[^\]]+\]\([^)]+\)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_FIT: dict[str, tuple] = {}
_EMB: dict | None = None

# `ask` fans out across the selected scopes in a ThreadPoolExecutor, so every
# lazy initializer below is reachable from several threads at once.
#
# Historically that DEADLOCKED. Unguarded, `if _MODEL is None: _MODEL =
# SentenceTransformer(...)` was a check-then-act race, and two threads
# constructing a sentence-transformer concurrently did not merely duplicate the
# work -- they hung, because the loader spun up a joblib/loky process pool and
# doing that from two threads blocks. Measured: `loci ask "which of my projects
# use Cloudflare workers or D1?"` never returned, while the same question with
# the model already warm took 0.2s, and the same question with one scope
# selected always worked. Enumerative set mode selects two or more scopes far
# more often than anything else, so a latent race became the first thing a user
# would run.
#
# No model is built here any more -- `embed` owns that, and onnxruntime has no
# process pool to fork, which is what made concurrent construction a hang
# rather than merely wasted work. The history stays because it is why the
# fan-out is written the way it is, and because the locks that remain guard
# DATA rather than a loader.
#
# Separate locks, not one: `_semantic` calls `_embeddings()` and then
# `_encode_query()`, and a single shared lock would serialize every query behind
# whichever thread is loading. Double-checked, so the lock is paid once.
_EMB_LOCK = threading.Lock()


# ==========================================================================
# chunking
# ==========================================================================
def is_stub(body: str) -> bool:
    """True for chunks that are only wikilinks, tags or `Category:` frontmatter."""
    words = 0
    for line in body.splitlines():
        if not line.strip() or STUB_LINE.match(line):
            continue
        words += len(line.split())
    return words < MIN_CONTENT_WORDS


def is_navigation(body: str) -> bool:
    """True for a table of contents: mostly lines that link elsewhere.

    Measured: a bi-encoder AND a cross-encoder independently ranked a vault's
    `Project Index > Critical Context Links` above the note actually holding the
    risks. Neither is wrong -- a contents page genuinely IS topically about
    risks -- so this is an indexing problem no reranker can fix. Two models
    agreeing a bad candidate looks good means it should not be a candidate.

    Per-chunk, not per-file: the same index note usually also has a real summary
    section that must survive. No minimum line count either -- a one-line
    bullet holding a single wikilink plus a gloss is still a pointer.
    """
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return False
    return sum(1 for ln in lines if ANY_LINK.search(ln)) / len(lines) >= NAV_LINK_RATIO


def pack(body: str) -> list[str]:
    """Pack paragraphs into ~TARGET_CHUNK_CHARS windows with a small overlap."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    units: list[str] = []
    for para in paras:
        if len(para) <= HARD_MAX_CHARS:
            units.append(para)
            continue
        buf = ""
        for sent in SENTENCE_END.split(para):
            if buf and len(buf) + len(sent) + 1 > HARD_MAX_CHARS:
                units.append(buf.strip())
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf.strip():
            units.append(buf.strip())

    out: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > TARGET_CHUNK_CHARS:
            out.append(cur)
            tail = cur[-OVERLAP_CHARS:]
            cut = tail.find(" ")
            cur = (tail[cut + 1:] + "\n\n" + u) if cut > 0 else u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def chunk_markdown(text: str, source: str, kind: str, ts: str) -> list[Chunk]:
    """Split on headings, carrying the heading path into each chunk."""
    chunks: list[Chunk] = []
    path: list[str] = []
    cur: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < MIN_CHUNK_CHARS or is_stub(body) or is_navigation(body):
            return
        for piece in pack(body):
            if len(piece) < MIN_CHUNK_CHARS or is_stub(piece) or is_navigation(piece):
                continue
            chunks.append(_chunk(kind, source, " > ".join(cur), piece, ts))

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            depth = len(m.group(1))
            path = path[:depth - 1] + [m.group(2).strip()]
            cur = list(path)
        else:
            buf.append(line)
    flush()
    return chunks


# ==========================================================================
# collection
# ==========================================================================
_REDACTIONS: Counter = Counter()


def _chunk(kind: str, source: str, heading: str, text: str, ts: str) -> Chunk:
    """Build a chunk with its text redacted. The only constructor collectors use.

    Funnelled through one function on purpose: there are four collection paths
    and a fifth is easy to add, so redaction lives where a new path cannot avoid
    it rather than being repeated at each call site.
    """
    clean, found = redact(text)
    if found:
        _REDACTIONS.update(found)
    head, head_found = redact(heading)
    if head_found:
        _REDACTIONS.update(head_found)
    return Chunk(kind=kind, source=source, heading=head, text=clean, ts=ts)


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _mtime(p: Path) -> str:
    try:
        return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()
    except Exception:
        return ""


def collect_code(scope: Scope, *, exclude=()) -> list[Chunk]:
    """Mine docstrings and comment blocks from the scope's source files."""
    from .docstrings import extract

    from ..walk import iter_files

    out: list[Chunk] = []
    files = 0
    for f in iter_files(scope.root, scope.code_globs or [], exclude=exclude):
        if files >= MAX_CODE_FILES:
            return out
        if is_sensitive_file(f):
            continue
        files += 1
        try:
            rel = str(f.relative_to(scope.root))
        except ValueError:
            rel = f.name
        ts = _mtime(f)
        for heading, body in extract(f, rel):
            out.append(_chunk("docstring", f"code:{rel}", heading, body, ts))
    return out


def collect_git(root: Path) -> list[Chunk]:
    """Commit messages are episodes: timestamped, authored, and always present.

    This is the cold-start answer. A freshly registered scope has no vault notes
    and possibly no docs, but it has history from day one.
    """
    if not (root / ".git").is_dir():
        return []
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", f"-{MAX_COMMITS}",
             "--pretty=format:%H%x1f%aI%x1f%s%x1f%b%x1e"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return []
    chunks: list[Chunk] = []
    for rec in out.split("\x1e"):
        parts = rec.strip("\n").split("\x1f")
        if len(parts) < 3 or not parts[0]:
            continue
        sha, ts, subject = parts[0], parts[1], parts[2]
        body = parts[3] if len(parts) > 3 else ""
        text = f"{subject}\n{body}".strip() or subject
        chunks.append(_chunk("commit", f"git:{sha[:10]}", subject,
                             text[:HARD_MAX_CHARS], ts))
    return chunks


class BuiltinEpisodeBackend:
    name = "builtin"

    def collect(self, scope: Scope, *, exclude=()) -> list[Chunk]:
        _REDACTIONS.clear()
        chunks: list[Chunk] = []
        seen: set[Path] = set()
        chunks += collect_code(scope, exclude=exclude)
        from ..walk import iter_files
        # An absolute glob lets a scope pull in prose that lives outside the
        # repo -- an external vault, a notes directory, a host app's store.
        for f in iter_files(scope.root, scope.episode_globs or [], exclude=exclude):
            if f in seen or is_sensitive_file(f):
                continue
            seen.add(f)
            kind = "doc" if f.suffix.lower() in {".md", ".markdown", ".rst", ".txt"} else "note"
            try:
                label = str(f.relative_to(scope.root))
            except ValueError:
                label = f.name
            chunks += chunk_markdown(_read(f), f"{kind}:{label}", kind, _mtime(f))
        chunks += collect_git(scope.root)
        return chunks

    def redactions(self) -> dict[str, int]:
        """What the last `collect` removed, by kind."""
        return dict(_REDACTIONS)

    def save_rankers(self, scope_id: str, chunks: list[Chunk]) -> None:
        save_rankers(scope_id, chunks)

    def vocabulary(self, chunks: list[Chunk]) -> Counter:
        counts: Counter = Counter()
        for c in chunks:
            counts.update(token_set(f"{c.heading} {c.text}"))
        return counts

    def search(self, question: str, chunks: list[Chunk], scope_id: str, *,
               k: int = 5, rerank: bool = False,
               score_floor: float | None = None,
               min_grounded: int | None = None,
               min_grounded_frac: float | None = None,
               semantic_floor: float | None = None,
               rerank_depth: int | None = None,
               gate: bool = True) -> list[EpisodeHit]:
        # Resolved here rather than in the signature: a default argument is
        # bound at import time, so sweeping a module constant would not reach a
        # caller that relies on the default.
        score_floor = SCORE_FLOOR if score_floor is None else score_floor
        min_grounded = MIN_GROUNDED if min_grounded is None else min_grounded
        min_grounded_frac = (MIN_GROUNDED_FRAC if min_grounded_frac is None
                             else min_grounded_frac)
        semantic_floor = (_calibrated_semantic_floor(scope_id)
                          if semantic_floor is None else semantic_floor)
        rerank_depth = RERANK_DEPTH if rerank_depth is None else rerank_depth
        if not chunks:
            return []
        lex = _fit(scope_id, chunks)
        q_tokens = vtokens(question)
        if not q_tokens or lex is None:
            return []

        sem = _semantic(question, scope_id, len(chunks))
        now = _now()

        # The gate, the fusion and the sort are Rust; every constant crosses as
        # an argument rather than being compiled in, because they are
        # calibrated -- several per corpus -- and `evals/` sweeps them. A sweep
        # must not need a rebuild.
        #
        # Recency is computed HERE, so `$LOCI_NOW` keeps working and no date
        # parsing crosses the boundary. Query tokens are NOT deduplicated:
        # rank_bm25 iterates the raw list, so a repeat counts twice.
        scored = lex.search(
            question,
            q_tokens,
            [float(x) for x in sem] if sem is not None else None,
            [len(c.text) for c in chunks],
            [_recency(c.ts, now) for c in chunks],
            (BM25_WEIGHT, CHAR_WEIGHT, EMBED_WEIGHT, RECENCY_WEIGHT),
            float(LENGTH_SATURATION),
            float(score_floor),
            int(min_grounded),
            float(min_grounded_frac),
            float(semantic_floor),
            bool(gate),
        )

        depth = max(k, rerank_depth) if rerank else k
        hits = [EpisodeHit(chunk=chunks[i], score=s) for i, s in scored[:depth]]
        if rerank and len(hits) > 1:
            hits = _rerank(question, hits, rerank_depth)
        return hits[:k]


# ==========================================================================
# ranking internals
# ==========================================================================
def _texts(chunks: list[Chunk]) -> tuple[list[str], list[list[str]]]:
    texts = [f"{c.heading} {c.text}" for c in chunks]
    return texts, [vtokens(t) for t in texts]


def fit(chunks: list[Chunk]):
    """Build the lexical rankers for one scope, in memory. See `_fit`.

    Rust, and no file: a query that finds no cached ranker must not write to
    the rankers directory as a side effect of being asked.
    """
    from loci._core import Lex

    texts, tokenized = _texts(chunks)
    return Lex.fit(texts, tokenized)


def save_rankers(scope_id: str, chunks: list[Chunk]) -> None:
    """Persist one scope's fitted rankers so queries never refit.

    Fitting char 3-5 gram TF-IDF over a large scope costs ~1.5s. The in-process
    cache below never survives a CLI or MCP invocation, so without this every
    single question paid that cost -- measured at 4.6-6.1s per query end to
    end, which is disqualifying for a tool an agent calls in a loop.

    The format is `.lex` rather than joblib, and that is the other half: a
    35MB pickle cost 1.046s to LOAD, every time, and the same data mmapped
    opens in microseconds.
    """
    from loci._core import lex_fit_write

    d = rankers_dir()
    d.mkdir(parents=True, exist_ok=True)
    texts, tokenized = _texts(chunks)
    atomic_write_via(d / f"{scope_id}{LEX_SUFFIX}",
                     lambda tmp: lex_fit_write(str(tmp), texts, tokenized))


def _load_rankers(scope_id: str, n_chunks: int):
    """Return the persisted rankers, or None when absent or stale.

    The document-count check lives in the Rust reader now, along with the
    checks for a truncated file, a foreign one, and a version this build does
    not read. All of them refuse rather than misread, for the reason this
    guard has always existed: a misaligned ranking is worse than none.
    """
    from loci._core import Lex

    return Lex.open(str(rankers_dir() / f"{scope_id}{LEX_SUFFIX}"), n_chunks)


def _fit(scope_id: str, chunks: list[Chunk]):
    if scope_id in _FIT:
        return _FIT[scope_id]
    cached = _load_rankers(scope_id, len(chunks))
    if cached is not None:
        _FIT[scope_id] = cached
        return cached
    _FIT[scope_id] = fit(chunks)
    return _FIT[scope_id]


def _calibrated_semantic_floor(scope_id: str | None = None) -> float:
    """The cosine floor fitted to this corpus, or the shipped default.

    The default was read off one corpus with one embedding model. Cosine scale
    belongs to the model and the size of the related/unrelated gap belongs to
    the corpus, so a number from one pairing describes neither in general.
    """
    try:
        from ..calibrate import load
        cal = load()
    except Exception:
        cal = None
    if not cal or not cal.semantic_n:
        return SEMANTIC_FLOOR
    per = getattr(cal, "semantic_by_scope", None) or {}
    return per.get(scope_id, cal.semantic_floor)


def _embeddings(path: Path | None = None):
    global _EMB
    if _EMB is None:
        with _EMB_LOCK:
            if _EMB is None:          # another thread may have won the race
                p = path or embeddings_file()
                if not Path(p).is_file():
                    _EMB = {}
                else:
                    import numpy as np
                    z = np.load(p, allow_pickle=False)
                    _EMB = {k: z[k] for k in z.files}
    return _EMB or None


def _semantic(question: str, scope_id: str, n_chunks: int):
    emb = _embeddings()
    if not emb:
        return None
    E = emb.get(scope_id)
    if E is None or E.shape[0] != n_chunks:
        return None  # stale cache: fall back to lexical rather than misalign
    qv = _encode_query(question, str(emb["_model"][0]) if "_model" in emb else "")
    return (E @ qv) if qv is not None else None


def _encode_query(text: str, model_name: str):
    """The query vector, or None when no model is configured.

    No cached model object and no lock here any more: `embed` owns session
    construction and its own double-checked lock, so this module's three locks
    are down to the one that guards data.
    """
    if not model_name:
        return None
    from .. import embed
    if not embed.available():
        # Vectors on disk but no encoder installed. Degrade to lexical, which
        # is what `_semantic` already does for absent or stale vectors -- the
        # alternative is dying inside `loci ask` with an ImportError, which is
        # what both 0.5.0 and 0.6.0 did until this. `doctor` reports it, so it
        # is quiet rather than silent.
        return None
    return embed.encode([text], model_name=model_name, is_query=True)[0]


def _rerank(question: str, hits: list[EpisodeHit], depth: int) -> list[EpisodeHit]:
    """Cross-encoder rerank of the head. Opt-in; see README for the numbers."""
    from .. import embed

    head, tail = hits[:depth], hits[depth:]
    if len(head) < 2:
        return hits
    scores = embed.rerank_scores(
        [(question, f"{h.chunk.heading} {h.chunk.text}") for h in head],
        model_name=RERANK_MODEL)
    for h, sc in zip(head, scores):
        h.rerank_score = float(sc)
    head.sort(key=lambda h: -(h.rerank_score or 0.0))
    return head + tail


def _now() -> datetime:
    """Scoring time, pinnable so a comparison is against ranking, not the clock.

    RECENCY_WEIGHT folds elapsed time into the score of every chunk carrying a
    timestamp, so two runs of one question over one unchanged corpus disagree
    by however much the clock moved between them. That is correct ranking
    behaviour, and it makes a RECORDED answer decay: measured at roughly 1e-4
    of score per hour for a month-old chunk -- enough to move the 4-decimal
    value `EpisodeHit.to_json` publishes -- and it grows without bound as the
    recording ages. Left alone, two hits decaying at different rates eventually
    swap, and a parity harness reports a ranking regression that no code change
    caused.

    $LOCI_NOW pins the instant. Unset -- which is everywhere except the parity
    harness -- behaviour is exactly as before.
    """
    raw = os.environ.get("LOCI_NOW")
    if raw:
        try:
            t = datetime.fromisoformat(raw)
            return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
        except ValueError:
            pass          # a malformed pin must not silently freeze the clock
    return datetime.now(timezone.utc)


def _recency(ts: str, now: datetime) -> float:
    if not ts:
        return 0.0
    try:
        t = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - t).total_seconds() / 86400)
    return 1.0 / (1.0 + math.log1p(days / 30))


def warm_up(store: dict | None = None, scope_ids: list[str] | None = None) -> None:
    """Pay the one-off costs before a user is waiting on them.

    Measured cold under torch: the first query in a process cost ~4.4s and
    every one after it 0.02-0.5s, almost all of it importing
    sentence_transformers (1.6s) and sklearn (0.6s) plus constructing the
    model -- process startup, not work.

    Under onnxruntime the same one-shot ask runs 1.9s against that 5.2s, so
    the cost this exists to hide is a fraction of what it was. It still earns
    its place: a long-lived server should absorb what remains at boot, and a
    one-shot CLI invocation still has no way to avoid paying it per question.
    """
    emb = _embeddings()
    if emb:
        try:
            _encode_query("warm up", str(emb["_model"][0]) if "_model" in emb else "")
        except Exception:
            pass
    if store and scope_ids:
        from ..index import chunks_for
        for sid in scope_ids:
            chunks = chunks_for(store, sid)
            if chunks:
                _fit(sid, chunks)


def reset_caches() -> None:
    """Drop fitted rankers and vectors. For tests and after reindexing."""
    global _EMB
    _FIT.clear()
    _EMB = None
