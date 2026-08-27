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
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ..paths import atomic_write_via, embeddings_file, rankers_dir
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
BM25_WEIGHT = 0.40
CHAR_WEIGHT = 0.22
EMBED_WEIGHT = 0.38
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

_QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: "}
_FIT: dict[str, tuple] = {}
_EMB: dict | None = None
_MODEL = None
_RERANKER = None


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


def collect_code(scope: Scope) -> list[Chunk]:
    """Mine docstrings and comment blocks from the scope's source files."""
    from .docstrings import extract

    from ..walk import iter_files

    out: list[Chunk] = []
    files = 0
    for f in iter_files(scope.root, scope.code_globs or []):
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

    def collect(self, scope: Scope) -> list[Chunk]:
        _REDACTIONS.clear()
        chunks: list[Chunk] = []
        seen: set[Path] = set()
        chunks += collect_code(scope)
        from ..walk import iter_files
        # An absolute glob lets a scope pull in prose that lives outside the
        # repo -- an external vault, a notes directory, a host app's store.
        for f in iter_files(scope.root, scope.episode_globs or []):
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
               score_floor: float = SCORE_FLOOR,
               min_grounded: int = MIN_GROUNDED,
               min_grounded_frac: float = MIN_GROUNDED_FRAC,
               semantic_floor: float = SEMANTIC_FLOOR,
               rerank_depth: int = RERANK_DEPTH,
               gate: bool = True) -> list[EpisodeHit]:
        if not chunks:
            return []
        bm25, vec, mat, vocab = _fit(scope_id, chunks)
        q_tokens = vtokens(question)
        if not q_tokens:
            return []

        grounded = sum(1 for t in dict.fromkeys(q_tokens) if t in vocab)
        need = max(min_grounded, int(round(min_grounded_frac * len(set(q_tokens)))))

        bm = bm25.get_scores(q_tokens) if bm25 else [0.0] * len(chunks)
        bm_max = max(bm) if len(bm) else 0.0
        bm_norm = [(s / bm_max) if bm_max > 0 else 0.0 for s in bm]

        if vec is not None:
            from sklearn.metrics.pairwise import linear_kernel
            ch = linear_kernel(vec.transform([question]), mat).ravel()
        else:
            ch = [0.0] * len(chunks)

        sem = _semantic(question, scope_id, len(chunks))

        # Two-tier relevance gate, before any ranking is trusted. An absolute
        # score floor stops working once a semantic ranker gives every chunk a
        # nonzero score -- measured, "the airspeed velocity of an unladen
        # swallow" scored 0.608 against a large scope with a HIGHER
        # top-to-p90 ratio than a genuine question against a small one, so
        # neither an absolute floor nor a distribution test separates them.
        # Lexical grounding does. But grounding alone would reject exactly the
        # questions embeddings were added for, so: grounded OR confident.
        # `gate=False` is for a scope the caller named outright. The gate's job
        # is to stop a question being answered from the WRONG scope; once the
        # user has picked one, that risk is gone and the only remaining risk is
        # a weak answer, which its score already reports. Measured: a 2-chunk
        # scope scores 0.535 on "what does this project do?" -- inside the
        # nonsense band (0.46-0.52), so the floor is right and must not move,
        # but returning nothing to someone standing in that project is not.
        semantic_ok = sem is not None and float(sem.max()) >= semantic_floor
        if gate and grounded < need and not semantic_ok:
            return []

        # Renormalize over whichever rankers actually produced a signal.
        #
        # BM25 needs the second guard as much as embeddings need the first:
        # Okapi IDF is log((N - df + 0.5) / (df + 0.5)), which is exactly 0 for
        # a term appearing in half the corpus. In a 2-chunk scope every query
        # term hits that case, BM25 returns all zeros, and left in the fusion it
        # drags the surviving rankers below the score floor -- so a thin scope
        # silently answers nothing rather than answering from what little it has.
        w_bm, w_ch, w_em = BM25_WEIGHT, CHAR_WEIGHT, EMBED_WEIGHT
        if sem is None:
            w_em = 0.0
        if bm_max <= 0:
            w_bm = 0.0
        tot = w_bm + w_ch + w_em
        if tot <= 0:
            return []
        w_bm, w_ch, w_em = w_bm / tot, w_ch / tot, w_em / tot

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, int]] = []
        for i, c in enumerate(chunks):
            s = w_bm * bm_norm[i] + w_ch * float(ch[i])
            if sem is not None:
                s += w_em * max(0.0, float(sem[i]))
            # Every ranker rewards brevity -- BM25 by length normalization,
            # cosine because a short vector is dominated by the query terms. A
            # one-line chunk echoing the question is not the answer.
            s *= min(1.0, len(c.text) / LENGTH_SATURATION)
            s += RECENCY_WEIGHT * _recency(c.ts, now) * (1.0 if s > 0 else 0.0)
            if s >= score_floor:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])

        depth = max(k, rerank_depth) if rerank else k
        hits = [EpisodeHit(chunk=chunks[i], score=s) for s, i in scored[:depth]]
        if rerank and len(hits) > 1:
            hits = _rerank(question, hits, rerank_depth)
        return hits[:k]


# ==========================================================================
# ranking internals
# ==========================================================================
def fit(chunks: list[Chunk]):
    """Build the lexical rankers for one scope. Expensive; see `_fit`."""
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [f"{c.heading} {c.text}" for c in chunks]
    tokenized = [vtokens(t) for t in texts]
    vocab: set[str] = set()
    for toks in tokenized:
        vocab.update(toks)
    bm25 = BM25Okapi(tokenized) if texts else None
    vec = mat = None
    if texts:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1,
                              max_features=60000, lowercase=True)
        mat = vec.fit_transform(texts)
    return bm25, vec, mat, vocab


def save_rankers(scope_id: str, chunks: list[Chunk]) -> None:
    """Persist one scope's fitted rankers so queries never refit.

    Fitting char 3-5 gram TF-IDF over a large scope costs ~1.5s. The in-process
    cache below never survives a CLI or MCP invocation, so without this every
    single question paid that cost -- measured at 4.6-6.1s per query end to end,
    which is disqualifying for a tool an agent calls in a loop.
    """
    import joblib

    d = rankers_dir()
    d.mkdir(parents=True, exist_ok=True)
    bm25, vec, mat, vocab = fit(chunks)
    blob = {"n": len(chunks), "bm25": bm25, "vec": vec, "mat": mat, "vocab": vocab}
    atomic_write_via(d / f"{scope_id}.joblib",
                     lambda tmp: joblib.dump(blob, tmp, compress=3))


def _load_rankers(scope_id: str, n_chunks: int):
    """Return persisted rankers, or None when absent or stale."""
    f = rankers_dir() / f"{scope_id}.joblib"
    if not f.is_file():
        return None
    try:
        import joblib
        blob = joblib.load(f)
    except Exception:
        return None
    if blob.get("n") != n_chunks:
        return None  # store changed under us; refit rather than misalign
    return blob["bm25"], blob["vec"], blob["mat"], blob["vocab"]


def _fit(scope_id: str, chunks: list[Chunk]):
    if scope_id in _FIT:
        return _FIT[scope_id]
    cached = _load_rankers(scope_id, len(chunks))
    if cached is not None:
        _FIT[scope_id] = cached
        return cached
    _FIT[scope_id] = fit(chunks)
    return _FIT[scope_id]


def _embeddings(path: Path | None = None):
    global _EMB
    if _EMB is None:
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
    global _MODEL
    if not model_name:
        return None
    if _MODEL is None:
        import warnings
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(model_name)
    prefix = next((v for kk, v in _QUERY_PREFIX.items() if kk in model_name.lower()), "")
    return _MODEL.encode([prefix + text], normalize_embeddings=True)[0]


def _rerank(question: str, hits: list[EpisodeHit], depth: int) -> list[EpisodeHit]:
    """Cross-encoder rerank of the head. Opt-in; see README for the numbers."""
    global _RERANKER
    head, tail = hits[:depth], hits[depth:]
    if len(head) < 2:
        return hits
    if _RERANKER is None:
        import warnings
        warnings.filterwarnings("ignore")
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANK_MODEL, max_length=512)
    scores = _RERANKER.predict(
        [(question, f"{h.chunk.heading} {h.chunk.text}") for h in head],
        show_progress_bar=False)
    for h, sc in zip(head, scores):
        h.rerank_score = float(sc)
    head.sort(key=lambda h: -(h.rerank_score or 0.0))
    return head + tail


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

    Measured cold: the first query in a process costs ~4.4s and every one after
    it costs 0.02-0.5s. Almost all of that is importing sentence_transformers
    (1.6s) and sklearn (0.6s) plus constructing the embedding model -- process
    startup, not work. A long-lived server should absorb it at boot; only a
    one-shot CLI invocation has no way to avoid paying it per question.
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
