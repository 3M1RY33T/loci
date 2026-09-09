"""The only place an embedding model is constructed.

Before this module there were three: `index.build_embeddings`,
`episodes._encode_query` and `episodes._rerank`, each with its own lazy import,
its own lock and its own `warnings.filterwarnings("ignore")`. Consolidating is
what makes the torch removal one change rather than three, and it is the seam
the Rust port later replaces.

torch is gone for three measured reasons, not for taste:

  1. 6.2s of a 7.77s `loci ask` was torch loading bge-small.
  2. `ask.py` fans out across scopes in a ThreadPoolExecutor and every thread
     reached torch. episodes.py already added three locks for a diagnosed
     deadlock there, and Delroy measured a SEGFAULT after that fix -- so the
     class of bug was never the check-then-act race, it was native threading.
  3. The loader drew a progress bar on stderr on every invocation, which
     Delroy strips with a regex before showing loci's errors to a model.

What must be reproduced exactly, or ~/.loci/calibration.json is invalid:
pooling mode (read from the model, never assumed), L2 normalisation after
pooling, the query prefix on queries only, and truncation at the same window.

Repo layout, probed 2026-09-09 -- both publish ONNX directly, so nothing here
exports with `optimum`:

    BAAI/bge-small-en-v1.5                onnx/model.onnx, tokenizer.json,
                                          1_Pooling/config.json
                                          pooling_mode_cls_token = True
                                          word_embedding_dimension = 384
                                          max_seq_length = 512
                                          modules: Transformer, Pooling, Normalize

    cross-encoder/ms-marco-MiniLM-L-6-v2  onnx/model.onnx, tokenizer.json
                                          num_labels = 1
                                          BertForSequenceClassification
                                          no 1_Pooling -- it is a classifier

That repo also ships model_O1..O4 and qint8 variants. They are deliberately
NOT used: an optimised or quantised graph drifts from the torch weights the
floors were fitted against, which is the one thing this module may not do.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

# bge asks for a prefix on the QUERY side only; indexed text gets none. Moved
# here from episodes._QUERY_PREFIX so both callers read one definition.
QUERY_PREFIX = {"bge": "Represent this sentence for searching relevant passages: "}

MAX_LENGTH = 512          # bge-small's window; sentence-transformers truncates here too

_SESSIONS: dict[str, tuple] = {}
_LOCK = threading.Lock()


def _pooling_mode(repo: str) -> str:
    """CLS or mean, read from the model rather than assumed.

    Guessing changes every vector by a little and the calibrated floors by
    enough to matter, while breaking nothing loudly enough to notice. bge-small
    is CLS; a mean-pooled model substituted later would be read correctly here
    rather than silently mis-encoded.
    """
    from huggingface_hub import hf_hub_download
    try:
        cfg = hf_hub_download(repo, "1_Pooling/config.json")
        d = json.loads(Path(cfg).read_text(encoding="utf-8"))
    except Exception:
        # The READ is inside the guard too. A repo with no 1_Pooling (every
        # cross-encoder) fails at the download, but a cached-but-unreadable
        # file failed at the read, outside it, and crashed the encoder rather
        # than falling back.
        return "cls"      # bge's published default
    if d.get("pooling_mode_cls_token"):
        return "cls"
    if d.get("pooling_mode_mean_tokens"):
        return "mean"
    return "cls"


def _session(repo: str, *, pooled: bool = True):
    """(InferenceSession, Tokenizer, pooling_mode), constructed once per repo.

    Double-checked lock, so the cost is paid once and the fan-out cannot build
    two. Unlike the sentence-transformers loader this replaces, onnxruntime has
    no process pool to fork, which is what made the concurrent construction a
    deadlock rather than merely duplicated work.
    """
    if repo in _SESSIONS:
        return _SESSIONS[repo]
    with _LOCK:
        if repo in _SESSIONS:          # another thread may have won the race
            return _SESSIONS[repo]
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(repo, "onnx/model.onnx")
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_LENGTH)
        tok.enable_padding(length=None)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3          # no banner on stderr, ever
        sess = ort.InferenceSession(model_path, opts,
                                    providers=["CPUExecutionProvider"])
        _SESSIONS[repo] = (sess, tok, _pooling_mode(repo) if pooled else "")
    return _SESSIONS[repo]


def _feed(sess, tok, encoded) -> dict:
    """Only the inputs this graph declares.

    bge-small takes token_type_ids and some exports do not; feeding an input
    the graph never declared is a hard onnxruntime error rather than an
    ignored key.
    """
    ids = np.array([e.ids for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    types = np.array([e.type_ids for e in encoded], dtype=np.int64)
    names = {i.name for i in sess.get_inputs()}
    feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
    return {k: v for k, v in feed.items() if k in names}


def encode(texts: list[str], *, model_name: str, is_query: bool = False,
           batch_size: int = 64) -> np.ndarray:
    """L2-normalised float32 embeddings, one row per text.

    Normalised at both build and query time so scoring stays a plain dot
    product -- which is what `episodes._semantic` already assumes when it
    writes `E @ qv`.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    prefix = ""
    if is_query:
        low = model_name.lower()
        prefix = next((v for k, v in QUERY_PREFIX.items() if k in low), "")

    sess, tok, mode = _session(model_name)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        feed = _feed(sess, tok, tok.encode_batch(batch))
        hidden = sess.run(None, feed)[0]                       # (b, seq, dim)
        if mode == "cls":
            vec = hidden[:, 0, :]
        else:
            mask = feed["attention_mask"][:, :, None].astype("float32")
            vec = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norm = np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
        out.append((vec / norm).astype("float32"))
    return np.vstack(out)


def rerank_scores(pairs: list[tuple[str, str]], *, model_name: str) -> list[float]:
    """Cross-encoder logits, one per (query, passage) pair.

    A raw logit, not a probability: `episodes._rerank` only sorts by it, and
    passing it through a sigmoid would change nothing but would differ from
    what CrossEncoder.predict returns, which the parity test compares against.
    """
    if not pairs:
        return []
    sess, tok, _ = _session(model_name, pooled=False)
    encoded = tok.encode_batch([list(p) for p in pairs])
    logits = sess.run(None, _feed(sess, tok, encoded))[0]
    return [float(x) for x in (logits[:, 0] if logits.ndim == 2 else logits.ravel())]


def reset_sessions() -> None:
    """Drop loaded sessions. For tests."""
    _SESSIONS.clear()
