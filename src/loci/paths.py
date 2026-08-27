"""Where loci keeps its own artifacts.

Index artifacts live in ONE place, not per-project, because routing is a
cross-scope decision: answering "which project is this question about" requires
every scope's vocabulary in the same lookup. Per-project indexes would make the
cheap question (one dict lookup) into the expensive one (open N indexes).
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "LOCI_HOME"


def home() -> Path:
    """Data root. ``$LOCI_HOME`` wins; otherwise ``~/.loci``."""
    raw = os.environ.get(ENV_HOME)
    if raw:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            raise ValueError(f"{ENV_HOME} must be an absolute path, got {raw!r}")
        return p
    return Path.home() / ".loci"


def ensure_home() -> Path:
    p = home()
    p.mkdir(parents=True, exist_ok=True)
    return p


def registry_file() -> Path:
    """The scope registry.

    JSON rather than TOML on purpose: ``tomllib`` is read-only and 3.11+, and
    hand-rolling a TOML *writer* is a known way to produce files the parser then
    rejects. One machine-managed file, human-editable, no encoder to get wrong.
    """
    return home() / "scopes.json"


def scope_index_file() -> Path:
    return home() / "scope_index.json"


def episode_store_file() -> Path:
    return home() / "episodes.json"


def embeddings_file() -> Path:
    return home() / "embeddings.npz"
