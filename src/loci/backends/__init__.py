"""Pluggable stores.

Two contracts, in `base`:

    StructureBackend   what calls what -- symbols, references, traversals
    EpisodeBackend     what happened and why -- verbatim prose

Nothing above this package knows which implementation is in use. That is why
loci depends on graphify rather than forking it: the entire integration is
`GraphifyBackend`, reading a documented data contract (graph.json) and shelling
out to one CLI verb. Swapping in tree-sitter, SCIP or an LSP index means writing
one adapter, not editing the router.
"""
from __future__ import annotations

from .base import EpisodeBackend, StructureBackend

__all__ = ["EpisodeBackend", "StructureBackend", "get_structure_backend",
           "get_episode_backend"]


def get_structure_backend(name: str = "graphify"):
    if name == "graphify":
        from .graphify import GraphifyBackend
        return GraphifyBackend()
    raise ValueError(f"unknown structure backend: {name!r}")


def get_episode_backend(name: str = "builtin"):
    if name == "builtin":
        from .episodes import BuiltinEpisodeBackend
        return BuiltinEpisodeBackend()
    raise ValueError(f"unknown episode backend: {name!r}")
