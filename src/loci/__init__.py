"""loci -- scoped memory for coding agents.

Two stores and a router:

    structure store   what calls what          (graphify adapter by default)
    episode store     what happened and why    (verbatim prose, BM25+ngram+embed)
    router            which scope to ask       (deterministic, abstains)

The router is the part that does not exist elsewhere. Namespaced memory systems
ask you to name the namespace when you WRITE. loci decides it when you READ,
so the same question path serves "how does auth work here" and "have I solved
this in any project".
"""
from __future__ import annotations

__version__ = "0.2.0"

from .types import Chunk, EpisodeHit, RouteResult, Scope, StructureHit

__all__ = [
    "__version__",
    "Chunk", "EpisodeHit", "RouteResult", "Scope", "StructureHit",
    "route", "ask", "load_scopes",
]


def __getattr__(name: str):
    # Deferred so `import loci` stays cheap; the CLI and MCP server both
    # import this module before doing any work.
    if name == "route":
        from .router import route
        return route
    if name == "ask":
        from .ask import ask
        return ask
    if name == "load_scopes":
        from .scopes import load_scopes
        return load_scopes
    raise AttributeError(name)
