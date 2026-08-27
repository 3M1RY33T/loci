"""Backend contracts."""
from __future__ import annotations

from collections import Counter
from typing import Protocol, runtime_checkable

from ..types import Chunk, EpisodeHit, Scope, StructureHit


@runtime_checkable
class StructureBackend(Protocol):
    """A store of code structure: symbols, references, call/import edges."""

    name: str

    def available(self) -> bool:
        """False when the backing tool or index is not installed."""

    def sources(self, scope: Scope) -> list[dict]:
        """Indexes this backend can query for `scope`; empty if unindexed."""

    def vocabulary(self, scope: Scope) -> Counter:
        """token -> number of nodes containing it. Feeds the router."""

    def query(self, scope: Scope, query: str, *, budget: int = 2000,
              dfs: bool = False) -> list[StructureHit]:
        """Traverse within one scope. Never across scopes."""


@runtime_checkable
class EpisodeBackend(Protocol):
    """A store of verbatim content: notes, sessions, commits, docs."""

    name: str

    def collect(self, scope: Scope) -> list[Chunk]:
        """Read the scope's prose and chunk it. Verbatim -- never summarized."""

    def vocabulary(self, chunks: list[Chunk]) -> Counter:
        """token -> number of chunks containing it."""

    def search(self, question: str, chunks: list[Chunk], scope_id: str, *,
               k: int = 5, rerank: bool = False) -> list[EpisodeHit]:
        """Rank within one scope.

        MUST be able to return an empty list. An episode store that always finds
        something cannot be trusted when it does -- see the two-tier gate in the
        builtin backend for why an absolute score floor is not enough once a
        semantic ranker is involved.
        """
