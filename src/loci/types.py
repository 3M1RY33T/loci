"""Core data types.

A Scope is the unit everything else is relative to: one project, one namespace.
It owns a structure store (what calls what) and an episode store (what happened
and why). Both are scoped; neither is ever merged across scopes -- a single
merged index makes the largest corpus win regardless of the question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Scope:
    """One project / namespace."""

    id: str
    name: str
    root: Path
    aliases: list[str] = field(default_factory=list)
    episode_globs: list[str] = field(default_factory=list)
    code_globs: list[str] = field(default_factory=list)
    updated_at: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    groups: list[str] | None = None

    def group_set(self) -> set[str]:
        """Membership as a set. `None` and `[]` both mean "belongs to nothing"."""
        return set(self.groups or [])

    def to_json(self) -> dict:
        d = {
            "id": self.id, "name": self.name, "root": str(self.root),
            "aliases": self.aliases, "episode_globs": self.episode_globs,
            "code_globs": self.code_globs,
            "updated_at": self.updated_at, "meta": self.meta,
        }
        # ABSENT means "not yet inferred"; an explicit [] means "deliberately
        # ungrouped". Writing [] for both would make `groups infer` skip every
        # scope it had never seen.
        if self.groups is not None:
            d["groups"] = list(self.groups)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Scope":
        # An ABSENT glob key means "use the defaults"; an explicitly empty list
        # means "disabled". Collapsing the two would silently disable a new
        # collector for every registry written before that collector existed --
        # which is exactly what happened when code_globs was introduced.
        from .defaults import DEFAULT_CODE_GLOBS, DEFAULT_EPISODE_GLOBS
        return cls(
            id=d["id"], name=d["name"], root=Path(d["root"]),
            aliases=list(d.get("aliases") or []),
            episode_globs=(list(d["episode_globs"]) if "episode_globs" in d
                           else list(DEFAULT_EPISODE_GLOBS)),
            code_globs=(list(d["code_globs"]) if "code_globs" in d
                        else list(DEFAULT_CODE_GLOBS)),
            updated_at=d.get("updated_at", ""), meta=dict(d.get("meta") or {}),
            groups=(list(d["groups"]) if "groups" in d else None),
        )


@dataclass(slots=True)
class Chunk:
    """A verbatim slice of episode content. Never summarized, never paraphrased."""

    kind: str          # note | session | memory | commit | doc
    source: str        # citable origin, e.g. "doc:README.md" or "git:8a39216bd5"
    heading: str       # heading path within the source, "A > B > C"
    text: str
    ts: str = ""

    def to_json(self) -> dict:
        return {"kind": self.kind, "source": self.source,
                "heading": self.heading, "text": self.text, "ts": self.ts}

    @classmethod
    def from_json(cls, d: dict) -> "Chunk":
        return cls(kind=d["kind"], source=d["source"], heading=d.get("heading", ""),
                   text=d["text"], ts=d.get("ts", ""))


@dataclass(slots=True)
class EpisodeHit:
    chunk: Chunk
    score: float
    rerank_score: float | None = None

    def to_json(self) -> dict:
        d = self.chunk.to_json()
        d["score"] = round(self.score, 4)
        if self.rerank_score is not None:
            d["rerank_score"] = round(self.rerank_score, 4)
        return d


@dataclass(slots=True)
class StructureHit:
    """One structure-store result: a traversal rendered as citable text."""

    source: str        # which graph produced it, e.g. "code"
    nodes: int
    text: str
    ok: bool = True

    def to_json(self) -> dict:
        return {"source": self.source, "nodes": self.nodes,
                "text": self.text, "ok": self.ok}


@dataclass(slots=True)
class RouteResult:
    question: str
    query_tokens: list[str]
    ranked: list[str]
    selected: list[str]
    abstain: bool
    top_score: float
    top_matched: int
    detail: dict[str, dict] = field(default_factory=dict)
    group: str | None = None
    mode: str | None = None
    abstain_reason: str | None = None   # deictic | no_evidence | out_of_group

    def to_json(self) -> dict:
        return {
            "question": self.question, "query_tokens": self.query_tokens,
            "ranked": self.ranked, "selected": self.selected,
            "abstain": self.abstain, "top_score": round(self.top_score, 4),
            "top_matched": self.top_matched,
            "group": self.group, "mode": self.mode,
            "abstain_reason": self.abstain_reason,
            "detail": self.detail,
        }
