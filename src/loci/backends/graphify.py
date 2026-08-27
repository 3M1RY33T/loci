"""Structure backend: graphify.

An adapter, not a fork. graphify's value is an extraction pipeline over ~18
tree-sitter grammars; none of that needs changing to make it scope-aware. loci
reads its `graph.json` data contract and shells out to `graphify query --graph`,
so upstream stays upstream.

The scoping gap that motivated loci lives here in miniature: graphify tags
merged-graph nodes with a `repo` attribute but exposes no way to filter on it,
and its query CLI accepts only --budget/--context/--graph/--dfs. loci does not
patch that; it keeps one graph per scope and points --graph at the right file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from ..text import token_set
from ..types import Scope, StructureHit

QUERY_TIMEOUT = 60
MIN_SOURCE_NODES = 20     # a 2-node registry graph is not worth a subprocess

# graph.json keys loci relies on. graphify is pre-1.0 and has already migrated
# its node-ID scheme once, so an adapter that assumes more than this is a
# liability. Checked at read time rather than trusted.
REQUIRED_TOP_KEYS = ("nodes",)


class GraphifyBackend:
    name = "graphify"

    def available(self) -> bool:
        return shutil.which("graphify") is not None

    # -- discovery ---------------------------------------------------------
    def graph_paths(self, scope: Scope) -> list[Path]:
        """Graphs belonging to this scope, primary first."""
        out = []
        primary = scope.root / "graphify-out" / "graph.json"
        if primary.is_file():
            out.append(primary)
        for extra in scope.meta.get("extra_graphs", []):
            p = Path(extra).expanduser()
            if p.is_file() and p not in out:
                out.append(p)
        return out

    def sources(self, scope: Scope) -> list[dict]:
        srcs = []
        for p in self.graph_paths(scope):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not all(k in raw for k in REQUIRED_TOP_KEYS):
                continue
            n = len(raw.get("nodes") or [])
            srcs.append({"kind": _kind_for(p, scope), "path": str(p), "nodes": n})
        return srcs

    # -- routing input -----------------------------------------------------
    def vocabulary(self, scope: Scope) -> Counter:
        """Token -> node document-frequency across every graph in this scope.

        Counted once per node, not per occurrence, so a token repeated inside a
        single label cannot inflate the scope's routing score.
        """
        counts: Counter = Counter()
        for p in self.graph_paths(scope):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for n in raw.get("nodes") or []:
                toks = token_set(n.get("label") or "")
                sf = n.get("source_file") or ""
                if sf:
                    toks |= token_set(sf)
                counts.update(toks)
        return counts

    def node_count(self, scope: Scope) -> int:
        return sum(s["nodes"] for s in self.sources(scope))

    # -- query -------------------------------------------------------------
    def query(self, scope: Scope, query: str, *, budget: int = 2000,
              dfs: bool = False) -> list[StructureHit]:
        srcs = [s for s in self.sources(scope) if s["nodes"] >= MIN_SOURCE_NODES]
        if not srcs or not query.strip():
            return []
        per = max(250, budget // len(srcs))
        hits: list[StructureHit] = []
        for s in srcs:
            ok, text = self._run(s["path"], query, per, dfs)
            hits.append(StructureHit(source=s["kind"], nodes=s["nodes"],
                                     text=text, ok=ok))
        return hits

    def _run(self, graph_path: str, query: str, budget: int,
             dfs: bool) -> tuple[bool, str]:
        cmd = ["graphify", "query", query, "--graph", graph_path,
               "--budget", str(budget)]
        if dfs:
            cmd.append("--dfs")
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=QUERY_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, f"(graphify query timed out after {QUERY_TIMEOUT}s)"
        except FileNotFoundError:
            return False, "(graphify CLI not on PATH; pip install 'loci-mem[graphify]')"
        if p.returncode != 0:
            return False, (p.stderr or p.stdout).strip()[:400]
        return True, p.stdout.strip()


def _kind_for(path: Path, scope: Scope) -> str:
    try:
        rel = path.relative_to(scope.root)
    except ValueError:
        return path.parent.parent.name or "graph"
    return "code" if rel.parts[0] == "graphify-out" else rel.parts[0]
