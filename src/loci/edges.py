"""What a project reaches for — the outbound half of a cross-project edge.

An edge is a claim that this project uses something by name. Resolving it
against another scope's signboard (`identity.targets`) is what turns "uses
`loci`" into "uses the scope `loci`".

Edges live here, in the registry layer, and never in either store. That is not
tidiness: `types.Scope` states that neither store is ever queried across a
scope boundary, and a `{"target": "loci"}` row folded into the episode store
would put the word `loci` in the vocabulary of every scope that shells out to
it — so a question about loci would route to whoever calls it. Collected per
scope from that scope's own tree, joined by arithmetic afterwards.

Two collectors, and the split is a measurement. Swept across fifteen real
repositories on 2026-09-06: zero `.gitmodules`, zero manifest dependencies
naming a sibling repository, and exactly one true cross-project edge —
`Delroy -> loci`, expressed as `shutil.which("loci")`. A collector that reads
manifests and stops has a measured recall of 0 on that corpus, which is why
`invoked_edges` exists beside `declared_edges` rather than after it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

# Dependency tables, in the four manifests loci reads. Cargo and go.mod are
# deliberately absent: `doctor` reports which kinds were read, and a kind that
# is silently unread is how "no edges" gets believed.
_NPM_TABLES = ("dependencies", "devDependencies", "peerDependencies",
               "optionalDependencies")

# `owner/repo` out of any git remote spelling, `.git` optional. Anchored on the
# host so a bare path never matches.
_GIT_URL = re.compile(
    r"(?:https?://|git\+https?://|ssh://(?:[^@/]+@)?|git@)"
    r"[^/:\s]+(?::\d+)?[/:]([^/\s]+)/([^/\s@#]+?)(?:\.git)?(?:[#@][^\s]*)?$")
# npm shorthand: `github:owner/repo`, `gitlab:owner/repo`, `owner/repo#ref`
_HOST_SHORTHAND = re.compile(r"^(?:github|gitlab|bitbucket):([^/]+)/([^#\s]+)")
_PATH_SPEC = re.compile(r"^(?:file|link|portal):(.+)$")
# PEP 508: a requirement's name is everything before the first marker char.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _edge(target: str, how: str, path: str, line: int) -> dict:
    return {"target": target.strip().lower(), "how": how,
            "source": f"{path}:{line}"}


def _slug(url: str) -> str | None:
    """`owner/repo`, lowercased, from a git URL or npm host shorthand."""
    for pattern in (_HOST_SHORTHAND, _GIT_URL):
        m = pattern.search(url.strip())
        if m:
            repo = m.group(2)
            if repo.endswith(".git"):
                repo = repo[:-4]
            return f"{m.group(1)}/{repo}".lower()
    return None


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _line_of(text: str, needle: str) -> int:
    """1-based line carrying `needle`, or 1.

    A citation is the whole point of an edge -- "Delroy uses loci" is an
    assertion, "Delroy uses loci at client/loci_memory.py:36" is checkable --
    and the structured readers below (json, tomllib) throw position away. So
    the value is found again in the raw text. Ambiguity is tolerable here: two
    identical spellings in one manifest are the same dependency.
    """
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return 1


# -- declared --------------------------------------------------------------
def _from_gitmodules(root: Path, out: list[dict]) -> None:
    text = _read(root / ".gitmodules")
    for i, line in enumerate(text.splitlines(), 1):
        if "url" not in line:
            continue
        _, _, value = line.partition("=")
        slug = _slug(value)
        if slug:
            out.append(_edge(slug, "declared", ".gitmodules", i))


def _from_package_json(root: Path, out: list[dict]) -> None:
    text = _read(root / "package.json")
    if not text:
        return
    try:
        data = json.loads(text)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for table in _NPM_TABLES:
        deps = data.get(table)
        if not isinstance(deps, dict):
            continue
        for name, spec in deps.items():
            line = _line_of(text, f'"{name}"')
            # The dependency NAME is an edge on its own: a plain registry
            # dependency on `loci-mem` points at the loci scope as surely as a
            # git URL does, and it is the only form a published package has.
            out.append(_edge(name, "declared", "package.json", line))
            if not isinstance(spec, str):
                continue
            slug = _slug(spec)
            if slug:
                out.append(_edge(slug, "declared", "package.json", line))
                continue
            path = _PATH_SPEC.match(spec)
            if path:
                # `file:../sibling` -- the directory name is what it is called.
                tail = path.group(1).rstrip("/").rsplit("/", 1)[-1]
                if tail and tail not in {".", ".."}:
                    out.append(_edge(tail, "declared", "package.json", line))


def _requirements(text: str, path: str, out: list[dict]) -> None:
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        slug = _slug(line)
        if slug:
            out.append(_edge(slug, "declared", path, i))
            continue
        m = _REQ_NAME.match(line)
        if m:
            out.append(_edge(m.group(1), "declared", path, i))


def _from_pyproject(root: Path, out: list[dict]) -> None:
    text = _read(root / "pyproject.toml")
    if not text:
        return
    from .identity import _tomllib

    reqs: list[str] = []
    if _tomllib is not None:
        try:
            project = (_tomllib.loads(text).get("project") or {})
        except Exception:
            project = {}
        if isinstance(project.get("dependencies"), list):
            reqs += [r for r in project["dependencies"] if isinstance(r, str)]
        extras = project.get("optional-dependencies")
        if isinstance(extras, dict):
            for group in extras.values():
                if isinstance(group, list):
                    reqs += [r for r in group if isinstance(r, str)]
    else:
        # No tomllib on 3.10. Requirement strings are quoted list items, and
        # reading them line-wise over-collects nothing a name regex accepts.
        reqs = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        reqs = [a or b for a, b in reqs]

    for req in reqs:
        slug = _slug(req)
        if slug:
            out.append(_edge(slug, "declared", "pyproject.toml",
                             _line_of(text, req)))
            continue
        m = _REQ_NAME.match(req)
        if m and (" " in req or any(c in req for c in "<>=!~[")
                  or req.strip() == m.group(1)):
            out.append(_edge(m.group(1), "declared", "pyproject.toml",
                             _line_of(text, req)))


def _from_requirements(root: Path, out: list[dict]) -> None:
    for p in sorted(root.glob("requirements*.txt")):
        _requirements(_read(p), p.name, out)


def _dedupe(edges: list[dict]) -> list[dict]:
    """One row per (target, how); the first citation wins.

    A dependency listed in four npm tables is one edge with four citations,
    and reporting it four times says "this project uses it a lot", which is
    not a fact about anything.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for e in edges:
        key = (e["target"], e["how"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def declared_edges(root: Path) -> list[dict]:
    """Outbound references this project states in a manifest.

    Root manifests only, the same rule as the signboard: an inner
    `package.json` belongs to a sub-scope, and its dependencies are that
    sub-scope's edges.
    """
    root = Path(root)
    out: list[dict] = []
    for reader in (_from_gitmodules, _from_package_json, _from_pyproject,
                   _from_requirements):
        try:
            reader(root, out)
        except Exception:
            continue
    return _dedupe(out)
