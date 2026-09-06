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

Two collectors, and the split is a measurement. Across fifteen real
repositories on 2026-09-06 there are exactly two cross-project edges, one of
each kind:

    3m1ry33t-github-io -> urthreads   "urthreads": "^1.2.0"   package.json:10
    delroy             -> loci        shutil.which("loci")    client/loci_memory.py:36

Neither collector finds the other's edge. There is no `.gitmodules` in the
corpus, no path dependency and no git-URL dependency, so `declared_edges`
rests entirely on the plain registry name; and `Delroy -> loci` is a binary
resolved on PATH, declared in no manifest, lockfile or submodule. Either
collector alone has a measured recall of 0.5, which is why `invoked_edges`
exists beside `declared_edges` rather than after it.
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


# -- invoked ---------------------------------------------------------------
# Files worth reading for a spawn. Not DEFAULT_CODE_GLOBS: only these three
# families are parsed, and walking Go and Rust sources to find nothing is the
# cost `SKIP_DIRS` exists to avoid paying.
_SCAN_GLOBS = ["**/*.py", "**/*.js", "**/*.mjs", "**/*.ts", "**/*.tsx",
               "**/*.json"]

# A generated lockfile is megabytes of dependency records and carries no spawn.
_MAX_SCAN_BYTES = 512 * 1024

# `module.func` forms that take a command as their first argument.
_SPAWN_ATTRS = {
    ("shutil", "which"),
    ("subprocess", "run"), ("subprocess", "Popen"), ("subprocess", "call"),
    ("subprocess", "check_call"), ("subprocess", "check_output"),
}
# Bare names, for `from subprocess import Popen`. `run` and `call` are
# deliberately absent: they are among the most common function names in any
# codebase, and a bare `run(["deploy"])` is not evidence of a subprocess.
_SPAWN_NAMES = {"which", "Popen", "check_call", "check_output"}

_JS_SPAWN = re.compile(
    r"\b(?:spawn|spawnSync|exec|execSync|execFile|execFileSync)\s*\(\s*"
    r"[\'\"`]([^\'\"`]+)[\'\"`]")


def _first_command(node: ast.AST) -> str | None:
    """The command out of a spawn's first argument.

    Both shapes: `which("loci")` and `run(["loci", "ask", q])`. A list whose
    first element is not a literal -- `run([exe, "ask"])` -- names nothing.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        head = node.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            return head.value
    return None


def _is_spawn(func: ast.AST) -> bool:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return (func.value.id, func.attr) in _SPAWN_ATTRS or (
            func.value.id == "os" and (func.attr.startswith("exec")
                                       or func.attr.startswith("spawn")))
    return isinstance(func, ast.Name) and func.id in _SPAWN_NAMES


def _python_spawns(text: str, rel: str, out: list[dict]) -> None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return                       # a file this parser cannot read is not a
                                     # reason to abandon the repository
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not _is_spawn(node.func):
            continue
        cmd = _first_command(node.args[0])
        if cmd:
            out.append(_edge(_basename(cmd), "command", rel, node.lineno))


def _js_spawns(text: str, rel: str, out: list[dict]) -> None:
    # Gated on the module being present in the file. `exec(` and `spawn(` are
    # ordinary words in a codebase that never starts a process; requiring the
    # import is what keeps a scheduler's `spawn("worker")` out of the table.
    if "child_process" not in text:
        return
    for i, line in enumerate(text.splitlines(), 1):
        for m in _JS_SPAWN.finditer(line):
            out.append(_edge(_basename(m.group(1)), "command", rel, i))


def _json_commands(obj, text: str, rel: str, out: list[dict]) -> None:
    """`{"command": ..., "args": [...]}` -- an MCP server or a task runner.

    `args` is required. A lone `command` key is how every configuration file
    that happens to use the word becomes an edge; the pair is what says a
    process is being described.
    """
    if isinstance(obj, dict):
        cmd = obj.get("command")
        if isinstance(cmd, str) and "args" in obj:
            out.append(_edge(_basename(cmd), "command", rel,
                             _line_of(text, f'"{cmd}"')))
        for v in obj.values():
            _json_commands(v, text, rel, out)
    elif isinstance(obj, list):
        for v in obj:
            _json_commands(v, text, rel, out)


def _basename(cmd: str) -> str:
    """`/usr/local/bin/loci` and `loci` are the same command."""
    return cmd.strip().rstrip("/").rsplit("/", 1)[-1]


def invoked_edges(root: Path) -> list[dict]:
    """Outbound references this project makes at runtime, by name.

    The measured majority of what a manifest cannot see. `Delroy -> loci` is
    `shutil.which("loci")` and nothing else -- no manifest, no lockfile, no
    submodule -- so a collector that reads declarations and stops misses half
    the edges in the corpus.

    Only spawn positions count. A bare `LOCI_SERVER_NAME = "loci"` is a name,
    not a call, and admitting string constants makes every string in a
    codebase an edge.
    """
    from .walk import iter_files

    root = Path(root)
    out: list[dict] = []
    for f in iter_files(root, _SCAN_GLOBS):
        try:
            if f.stat().st_size > _MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        rel = str(f.relative_to(root)) if f.is_relative_to(root) else f.name
        text = _read(f)
        if not text:
            continue
        try:
            if f.suffix == ".py":
                _python_spawns(text, rel, out)
            elif f.suffix == ".json":
                _json_commands(json.loads(text), text, rel, out)
            else:
                _js_spawns(text, rel, out)
        except Exception:
            continue
    return _dedupe(out)


def edges_for(scope) -> list[dict]:
    """Every outbound edge of one scope, minus the ones pointing at itself.

    A console script invoking its own binary is not a cross-project edge, and
    dropping it here means nothing downstream has to know to ignore it.
    """
    from .identity import signboard_of, targets

    own = targets(signboard_of(scope) or {})
    own.add(scope.id.lower())
    root = Path(scope.root)
    found = _dedupe(declared_edges(root) + invoked_edges(root))
    return [e for e in found if e["target"] not in own]


# -- resolve and persist ---------------------------------------------------
EDGES_VERSION = 1


def build(scopes) -> dict[str, list[dict]]:
    """Every scope's outbound edges, keyed by scope id.

    Collected per scope from that scope's own tree. Nothing here reads another
    scope, and nothing here resolves anything: the join is `resolved`, and
    keeping it separate is what lets a table survive registering a new scope --
    an edge naming `zim-compress` becomes a cross-project edge the moment that
    project is registered, without recollecting anything.
    """
    out: dict[str, list[dict]] = {}
    for s in scopes:
        if not Path(s.root).is_dir():
            continue
        found = edges_for(s)
        if found:
            out[s.id] = found
    return out


def resolved(scopes, table: dict[str, list[dict]]) -> list[dict]:
    """Edges whose target names another REGISTERED scope.

    Arithmetic over the registry: no ranking, no floor, no abstention. An edge
    that resolves is a fact with a citation, and one that does not is an
    ordinary external dependency -- kept in the table, absent from the answer.
    """
    from .identity import signboard_of, targets

    owners: list[tuple[str, set[str]]] = []
    for s in scopes:
        own = targets(signboard_of(s) or {})
        own.add(s.id.lower())
        owners.append((s.id, own))

    out: list[dict] = []
    for sid, edges in table.items():
        for e in edges:
            for other, own in owners:
                if other == sid or e["target"] not in own:
                    continue
                out.append({"from": sid, "to": other, "target": e["target"],
                            "how": e["how"], "source": e["source"]})
    return out


def save_edges(table: dict[str, list[dict]]) -> None:
    from .paths import atomic_write, edges_file, ensure_home

    ensure_home()
    atomic_write(edges_file(),
                 json.dumps({"version": EDGES_VERSION, "scopes": table},
                            indent=2, sort_keys=True))


def load_edges() -> dict[str, list[dict]]:
    """The stored table, or `{}` when there is none.

    A malformed file degrades to "no edges collected" rather than raising, the
    same way `load_policy` does: a traceback out of `loci ask` is a worse
    answer than a missing one, and `doctor` reports the emptiness either way.
    """
    from .paths import edges_file

    try:
        data = json.loads(edges_file().read_text(encoding="utf-8"))
        table = data.get("scopes")
        return table if isinstance(table, dict) else {}
    except Exception:
        return {}
