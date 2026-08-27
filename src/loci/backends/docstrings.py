"""Third episode source: docstrings and significant comment blocks.

The seam this closes was found by the eval set, not by inspection. Asked *"why
would the interface load fine but stop responding to clicks on Windows?"*,
loci returned commits about API-key encryption -- while a five-line docstring
on ``register_static_mime_types()`` in ``odysseus/app.py`` answered it exactly.

Neither store could see it. The structure graph indexes that function's *label*
and its path; the episode store indexed README, docs and commits. The prose that
explains the code is written inside the code, and it fell between them.

Both forms are collected, because both carry explanations:

    docstring       ast.get_docstring on module / class / function
    comment block   a run of consecutive full-line comments

The second is not an afterthought. The other failing eval question -- why an
embedding model fails on a network drive -- is answered by a ``#`` comment
block, not a docstring.

Each chunk is headed by the symbol it belongs to, so a hit cites
``app.py > register_static_mime_types()`` and lines up with the same symbol in
the structure graph.
"""
from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

# Only prose worth retrieving. "Return the thing." explains nothing and would
# dilute both rankers; the threshold is words, not characters, so a short but
# real sentence survives while a type restatement does not.
MIN_WORDS = 12
MAX_CHARS = 900
MAX_PER_FILE = 60

# A run of `//` or `#` lines shorter than this is a label, not an explanation.
MIN_COMMENT_LINES = 2

BOILERPLATE = re.compile(
    r"^\s*(?:copyright|licen[sc]e|spdx-|all rights reserved|"
    r"this file (?:is|was) (?:auto[- ]?)?generated|@generated|eslint-|prettier-|"
    r"type: ignore|noqa|pylint:|mypy:)",
    re.IGNORECASE,
)

BLOCK_COMMENT = re.compile(r"/\*\*?(?P<body>[\s\S]*?)\*/")
LINE_COMMENT = re.compile(r"^\s*//(?P<body>.*)$")

PY_SUFFIXES = {".py", ".pyi"}
CLIKE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".go", ".rs",
                  ".java", ".swift", ".kt", ".c", ".h", ".cc", ".cpp", ".cs",
                  ".dart", ".php", ".scala"}


def _words(text: str) -> int:
    return len(text.split())


def _keep(text: str) -> bool:
    t = text.strip()
    if not t or _words(t) < MIN_WORDS:
        return False
    return not BOILERPLATE.match(t)


def _clean(text: str) -> str:
    """Strip comment furniture: leading `*`, `#`, `//` and blank padding."""
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*(?:\*|//+|#+)\s?", "", ln.rstrip())
        lines.append(ln)
    return "\n".join(lines).strip()


# ==========================================================================
# Python
# ==========================================================================
def _py_symbol_ranges(tree: ast.AST) -> list[tuple[int, int, str]]:
    """(start, end, qualified name) for every def/class, innermost last."""
    out: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                label = name if isinstance(child, ast.ClassDef) else f"{name}()"
                end = getattr(child, "end_lineno", child.lineno) or child.lineno
                out.append((child.lineno, end, label))
                walk(child, f"{name}.")
    walk(tree, "")
    return sorted(out, key=lambda r: (r[0], -(r[1] - r[0])))


def _symbol_at(ranges: list[tuple[int, int, str]], line: int) -> str:
    best = ""
    for start, end, label in ranges:
        if start <= line <= end:
            best = label          # later matches are nested deeper
    return best


def extract_python(text: str, rel: str) -> list[tuple[str, str]]:
    r"""Return (heading, body) pairs for one Python file.

    Warnings are suppressed because this parses code the user did not write:
    a repository containing `"\*"` in a docstring makes CPython emit a
    SyntaxWarning, and `loci index` printing parser complaints about somebody
    else's source is noise the user can do nothing about.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out: list[tuple[str, str]] = []
    ranges = _py_symbol_ranges(tree)

    mod_doc = ast.get_docstring(tree)
    if mod_doc and _keep(mod_doc):
        out.append((rel, mod_doc.strip()))

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or not _keep(doc):
            continue
        label = _symbol_at(ranges, node.lineno) or node.name
        out.append((label, doc.strip()))

    # Comment runs. The AST discards them, so they are recovered from the token
    # stream and attributed to whichever symbol encloses them.
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = []
    run: list[str] = []
    run_line = 0
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            if not run:
                run_line = tok.start[0]
            run.append(tok.string)
        elif run and tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        elif run:
            if len(run) >= MIN_COMMENT_LINES:
                body = _clean("\n".join(run))
                if _keep(body):
                    sym = _symbol_at(ranges, run_line)
                    out.append((f"{sym} (comment)" if sym else f"{rel} (comment)", body))
            run = []
    if run and len(run) >= MIN_COMMENT_LINES:
        body = _clean("\n".join(run))
        if _keep(body):
            sym = _symbol_at(ranges, run_line)
            out.append((f"{sym} (comment)" if sym else f"{rel} (comment)", body))
    return out[:MAX_PER_FILE]


# ==========================================================================
# C-family
# ==========================================================================
_DECL = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|static\s+|async\s+)*"
    r"(?:function|class|const|let|var|def|func|fn|interface|type|struct|enum)\s+"
    r"([A-Za-z_$][\w$]*)")


def _following_symbol(lines: list[str], idx: int) -> str:
    """Name of the first declaration after `idx`; comments usually precede one."""
    for ln in lines[idx:idx + 4]:
        m = _DECL.match(ln)
        if m:
            return m.group(1)
    return ""


def extract_clike(text: str, rel: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    lines = text.splitlines()

    for m in BLOCK_COMMENT.finditer(text):
        body = _clean(m.group("body"))
        if not _keep(body):
            continue
        line_no = text.count("\n", 0, m.end())
        sym = _following_symbol(lines, line_no)
        out.append((f"{sym}()" if sym else rel, body))

    run: list[str] = []
    start = 0
    for i, ln in enumerate(lines):
        m = LINE_COMMENT.match(ln)
        if m:
            if not run:
                start = i
            run.append(m.group("body"))
            continue
        if run:
            if len(run) >= MIN_COMMENT_LINES:
                body = _clean("\n".join(run))
                if _keep(body):
                    sym = _following_symbol(lines, i)
                    out.append((f"{sym}() (comment)" if sym else f"{rel} (comment)", body))
            run = []
    return out[:MAX_PER_FILE]


def extract(path: Path, rel: str) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    if suffix in PY_SUFFIXES:
        return [(h, b[:MAX_CHARS]) for h, b in extract_python(text, rel)]
    if suffix in CLIKE_SUFFIXES:
        return [(h, b[:MAX_CHARS]) for h, b in extract_clike(text, rel)]
    return []
