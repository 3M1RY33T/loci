"""Default collection globs, in their own module so `types` can read them
without importing `scopes` (which imports `types`)."""
from __future__ import annotations

# Episode sources every project has, whether or not anyone wrote docs for it.
# Cold start matters: on day one a scope has git history and a README and
# nothing else, and a memory tool that is empty for a week does not get adopted.
DEFAULT_EPISODE_GLOBS = [
    "README*",
    "docs/**/*.md",
    "*.md",
]

# Source files mined for docstrings and comment blocks. The prose that explains
# code is largely written inside the code, and it is invisible both to a graph
# that indexes symbol names and to a store that indexes markdown files.
# Pruned during traversal, not filtered afterwards. `Path.glob("**/*.py")`
# descends into node_modules and .venv before any filter can reject them, which
# measured 32.9s to fingerprint one repo; pruning the walk makes it ~0.2s.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env",
    "__pycache__", "dist", "build", "vendor", "target", ".next", ".nuxt",
    ".tox", "site-packages", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "graphify-out", ".gradle", "Pods", "DerivedData", ".terraform",
})

DEFAULT_CODE_GLOBS = [
    "**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.mjs",
    "**/*.go", "**/*.rs", "**/*.swift", "**/*.dart", "**/*.rb",
    "**/*.java", "**/*.kt", "**/*.cs",
]
