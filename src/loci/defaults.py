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
DEFAULT_CODE_GLOBS = [
    "**/*.py", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.mjs",
    "**/*.go", "**/*.rs", "**/*.swift", "**/*.dart", "**/*.rb",
    "**/*.java", "**/*.kt", "**/*.cs",
]
