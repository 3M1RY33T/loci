"""The agent-facing skill, and putting it where a client will find it.

The skill ships inside the package rather than in a dotfile repository or a
plugin, for one reason: it documents flags. `--scope`, `--group`, `--fast` and
the abstention reasons are all things the CLI can change, and a copy of that
surface maintained anywhere else drifts silently -- the user finds out when an
agent runs a flag that stopped existing two releases ago. Shipped together,
`loci skill install` after an upgrade is the whole update path.

Installing is a copy into `<skills root>/loci/SKILL.md` and nothing else. No
client config is written: registering the MCP server means editing a file the
user shares with every other server they have, and a printed one-line command
they can read before running is worth more than a silent edit that works.
"""
from __future__ import annotations

from pathlib import Path

SKILL_NAME = "loci"
CLAUDE_SKILLS = Path.home() / ".claude" / "skills"
MCP_HINT = "claude mcp add loci -- loci mcp"


def skill_text() -> str:
    """The packaged SKILL.md, verbatim."""
    from importlib.resources import files

    return (files("loci") / "skill" / "SKILL.md").read_text(encoding="utf-8")


def install(root: Path | None = None, *, force: bool = False) -> tuple[bool, Path, str]:
    """Copy the skill under `root`. Returns (wrote, path, detail).

    An existing file that differs is left alone unless forced. It is either a
    user's own edit or a copy from a different version, and there is no way to
    tell which from here -- so the one case that must not happen silently is
    overwriting something someone wrote by hand.
    """
    dest_dir = Path(root or CLAUDE_SKILLS).expanduser() / SKILL_NAME
    dest = dest_dir / "SKILL.md"
    text = skill_text()
    if dest.is_file():
        try:
            current = dest.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current == text:
            return False, dest, "already up to date"
        if not force:
            return False, dest, "differs from the shipped copy; --force overwrites"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return True, dest, "installed"
