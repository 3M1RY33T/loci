"""Credential redaction, applied before anything is written to disk.

The episode store indexes commit bodies, docstrings and comment blocks — three
places where secrets get committed by accident and then live forever in history
even after the file is fixed. A scan of one real 16,015-chunk store came back
clean, but that is a property of those repositories, not of this tool, and a
memory system whose pitch is "it holds your private working history" cannot ship
without an answer here.

Redaction happens at collection time, so a secret never reaches `episodes.json`,
the embedding vectors, or the fitted rankers. Nothing downstream has to be
trusted to handle it.

Deliberately biased toward over-redaction. A false positive costs one chunk a
little retrieval quality; a false negative copies a live credential into a
plaintext file and a vector index.
"""
from __future__ import annotations

import re
from pathlib import Path

# Files that are secret-bearing by nature. The default globs never reach these,
# but a user can add any glob they like, so the denylist is enforced regardless.
SENSITIVE_NAMES = frozenset({
    ".env", ".envrc", "credentials", "credentials.json", "secrets.json",
    "secrets.yaml", "secrets.yml", ".netrc", ".pgpass", "id_rsa", "id_ed25519",
    ".dev.vars", "service-account.json",
})
SENSITIVE_SUFFIXES = frozenset({
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk", ".asc",
})
SENSITIVE_STEMS = ("secret", "credential", "private-key", "privatekey")

# (label, pattern). Ordered most-specific first so a vendor-shaped token is
# reported as itself rather than as a generic assignment.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private-key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("aws-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe-key", re.compile(r"\b(?:sk|rk)_live_[0-9A-Za-z]{16,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("connection-string", re.compile(
        r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:@/]+:[^\s:@/]{6,}@[^\s/]+")),
    ("bearer-token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{24,}")),
    # Generic assignment last: a long opaque value bound to a secret-ish name.
    ("secret-assignment", re.compile(
        r"""(?ix)
        \b(?:pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key
           |auth[_-]?token|client[_-]?secret|private[_-]?key)\b
        \s*[:=]\s*
        (['"]?)(?P<v>[A-Za-z0-9/+_\-\.]{16,})\1
        """)),
]


def is_sensitive_file(path: Path) -> bool:
    """True for files that should never be collected at all."""
    name = path.name.lower()
    if name in SENSITIVE_NAMES or path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True
    if name.startswith(".env"):
        return True
    stem = path.stem.lower()
    return any(s in stem for s in SENSITIVE_STEMS) and path.suffix.lower() in {
        ".json", ".yaml", ".yml", ".toml", ".ini", ".conf", ".txt"}


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted text, {label: count}). Empty dict when nothing matched."""
    if not text:
        return text, {}
    found: dict[str, int] = {}
    for label, pat in PATTERNS:
        def _sub(m: re.Match) -> str:
            found[label] = found.get(label, 0) + 1
            # For an assignment, keep the key name so the chunk still reads
            # sensibly and still routes -- only the value is destroyed.
            if label == "secret-assignment":
                whole = m.group(0)
                return whole[:whole.index(m.group("v"))] + f"[REDACTED:{label}]"
            return f"[REDACTED:{label}]"
        text = pat.sub(_sub, text)
    return text, found
