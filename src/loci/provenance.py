"""Who a repository belongs to, inferred from git rather than configured.

`loci scan` registered every git repository it found and read nothing about who
owned it. On the development corpus that put a stranger's project -- 5,939
chunks, the second-largest scope -- into the same flat namespace as the user's
own work, competing for every question.

The signal was already on disk. This module reads it: the remote org, and
failing that the dominant commit author. Nothing here is configured, because a
configuration step that must be completed before the tool behaves correctly is
a step most users will not take.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

GIT_TIMEOUT = 15

# Below this share of remotes, no org is "yours" and the module says so rather
# than guessing. A wrong identity does not degrade gracefully -- it inverts
# every classification at once, labelling the user's own work as a vendor's.
PLURALITY = 0.40

_SSH = re.compile(r"^(?:ssh://)?git@[^:/]+[:/]([^/]+)/")
_HTTPS = re.compile(r"^https?://(?:[^@/]+@)?[^/]+/([^/]+)/")


@dataclass(slots=True)
class Identity:
    """Who the user appears to be, and whether that is worth acting on."""

    org: str | None = None
    email: str | None = None
    confident: bool = False

    def describe(self) -> str:
        if not self.confident:
            return "no dominant remote org; treating every project as yours"
        return f"you are {self.org}" + (f" <{self.email}>" if self.email else "")


def _git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""
    return p.stdout.strip() if p.returncode == 0 else ""


def remote_org(root: Path) -> str | None:
    """The owning org of `origin`, lowercased, or None."""
    url = _git(root, "remote", "get-url", "origin")
    if not url:
        return None
    for pattern in (_SSH, _HTTPS):
        m = pattern.match(url)
        if m:
            return m.group(1).lower()
    return None


def author_emails(root: Path, n: int = 200) -> Counter:
    """Commit author emails over the last `n` commits, lowercased."""
    out = _git(root, "log", f"-{n}", "--pretty=format:%ae")
    return Counter(line.strip().lower() for line in out.splitlines() if line.strip())


def _config_email() -> str | None:
    try:
        p = subprocess.run(["git", "config", "--get", "user.email"],
                           capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    return (p.stdout.strip().lower() or None) if p.returncode == 0 else None


def infer_identity(roots: list[Path]) -> Identity:
    """The modal remote org across `roots`, when one clearly dominates."""
    orgs: Counter = Counter()
    for r in roots:
        org = remote_org(Path(r))
        if org:
            orgs[org] += 1

    email = _config_email()
    if not orgs:
        return Identity(org=None, email=email, confident=False)

    ranked = orgs.most_common()
    top_org, top_count = ranked[0]
    share = top_count / sum(orgs.values())
    tied = len(ranked) > 1 and ranked[1][1] == top_count
    if share < PLURALITY or tied:
        return Identity(org=None, email=email, confident=False)
    return Identity(org=top_org, email=email, confident=True)


def classify(root: Path, identity: Identity) -> str:
    """The provenance group for one repository.

    `client:*` groups are never inferred -- a client relationship is not visible
    in git. The user asserts those with `loci group add`.
    """
    root = Path(root)

    org = remote_org(root)
    if org:
        if not identity.confident:
            return "me"
        return "me" if org == identity.org else f"vendor:{org}"

    if not (root / ".git").is_dir():
        return "me"          # you placed it here deliberately

    emails = author_emails(root)
    if not emails or not identity.email:
        return "me"
    top_email, _ = emails.most_common(1)[0]
    return "me" if top_email == identity.email else "vendor:unknown"
