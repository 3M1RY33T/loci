r"""Tokenization shared by routing and episode search.

Deliberately not a plain ``\w+`` split: routing matches natural-language prose
against identifier vocabularies, so identifiers must decompose. ``GlassesBridge``
has to yield ``glasses`` and ``bridge`` or a question phrased in English can
never reach a symbol named in camelCase.
"""
from __future__ import annotations

import re
import unicodedata

_WORDISH = re.compile(r"[^\W\d_]+", re.UNICODE)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")

MIN_LEN = 3
MAX_LEN = 30

# Pure prose function words only. Common English VERBS are deliberately absent:
# identifier vocabularies are full of them (run_agent_turn, get_node, use_cache,
# worktree -> work+tree), and dropping them would delete exactly the tokens that
# discriminate between scopes.
STOPWORDS = frozenset("""
the and for with that this from does did how why what when where which who
was were are you your our ours its not but into out off over under about
again more most some such only own same too very just now then than there
here they them their any all one two both can could should would have has had
""".split())


def strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase tokens with camelCase / snake_case / path decomposition."""
    out: list[str] = []
    for chunk in _WORDISH.findall(strip_diacritics(text)):
        for part in (_CAMEL.findall(chunk) or [chunk]):
            t = part.lower()
            if MIN_LEN <= len(t) <= MAX_LEN:
                if drop_stopwords and t in STOPWORDS:
                    continue
                out.append(t)
    return out


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    return set(tokens(text, drop_stopwords=drop_stopwords))


def unique_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokens, deduplicated, original order preserved."""
    return list(dict.fromkeys(tokens(text, drop_stopwords=drop_stopwords)))
