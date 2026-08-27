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
# A chunk may mix scripts. The camelCase pattern above matches ASCII Latin only,
# so applying it to a mixed chunk SILENTLY DISCARDS everything else: `invoice設定`
# tokenized to `['invoice']` and the identifying half vanished. Split into
# per-script runs first, and only camel-split the Latin ones.
_SCRIPT_RUN = re.compile(r"[A-Za-z]+|[^\WA-Za-z\d_]+", re.UNICODE)

MIN_LEN = 3
MAX_LEN = 30
# Scripts written without spaces pack more meaning per character, so a 3-char
# floor tuned for English discards whole words. Two characters is a common noun
# in Chinese and Japanese.
MIN_LEN_NON_LATIN = 2
# Scripts written without spaces produce one token per phrase, not per word.
# Measured on a real Chinese-documented repository: only 35% of CJK tokens were
# short enough to be usable as search terms, 25% ran past eight characters, and
# the longest was a thirty-character sentence -- matchable only by a query
# containing that exact sentence.
#
# Character bigrams are the standard answer when no segmenter is available: they
# approximate word boundaries well enough for retrieval without adding a
# dependency or a language model. The whole run is kept as well, so an exact
# phrase still matches exactly.
NGRAM_SCRIPT_MIN = 3      # runs at least this long also emit bigrams
NGRAM_SIZE = 2

# Han, Hiragana, Katakana, Hangul. Thai and Khmer are also space-free but are
# left out until there is a corpus to measure them against.
_UNSEGMENTED = (
    ("\u4e00", "\u9fff"), ("\u3040", "\u309f"), ("\u30a0", "\u30ff"),
    ("\uac00", "\ud7af"), ("\u3400", "\u4dbf"),
)


def is_unsegmented(text: str) -> bool:
    """True when the text is written in a script that does not use spaces."""
    return any(any(lo <= ch <= hi for lo, hi in _UNSEGMENTED) for ch in text)

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
        for run in (_SCRIPT_RUN.findall(chunk) or [chunk]):
            # Latin runs decompose by case; other scripts have no case to read,
            # so they pass through whole. Unsegmented is imperfect for languages
            # written without spaces, but it is not silently lost.
            if run.isascii():
                parts = _CAMEL.findall(run) or [run]
            else:
                parts = [run]
                if len(run) >= NGRAM_SCRIPT_MIN and is_unsegmented(run):
                    parts += [run[i:i + NGRAM_SIZE]
                              for i in range(len(run) - NGRAM_SIZE + 1)]
            for part in (parts or [run]):
                t = part.lower()
                floor = MIN_LEN if t.isascii() else MIN_LEN_NON_LATIN
                if floor <= len(t) <= MAX_LEN:
                    if drop_stopwords and t in STOPWORDS:
                        continue
                    out.append(t)
    return out


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    return set(tokens(text, drop_stopwords=drop_stopwords))


def unique_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokens, deduplicated, original order preserved."""
    return list(dict.fromkeys(tokens(text, drop_stopwords=drop_stopwords)))
