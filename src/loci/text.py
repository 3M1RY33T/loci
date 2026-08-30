r"""Tokenization shared by routing and episode search.

Deliberately not a plain ``\w+`` split: routing matches natural-language prose
against identifier vocabularies, so identifiers must decompose. ``GlassesBridge``
has to yield ``glasses`` and ``bridge`` or a question phrased in English can
never reach a symbol named in camelCase.
"""
from __future__ import annotations

import re
import unicodedata

# Digits are part of the word, not a separator. The old pattern was
# `[^\W\d_]+`, which treated every digit as whitespace AND discarded it, so
# `D1` became `d` and died on MIN_LEN, `S3` became `s`, and `base64` became
# `base`. Measured over 53 markdown files in three repositories: 258 distinct
# alphanumeric terms deleted outright across 1,246 occurrences, `s3` (102),
# `n8n` (70) and `d1` (64) among them -- and `d1` appears in this project's own
# README as a routing example that could never have worked.
#
# It also killed aliases. A scope named `3M1RY33T` tokenized its only alias to
# `[]`, so ALIAS_BOOST -- the strongest signal in the router at 6.0 -- could
# never fire for it, and naming the project outright abstained.
_WORDISH = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")
# A chunk may mix scripts. The camelCase pattern above matches ASCII Latin only,
# so applying it to a mixed chunk SILENTLY DISCARDS everything else: `invoice設定`
# tokenized to `['invoice']` and the identifying half vanished. Split into
# per-script runs first, and only camel-split the Latin ones.
_SCRIPT_RUN = re.compile(r"[A-Za-z]+|[^\WA-Za-z\d_]+", re.UNICODE)

MIN_LEN = 3
MAX_LEN = 30
# A term that mixes letters and digits is admitted at two characters. `d1`,
# `s3`, `g2` and `f1` are ordinary technical vocabulary and all of them are
# shorter than the prose floor, which exists to drop `a`, `of` and `to` -- none
# of which contain a digit.
MIN_LEN_ALNUM = 2
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


# Keeping digits lets commit hashes in too, and a corpus of git history is full
# of them: `e4bcfe0`, `741bbb9`, `41768130f0d5a159ec5100160890b2315ebb4fcb`.
# They are unique by construction, so scope-IDF scores every one of them as
# maximally discriminative for whichever scope happens to mention it.
#
# The test is deliberately narrow -- long, entirely hexadecimal, AND mixing
# digits with letters. Dropping the digit requirement would delete real English
# words that happen to be spellable in hex: `decade`, `facade`, `deface`,
# `deeded`. Requiring both halves keeps every one of those and still catches
# every hash observed in the corpus. `base64` survives on its `s`, `sha256` on
# its `h`, and anything under six characters is never tested at all, so `2fa`,
# `md5`, `sha1` and `utf8` pass untouched.
HEX_BLOB_MIN = 6
_HEX = re.compile(r"[0-9a-f]+", re.IGNORECASE)


def is_hex_blob(chunk: str) -> bool:
    """True for a git-hash-shaped run: long, all hex, and not a word."""
    if len(chunk) < HEX_BLOB_MIN or not _HEX.fullmatch(chunk):
        return False
    return any(c.isdigit() for c in chunk) and any(c.isalpha() for c in chunk)


def strip_diacritics(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(ch)
    )


def tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase tokens with camelCase / snake_case / path decomposition."""
    out: list[str] = []
    for chunk in _WORDISH.findall(strip_diacritics(text)):
        # A bare number is not vocabulary. Years, counts and version fragments
        # would otherwise become index terms now that digits survive.
        if chunk.isdigit() or is_hex_blob(chunk):
            continue
        # The whole alphanumeric form FIRST, then the letter runs below, so the
        # output is a superset of what the digit-stripping tokenizer produced:
        # `base64` yields `base64` AND `base`, and no query that matched before
        # can stop matching. Restricted to ASCII because a CJK run carrying a
        # digit is better served by the bigram path below.
        if chunk.isascii() and any(c.isdigit() for c in chunk):
            t = chunk.lower()
            if MIN_LEN_ALNUM <= len(t) <= MAX_LEN:
                if not (drop_stopwords and t in STOPWORDS):
                    out.append(t)
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


def rules_signature() -> str:
    """Stable hash of everything that decides what a token IS.

    The index fingerprint is a content signature -- which files exist and when
    they changed -- so a change to the TOKENIZER left every scope looking
    unchanged and `loci index` reused the old vocabulary wholesale. Shipping
    the alphanumeric fix, 12 of 14 scopes were silently kept at the previous
    rules; the routing index and the tokenizer disagreed about what a word is,
    and nothing said so.

    Seeding the fingerprint with this makes any future change to the rules
    invalidate the cache by construction, which a hand-bumped version constant
    cannot do -- it only works when somebody remembers.
    """
    import hashlib

    parts = [_WORDISH.pattern, _CAMEL.pattern, _SCRIPT_RUN.pattern, _HEX.pattern,
             str(MIN_LEN), str(MAX_LEN), str(MIN_LEN_ALNUM),
             str(MIN_LEN_NON_LATIN), str(NGRAM_SCRIPT_MIN), str(NGRAM_SIZE),
             str(HEX_BLOB_MIN), ",".join(sorted(STOPWORDS))]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    return set(tokens(text, drop_stopwords=drop_stopwords))


def unique_tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokens, deduplicated, original order preserved."""
    return list(dict.fromkeys(tokens(text, drop_stopwords=drop_stopwords)))
