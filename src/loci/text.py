r"""Tokenization shared by routing and episode search.

Deliberately not a plain ``\w+`` split: routing matches natural-language prose
against identifier vocabularies, so identifiers must decompose. ``GlassesBridge``
has to yield ``glasses`` and ``bridge`` or a question phrased in English can
never reach a symbol named in camelCase.

The implementation is Rust, in ``rust/loci-core/src/text.rs``, and this module
is the shim that keeps every existing import path working. It moved for cost:
the tokenizer runs over every file at index time and every question at query
time, and is 3.3x faster there -- 4.4ms against 14.6ms on 91k characters of
real corpus.

What the port had to preserve, and what proves it did:

* ``rules_signature()`` returns the same 16 characters. ``index.fingerprint``
  seeds itself with it, so a different value would make every installed index
  look stale and silently reindex the world.
* Python's ``[^\W_]`` and its camelCase lookahead have no equivalent in Rust's
  regex crate, and are written out there rather than approximated.
* ``tests/test_rust_parity.py`` compares the two over 16,774 real chunks,
  9,566 index terms and 585,708 individual tokens, against a verbatim copy of
  0.5.0 kept in ``tests/reference/text_0_5_0.py``.
"""
from __future__ import annotations

from loci._core import (  # noqa: F401
    is_hex_blob,
    is_unsegmented,
    rules_signature,
    strip_diacritics,
    tokens,
    unique_tokens,
)

# The constants stay here as well as in Rust. They are cheap to duplicate, they
# are what a reader of this module comes looking for, and `rules_signature`
# hashes the Rust copies -- so the parity test comparing signatures is also
# what catches the two drifting apart.
MIN_LEN = 3
MAX_LEN = 30
MIN_LEN_ALNUM = 2
MIN_LEN_NON_LATIN = 2
NGRAM_SCRIPT_MIN = 3
NGRAM_SIZE = 2
HEX_BLOB_MIN = 6

STOPWORDS = frozenset("""
the and for with that this from does did how why what when where which who
was were are you your our ours its not but into out off over under about
again more most some such only own same too very just now then than there
here they them their any all one two both can could should would have has had
""".split())


def token_set(text: str, *, drop_stopwords: bool = True) -> set[str]:
    """Tokens as a set.

    Stays Python: it is one `set()` over the Rust result, and crossing the
    boundary again to build it there would cost a conversion to buy nothing.
    """
    return set(tokens(text, drop_stopwords))
