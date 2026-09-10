//! Tokenization shared by routing and episode search.
//!
//! A port of `src/loci/text.py`. The reasoning behind every constant lives
//! there and is not duplicated; what follows is only what the port has to get
//! right that the Python did not have to say.
//!
//! Two of Python's regexes have no equivalent here and are written out rather
//! than approximated:
//!
//!   `[^\W_]`  -- the regex crate cannot negate `\W` inside a class. Python's
//!               `\w` is `isalnum() || '_'`, so minus underscore it is
//!               `[\p{L}\p{N}]`. Combining marks are NOT word characters in
//!               Python and are absent here for the same reason.
//!
//!   `[A-Z]+(?=[A-Z][a-z])` -- no lookahead. `camel_split` implements the
//!               whole alternation directly, because it is what decides that
//!               OAuth is O+Auth, HTTPServer is HTTP+Server and macOS is
//!               mac+OS, which is exactly what the Python comments were
//!               written about.
//!
//! Whether the two agree is not decided by reading them.
//! `tests/test_rust_parity.py` compares them over every chunk in the real
//! episode store and every term in the real scope index.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;
use unicode_normalization::char::canonical_combining_class;
use unicode_normalization::UnicodeNormalization;

pub const MIN_LEN: usize = 3;
pub const MAX_LEN: usize = 30;
pub const MIN_LEN_ALNUM: usize = 2;
pub const MIN_LEN_NON_LATIN: usize = 2;
pub const NGRAM_SCRIPT_MIN: usize = 3;
pub const NGRAM_SIZE: usize = 2;
pub const HEX_BLOB_MIN: usize = 6;

/// The patterns as PYTHON spells them, kept verbatim because
/// `rules_signature` hashes these strings and the hash must not move. The
/// Rust equivalents actually used for matching are built below.
pub const WORDISH_SRC: &str = r"[^\W_]+";
pub const CAMEL_SRC: &str = r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+";
pub const SCRIPT_RUN_SRC: &str = r"[A-Za-z]+|[^\WA-Za-z\d_]+";
pub const HEX_SRC: &str = r"[0-9a-f]+";

pub const STOPWORDS_SRC: &str = "\
the and for with that this from does did how why what when where which who
was were are you your our ours its not but into out off over under about
again more most some such only own same too very just now then than there
here they them their any all one two both can could should would have has had
";

/// Han, Hiragana, Katakana, Hangul, CJK Ext A. Mirrors `_UNSEGMENTED`.
const UNSEGMENTED: [(char, char); 5] = [
    ('\u{4e00}', '\u{9fff}'),
    ('\u{3040}', '\u{309f}'),
    ('\u{30a0}', '\u{30ff}'),
    ('\u{ac00}', '\u{d7af}'),
    ('\u{3400}', '\u{4dbf}'),
];

fn stopwords() -> &'static HashSet<String> {
    static S: OnceLock<HashSet<String>> = OnceLock::new();
    S.get_or_init(|| STOPWORDS_SRC.split_whitespace().map(str::to_string).collect())
}

fn wordish() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[\p{L}\p{N}]+").unwrap())
}

/// `[A-Za-z]+ | [^\WA-Za-z\d_]+` -- ASCII letter runs, or word characters that
/// are neither ASCII letters nor decimal digits. Written with class
/// subtraction because the negated form does not exist here.
fn script_run() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Za-z]+|[\p{L}\p{N}&&[^A-Za-z\p{Nd}]]+").unwrap())
}

fn hex_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)^[0-9a-f]+$").unwrap())
}

pub fn is_unsegmented(text: &str) -> bool {
    text.chars()
        .any(|c| UNSEGMENTED.iter().any(|(lo, hi)| c >= *lo && c <= *hi))
}

pub fn is_hex_blob(chunk: &str) -> bool {
    if chunk.chars().count() < HEX_BLOB_MIN || !hex_re().is_match(chunk) {
        return false;
    }
    chunk.chars().any(|c| c.is_ascii_digit()) && chunk.chars().any(|c| c.is_alphabetic())
}

/// NFKD, then drop every character with a nonzero canonical combining class.
/// Python filters on `unicodedata.combining(ch)` being truthy, which is that
/// same class.
pub fn strip_diacritics(text: &str) -> String {
    text.nfkd().filter(|c| canonical_combining_class(*c) == 0).collect()
}

/// Python's `_CAMEL` alternation, lookahead and all.
///
///   1. `[A-Z]+(?=[A-Z][a-z])` -- an uppercase run whose LAST capital starts
///      the following word. Greedy then backtracking, so it stops one short of
///      the run's end, and only when a lowercase letter follows the run.
///   2. `[A-Z]?[a-z]+`
///   3. `[A-Z]+`
///
/// Verified against the Python for OAuth, HTTPServer, IOError, macOS,
/// GraphQL, OpenID, ABCDef, CONSTANT, aB and A.
fn camel_split(run: &str) -> Vec<String> {
    let cs: Vec<char> = run.chars().collect();
    let n = cs.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if cs[i].is_ascii_uppercase() {
            let mut end = i;
            while end < n && cs[end].is_ascii_uppercase() {
                end += 1;
            }
            // Alternative 1: matches [i, end-1) exactly when the uppercase run
            // is at least two long AND a lowercase letter follows it, because
            // the lookahead's `[A-Z]` must be the run's last capital and its
            // `[a-z]` the character after the run.
            if end > i + 1 && end < n && cs[end].is_ascii_lowercase() {
                out.push(cs[i..end - 1].iter().collect());
                i = end - 1;
            } else if end == i + 1 && end < n && cs[end].is_ascii_lowercase() {
                // Alternative 2, with its optional leading capital.
                let mut j = end;
                while j < n && cs[j].is_ascii_lowercase() {
                    j += 1;
                }
                out.push(cs[i..j].iter().collect());
                i = j;
            } else {
                // Alternative 3.
                out.push(cs[i..end].iter().collect());
                i = end;
            }
        } else if cs[i].is_ascii_lowercase() {
            let mut j = i;
            while j < n && cs[j].is_ascii_lowercase() {
                j += 1;
            }
            out.push(cs[i..j].iter().collect());
            i = j;
        } else {
            i += 1; // findall skips what no alternative matches
        }
    }
    out
}

pub fn tokens(text: &str, drop_stopwords: bool) -> Vec<String> {
    let stripped = strip_diacritics(text);
    let mut out: Vec<String> = Vec::new();
    let sw = stopwords();

    for m in wordish().find_iter(&stripped) {
        let chunk = m.as_str();
        if chunk.chars().all(|c| c.is_numeric()) || is_hex_blob(chunk) {
            continue;
        }

        // The whole alphanumeric form FIRST, so the output stays a superset of
        // the letters-only form -- `base64` yields `base64` AND `base`.
        if chunk.is_ascii() && chunk.chars().any(|c| c.is_ascii_digit()) {
            let t = chunk.to_lowercase();
            let n = t.chars().count();
            if (MIN_LEN_ALNUM..=MAX_LEN).contains(&n) && !(drop_stopwords && sw.contains(&t)) {
                out.push(t);
            }
        }

        let found: Vec<&str> = script_run().find_iter(chunk).map(|m| m.as_str()).collect();
        let runs: Vec<&str> = if found.is_empty() { vec![chunk] } else { found };

        for run in runs {
            let mut parts: Vec<String> = if run.is_ascii() {
                let p = camel_split(run);
                let p = if p.is_empty() { vec![run.to_string()] } else { p };
                // A split that DISCARDS a piece keeps the whole run as well:
                // `OAuth` -> O + Auth, the solitary O dies on MIN_LEN, and
                // `oauth` would never be reachable.
                if p.len() > 1 && p.iter().any(|x| x.chars().count() < MIN_LEN) {
                    let mut with_run = vec![run.to_string()];
                    with_run.extend(p);
                    with_run
                } else {
                    p
                }
            } else {
                let mut p = vec![run.to_string()];
                let cs: Vec<char> = run.chars().collect();
                if cs.len() >= NGRAM_SCRIPT_MIN && is_unsegmented(run) {
                    for i in 0..=(cs.len() - NGRAM_SIZE) {
                        p.push(cs[i..i + NGRAM_SIZE].iter().collect());
                    }
                }
                p
            };
            if parts.is_empty() {
                parts = vec![run.to_string()];
            }

            for part in parts {
                let t = part.to_lowercase();
                let n = t.chars().count();
                let floor = if t.is_ascii() { MIN_LEN } else { MIN_LEN_NON_LATIN };
                if (floor..=MAX_LEN).contains(&n) {
                    if drop_stopwords && sw.contains(&t) {
                        continue;
                    }
                    out.push(t);
                }
            }
        }
    }
    out
}

pub fn unique_tokens(text: &str, drop_stopwords: bool) -> Vec<String> {
    let mut seen = HashSet::new();
    tokens(text, drop_stopwords)
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// Stable hash of everything that decides what a token IS.
///
/// Must equal Python's byte for byte: `index.fingerprint` seeds itself with
/// this, so a different value makes every installed index look stale and
/// silently reindexes the world.
pub fn rules_signature() -> String {
    use sha2::{Digest, Sha256};

    let mut words: Vec<&str> = STOPWORDS_SRC.split_whitespace().collect();
    words.sort_unstable();
    let parts: Vec<String> = vec![
        WORDISH_SRC.to_string(),
        CAMEL_SRC.to_string(),
        SCRIPT_RUN_SRC.to_string(),
        HEX_SRC.to_string(),
        MIN_LEN.to_string(),
        MAX_LEN.to_string(),
        MIN_LEN_ALNUM.to_string(),
        MIN_LEN_NON_LATIN.to_string(),
        NGRAM_SCRIPT_MIN.to_string(),
        NGRAM_SIZE.to_string(),
        HEX_BLOB_MIN.to_string(),
        words.join(","),
    ];
    let mut h = Sha256::new();
    h.update(parts.join("\u{0}").as_bytes());
    let digest = h.finalize();
    digest[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_split_matches_the_python_alternation() {
        // Captured from `_CAMEL.findall` on 0.5.0.
        let cases: [(&str, &[&str]); 12] = [
            ("OAuth", &["O", "Auth"]),
            ("HTTPServer", &["HTTP", "Server"]),
            ("IOError", &["IO", "Error"]),
            ("GlassesBridge", &["Glasses", "Bridge"]),
            ("macOS", &["mac", "OS"]),
            ("CONSTANT", &["CONSTANT"]),
            ("XMLHttpRequest", &["XML", "Http", "Request"]),
            ("GraphQL", &["Graph", "QL"]),
            ("OpenID", &["Open", "ID"]),
            ("ABCDef", &["ABC", "Def"]),
            ("A", &["A"]),
            ("aB", &["a", "B"]),
        ];
        for (input, want) in cases {
            assert_eq!(camel_split(input), want, "camel_split({input:?})");
        }
    }

    #[test]
    fn a_split_that_loses_a_piece_keeps_the_whole_run() {
        assert!(tokens("OAuth", true).contains(&"oauth".to_string()));
        assert!(tokens("OAuth", true).contains(&"auth".to_string()));
        // GlassesBridge splits cleanly and gains nothing.
        assert!(!tokens("GlassesBridge", true).contains(&"glassesbridge".to_string()));
    }

    #[test]
    fn alphanumeric_output_is_a_superset_of_the_letters_only_form() {
        let t = tokens("base64", true);
        assert!(t.contains(&"base64".to_string()) && t.contains(&"base".to_string()));
    }

    #[test]
    fn commit_hashes_do_not_become_vocabulary() {
        assert!(tokens("41768130f0d5a159ec5100160890b2315ebb4fcb", true).is_empty());
        assert!(!tokens("decade", true).is_empty(), "hex-spellable word survives");
    }

    #[test]
    fn unsegmented_runs_also_emit_bigrams() {
        let t = tokens("日本語のドキュメント", true);
        assert!(t.len() > 2, "expected bigrams, got {t:?}");
    }

    #[test]
    fn unique_tokens_preserves_order() {
        // Single letters die on MIN_LEN, so the words have to be real ones.
        assert_eq!(
            unique_tokens("alpha beta alpha gamma beta", false),
            vec!["alpha", "beta", "gamma"]
        );
    }

    #[test]
    fn the_signature_is_sixteen_hex_characters() {
        let s = rules_signature();
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
