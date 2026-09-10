//! BM25Okapi and char_wb 3-5 gram TF-IDF, reproduced rather than reinvented.
//!
//! Both replace a Python library, and the details that decide whether they
//! agree are not the obvious ones. Each is written down beside the code that
//! depends on it, because none of them is guessable from the formula.
//!
//! **rank_bm25 (`BM25Okapi`, k1=1.5 b=0.75 epsilon=0.25)**
//!
//! * `idf(t) = ln(N - df + 0.5) - ln(df + 0.5)`.
//! * `average_idf` is the mean over ALL terms, taken from the RAW values
//!   including the negative ones, before any flooring.
//! * Terms whose idf came out negative are then set to `epsilon *
//!   average_idf`. This is why a 2-chunk scope returns all zeros -- Okapi idf
//!   is exactly 0 for a term in half the corpus -- and why `search` drops the
//!   BM25 weight entirely in that case.
//! * **Query terms are NOT deduplicated.** `get_scores` iterates the raw list,
//!   and `search` passes `vtokens(question)` straight in, so a repeated term
//!   counts twice.
//! * An unknown query term contributes 0.
//!
//! **sklearn (`TfidfVectorizer`, analyzer="char_wb", ngram_range=(3,5),
//! min_df=1, lowercase=True, norm="l2", smooth_idf=True, sublinear_tf=False)**
//!
//! * Each whitespace-split word is padded to `" " + w + " "` before n-grams
//!   are taken from it.
//! * For a padded word SHORTER than n, the whole word is emitted once and the
//!   loop over n stops entirely -- sklearn `break`s out of the n loop, not just
//!   the inner one.
//! * `idf = ln((1 + N) / (1 + df)) + 1`, rows L2-normalised.
//! * The query goes through the same vocabulary, idf and normalisation, so
//!   scoring is a cosine.
//! * `max_features` is absent, deliberately. See the store's module docs and
//!   §10 R1 of the spec: it selected its last columns with an unstable sort,
//!   which is not a specification anything can be ported against.

use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;

use crate::store::{LexModel, LexView};

pub const K1: f64 = 1.5;
pub const B: f64 = 0.75;
pub const EPSILON: f64 = 0.25;
pub const NGRAM_MIN: usize = 3;
pub const NGRAM_MAX: usize = 5;

/// sklearn's `_char_wb_ngrams`, exactly.
pub fn char_wb_ngrams(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let mut out = Vec::new();
    for w in text.split_whitespace() {
        let padded: Vec<char> = std::iter::once(' ')
            .chain(w.chars())
            .chain(std::iter::once(' '))
            .collect();
        let w_len = padded.len();
        for n in min_n..=max_n {
            let mut offset = 0usize;
            out.push(padded[..n.min(w_len)].iter().collect::<String>());
            while offset + n < w_len {
                offset += 1;
                out.push(padded[offset..offset + n].iter().collect::<String>());
            }
            // A padded word shorter than n is counted once, and no larger n is
            // tried at all. sklearn breaks the OUTER loop here.
            if offset == 0 {
                break;
            }
        }
    }
    out
}

fn count<'a>(items: impl Iterator<Item = &'a str>) -> HashMap<&'a str, u32> {
    let mut m: HashMap<&str, u32> = HashMap::new();
    for it in items {
        *m.entry(it).or_insert(0) += 1;
    }
    m
}

/// Fit both rankers for one scope.
///
/// `texts` are the documents as sklearn sees them; `tokenized` are the same
/// documents through `text::tokens`, as rank_bm25 sees them.
pub fn fit(texts: &[String], tokenized: &[Vec<String>]) -> LexModel {
    let n_docs = texts.len();
    if n_docs == 0 {
        return LexModel { indptr: vec![0], ..Default::default() };
    }

    // -- BM25 --------------------------------------------------------------
    let doc_len: Vec<u32> = tokenized.iter().map(|t| t.len() as u32).collect();
    let total: u64 = doc_len.iter().map(|d| *d as u64).sum();
    let avgdl = total as f64 / n_docs as f64;

    let mut df: BTreeMap<&str, u32> = BTreeMap::new();
    let per_doc: Vec<HashMap<&str, u32>> = tokenized
        .iter()
        .map(|toks| count(toks.iter().map(|s| s.as_str())))
        .collect();
    for freqs in &per_doc {
        for term in freqs.keys() {
            *df.entry(term).or_insert(0) += 1;
        }
    }

    let n = n_docs as f64;
    let raw: Vec<f64> = df
        .values()
        .map(|d| ((n - *d as f64 + 0.5).ln()) - ((*d as f64 + 0.5).ln()))
        .collect();
    // The average is over the RAW values, negatives included, before flooring.
    let average_idf = if raw.is_empty() { 0.0 } else { raw.iter().sum::<f64>() / raw.len() as f64 };
    let eps = EPSILON * average_idf;

    let mut bm25: BTreeMap<String, (u32, f64)> = BTreeMap::new();
    for ((term, d), idf) in df.iter().zip(raw.iter()) {
        bm25.insert(term.to_string(), (*d, if *idf < 0.0 { eps } else { *idf }));
    }

    let order: HashMap<&str, usize> =
        bm25.keys().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let mut postings: Vec<Vec<(u32, u32)>> = vec![Vec::new(); bm25.len()];
    for (d, freqs) in per_doc.iter().enumerate() {
        for (term, tf) in freqs {
            postings[order[term]].push((d as u32, *tf));
        }
    }
    for p in postings.iter_mut() {
        p.sort_unstable();
    }

    let mut vocab: Vec<String> = per_doc
        .iter()
        .flat_map(|f| f.keys().map(|k| k.to_string()))
        .collect();
    vocab.sort_unstable();
    vocab.dedup();

    // -- char n-gram TF-IDF -------------------------------------------------
    let grams: Vec<HashMap<String, u32>> = texts
        .par_iter()
        .map(|t| {
            let lowered = t.to_lowercase();
            let mut m: HashMap<String, u32> = HashMap::new();
            for g in char_wb_ngrams(&lowered, NGRAM_MIN, NGRAM_MAX) {
                *m.entry(g).or_insert(0) += 1;
            }
            m
        })
        .collect();

    let mut gdf: BTreeMap<String, u32> = BTreeMap::new();
    for m in &grams {
        for g in m.keys() {
            *gdf.entry(g.clone()).or_insert(0) += 1;
        }
    }
    // Columns ARE positions in the sorted vocabulary; the store keeps no
    // second lookup table, and a permutation of columns cannot change a dot
    // product.
    let col: HashMap<&str, u32> =
        gdf.keys().enumerate().map(|(i, k)| (k.as_str(), i as u32)).collect();

    let mut ngrams: BTreeMap<String, f64> = BTreeMap::new();
    for (g, d) in &gdf {
        ngrams.insert(g.clone(), ((1.0 + n) / (1.0 + *d as f64)).ln() + 1.0);
    }
    let idf_by_col: Vec<f64> = ngrams.values().copied().collect();

    let mut indptr: Vec<u32> = Vec::with_capacity(n_docs + 1);
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    indptr.push(0);
    for m in &grams {
        let mut row: Vec<(u32, f64)> = m
            .iter()
            .map(|(g, tf)| {
                let c = col[g.as_str()];
                (c, *tf as f64 * idf_by_col[c as usize])
            })
            .collect();
        row.sort_unstable_by_key(|(c, _)| *c);
        let norm: f64 = row.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
        for (c, v) in row {
            indices.push(c);
            data.push(if norm > 0.0 { v / norm } else { 0.0 });
        }
        indptr.push(indices.len() as u32);
    }

    LexModel {
        n_docs: n_docs as u32,
        avgdl,
        doc_len,
        bm25,
        postings,
        vocab,
        ngrams,
        indptr,
        indices,
        data,
    }
}

/// BM25 scores per document. Query tokens are NOT deduplicated.
pub fn bm25_scores(v: &LexView, query: &[String]) -> Vec<f64> {
    let n = v.n_docs as usize;
    let mut score = vec![0.0f64; n];
    if n == 0 || v.avgdl <= 0.0 {
        return score;
    }
    let terms = v.bm25_terms();
    let idfs = v.bm25_idf();
    let (indptr, flat) = v.bm25_postings();
    let dl = v.doc_len();

    for q in query {
        let Some(ti) = terms.find(q) else { continue }; // unknown term: 0
        let idf = idfs[ti];
        let (a, b) = (indptr[ti] as usize, indptr[ti + 1] as usize);
        for k in a..b {
            let doc = flat[k * 2] as usize;
            let tf = flat[k * 2 + 1] as f64;
            // A document the term does not occur in contributes 0 in the
            // Python too, so iterating postings alone is equivalent.
            let denom = tf + K1 * (1.0 - B + B * dl[doc] as f64 / v.avgdl);
            score[doc] += idf * (tf * (K1 + 1.0) / denom);
        }
    }
    score
}

/// Cosine between the query and each document, over the char-gram matrix.
pub fn char_scores(v: &LexView, query: &str) -> Vec<f64> {
    let n = v.n_docs as usize;
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    let ng = v.ngrams();
    let idf = v.tfidf_idf();

    let mut counts: HashMap<usize, f64> = HashMap::new();
    for g in char_wb_ngrams(&query.to_lowercase(), NGRAM_MIN, NGRAM_MAX) {
        if let Some(c) = ng.find(&g) {
            *counts.entry(c).or_insert(0.0) += 1.0;
        }
    }
    if counts.is_empty() {
        return out; // no shared n-gram: every score is 0, as sklearn gives
    }

    let mut qvec = vec![0.0f64; ng.len()];
    let mut norm = 0.0f64;
    for (c, tf) in &counts {
        let w = tf * idf[*c];
        qvec[*c] = w;
        norm += w * w;
    }
    let norm = norm.sqrt();
    if norm > 0.0 {
        for x in qvec.iter_mut() {
            *x /= norm;
        }
    }

    let (indptr, indices, data) = v.csr();
    for (d, slot) in out.iter_mut().enumerate() {
        let (a, b) = (indptr[d] as usize, indptr[d + 1] as usize);
        let mut s = 0.0f64;
        for k in a..b {
            s += data[k] * qvec[indices[k] as usize];
        }
        *slot = s;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_wb_pads_each_word_with_spaces() {
        let g = char_wb_ngrams("ab", 3, 3);
        assert_eq!(g, vec![" ab", "ab "]);
    }

    #[test]
    fn a_word_shorter_than_n_is_emitted_once_and_stops_the_n_loop() {
        // sklearn breaks the OUTER loop. "a" pads to " a " (3 chars): n=3
        // emits it once with offset still 0, so n=4 and n=5 never run.
        let g = char_wb_ngrams("a", 3, 5);
        assert_eq!(g, vec![" a "]);
    }

    #[test]
    fn n_grams_slide_within_the_padded_word_only() {
        let g = char_wb_ngrams("abc def", 3, 3);
        assert_eq!(g, vec![" ab", "abc", "bc ", " de", "def", "ef "]);
    }

    #[test]
    fn whitespace_runs_do_not_produce_empty_words() {
        assert_eq!(char_wb_ngrams("  a\t\nb  ", 3, 3), vec![" a ", " b "]);
        assert!(char_wb_ngrams("   ", 3, 5).is_empty());
    }

    #[test]
    fn a_repeated_query_term_counts_twice() {
        let texts = vec!["alpha beta".to_string(), "alpha gamma".to_string()];
        let toks = vec![
            vec!["alpha".to_string(), "beta".to_string()],
            vec!["alpha".to_string(), "gamma".to_string()],
        ];
        let m = fit(&texts, &toks);
        let p = std::env::temp_dir().join(format!("loci-lex-dup-{}.lex", std::process::id()));
        std::fs::write(&p, crate::store::serialize(&m)).unwrap();
        let v = LexView::open(&p, Some(2)).unwrap();

        let once = bm25_scores(&v, &["beta".to_string()]);
        let twice = bm25_scores(&v, &["beta".to_string(), "beta".to_string()]);
        assert!((twice[0] - 2.0 * once[0]).abs() < 1e-12, "{once:?} {twice:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_two_document_scope_gives_bm25_all_zeros() {
        // Okapi idf is exactly 0 for a term in half the corpus, so every query
        // term in a 2-chunk scope floors out. `search` drops the BM25 weight
        // in response, and the port must produce the same zeros to trigger it.
        let texts = vec!["alpha".to_string(), "beta".to_string()];
        let toks = vec![vec!["alpha".to_string()], vec!["beta".to_string()]];
        let m = fit(&texts, &toks);
        let (_, idf) = m.bm25["alpha"];
        assert!(idf.abs() < 1e-12, "expected a floored idf, got {idf}");
    }

    #[test]
    fn an_unknown_query_term_contributes_nothing() {
        let texts = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let toks: Vec<Vec<String>> = texts.iter().map(|t| vec![t.clone()]).collect();
        let m = fit(&texts, &toks);
        let p = std::env::temp_dir().join(format!("loci-lex-unk-{}.lex", std::process::id()));
        std::fs::write(&p, crate::store::serialize(&m)).unwrap();
        let v = LexView::open(&p, Some(3)).unwrap();
        assert!(bm25_scores(&v, &["nowhere".to_string()]).iter().all(|s| *s == 0.0));
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rows_are_l2_normalised_so_a_document_matches_itself_at_one() {
        let texts = vec!["the quick brown fox".to_string(), "lorem ipsum dolor".to_string()];
        let toks: Vec<Vec<String>> = texts
            .iter()
            .map(|t| t.split(' ').map(str::to_string).collect())
            .collect();
        let m = fit(&texts, &toks);
        let p = std::env::temp_dir().join(format!("loci-lex-norm-{}.lex", std::process::id()));
        std::fs::write(&p, crate::store::serialize(&m)).unwrap();
        let v = LexView::open(&p, Some(2)).unwrap();
        let s = char_scores(&v, "the quick brown fox");
        assert!((s[0] - 1.0).abs() < 1e-9, "self-similarity should be 1, got {}", s[0]);
        assert!(s[1] < s[0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_empty_corpus_fits_without_panicking() {
        let m = fit(&[], &[]);
        assert_eq!(m.n_docs, 0);
        assert_eq!(m.indptr, vec![0]);
    }
}
