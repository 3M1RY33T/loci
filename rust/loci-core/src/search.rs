//! The relevance gate and the fusion of three rankers into one score.
//!
//! A port of the body of `episodes.search`. The order of operations is
//! load-bearing and is copied exactly:
//!
//!   1. weights renormalise over the rankers that actually produced a signal
//!   2. `s = w_bm * bm_norm + w_ch * char`, then `+= w_em * max(0, sem)`
//!   3. `s *= min(1, len(text) / LENGTH_SATURATION)`
//!   4. `s += RECENCY_WEIGHT * recency * (1 if s > 0 else 0)`
//!   5. keep when `s >= score_floor`, then sort descending
//!
//! Step 4 multiplies by a gate on `s > 0`, so a chunk that scored nothing on
//! every ranker cannot be lifted over the floor by being recent alone. Step 3
//! runs BEFORE it, so recency is not scaled by length. Both were arrived at in
//! the Python and neither is safe to reorder.
//!
//! Every constant crosses from Python as an argument. They are calibrated,
//! several of them per corpus, and `evals/` sweeps them -- a sweep must not
//! need a rebuild.

use crate::lexical;
use crate::store::LexView;

/// One scope's fused, filtered, ordered hits: (chunk index, score).
pub type Hits = Vec<(u32, f64)>;

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub bm25: f64,
    pub char_gram: f64,
    pub embed: f64,
    pub recency: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub min_grounded: usize,
    pub min_grounded_frac: f64,
    pub semantic_floor: f64,
    /// False when the caller named the scope outright. The gate exists to stop
    /// a question being answered from the WRONG scope; once the user has
    /// picked one that risk is gone, and the only one left is a weak answer,
    /// which its score already reports.
    pub enabled: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn search(
    lex: &LexView,
    question: &str,
    q_tokens: &[String],
    sem: Option<&[f64]>,
    text_len: &[u32],
    recency: &[f64],
    w: Weights,
    length_saturation: f64,
    score_floor: f64,
    gate: Gate,
) -> Hits {
    let n = lex.n_docs as usize;
    if n == 0 || q_tokens.is_empty() {
        return Vec::new();
    }

    let grounded = {
        let v = lex.vocab();
        let mut seen = std::collections::HashSet::new();
        q_tokens
            .iter()
            .filter(|t| seen.insert(t.as_str()))
            .filter(|t| v.find(t).is_some())
            .count()
    };
    let distinct: std::collections::HashSet<&str> =
        q_tokens.iter().map(|s| s.as_str()).collect();
    let need = gate
        .min_grounded
        .max((gate.min_grounded_frac * distinct.len() as f64).round() as usize);

    let bm = lexical::bm25_scores(lex, q_tokens);
    let bm_max = bm.iter().copied().fold(0.0f64, f64::max);
    let ch = lexical::char_scores(lex, question);

    // Two-tier relevance gate, before any ranking is trusted: grounded OR
    // confident. An absolute floor alone stops working once a semantic ranker
    // gives every chunk a nonzero score.
    let semantic_ok = sem
        .map(|s| s.iter().copied().fold(f64::MIN, f64::max) >= gate.semantic_floor)
        .unwrap_or(false);
    if gate.enabled && grounded < need && !semantic_ok {
        return Vec::new();
    }

    // Renormalise over whichever rankers produced a signal. BM25 needs the
    // second guard as much as embeddings need the first: Okapi idf is exactly
    // 0 for a term in half the corpus, so in a 2-chunk scope BM25 returns all
    // zeros, and left in the fusion it drags the survivors below the floor.
    let mut w_bm = w.bm25;
    let w_ch = w.char_gram;
    let mut w_em = w.embed;
    if sem.is_none() {
        w_em = 0.0;
    }
    if bm_max <= 0.0 {
        w_bm = 0.0;
    }
    let tot = w_bm + w_ch + w_em;
    if tot <= 0.0 {
        return Vec::new();
    }
    let (w_bm, w_ch, w_em) = (w_bm / tot, w_ch / tot, w_em / tot);

    let mut scored: Hits = Vec::new();
    for i in 0..n {
        let bm_norm = if bm_max > 0.0 { bm[i] / bm_max } else { 0.0 };
        let mut s = w_bm * bm_norm + w_ch * ch[i];
        if let Some(sm) = sem {
            s += w_em * sm[i].max(0.0);
        }
        // Every ranker rewards brevity -- BM25 by length normalisation, cosine
        // because a short vector is dominated by the query terms. A one-line
        // chunk echoing the question is not the answer.
        s *= (text_len[i] as f64 / length_saturation).min(1.0);
        s += w.recency * recency[i] * if s > 0.0 { 1.0 } else { 0.0 };
        if s >= score_floor {
            scored.push((i as u32, s));
        }
    }
    // Descending by score. Python sorts with `key=lambda x: -x[0]`, which is
    // stable, so ties keep document order -- `sort_by` here is stable too.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::serialize;

    /// `tag` must be unique per test: cargo runs them in parallel and a name
    /// derived from the input alone collided, so tests overwrote each other's
    /// files and failed only when run together.
    fn build(tag: &str, texts: &[&str]) -> (LexView, std::path::PathBuf) {
        let t: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let toks: Vec<Vec<String>> = t
            .iter()
            .map(|s| s.split(' ').map(str::to_string).collect())
            .collect();
        let m = lexical::fit(&t, &toks);
        let p = std::env::temp_dir()
            .join(format!("loci-search-{}-{}.lex", std::process::id(), tag));
        std::fs::write(&p, serialize(&m)).unwrap();
        (LexView::open(&p, None).unwrap(), p)
    }

    fn weights() -> Weights {
        Weights { bm25: 0.20, char_gram: 0.20, embed: 0.60, recency: 0.05 }
    }

    fn open_gate() -> Gate {
        Gate { min_grounded: 2, min_grounded_frac: 0.25, semantic_floor: 0.57, enabled: false }
    }

    #[test]
    fn a_scope_with_no_shared_vocabulary_returns_nothing_when_gated() {
        let (v, p) = build("gate", &["alpha beta gamma", "delta epsilon zeta"]);
        let mut g = open_gate();
        g.enabled = true;
        let q: Vec<String> = "kubernetes ingress controller"
            .split(' ')
            .map(str::to_string)
            .collect();
        let hits = search(&v, "kubernetes ingress controller", &q, None,
                          &[300, 300], &[0.0, 0.0], weights(), 260.0, 0.12, g);
        assert!(hits.is_empty(), "ungrounded question should be refused");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_named_scope_bypasses_the_gate() {
        // gate=False is for a scope the caller named outright: the risk the
        // gate exists to stop -- answering from the WRONG scope -- is gone.
        let (v, p) = build("named", &["alpha beta gamma", "delta epsilon zeta"]);
        let q: Vec<String> = vec!["alpha".into()];
        let gated = { let mut g = open_gate(); g.enabled = true;
                      search(&v, "alpha", &q, None, &[300, 300], &[0.0, 0.0],
                             weights(), 260.0, 0.12, g) };
        let ungated = search(&v, "alpha", &q, None, &[300, 300], &[0.0, 0.0],
                             weights(), 260.0, 0.12, open_gate());
        assert!(ungated.len() >= gated.len());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn recency_cannot_lift_a_chunk_that_scored_nothing() {
        // The `* (1 if s > 0 else 0)` gate. Without it a recent chunk matching
        // nothing would clear the floor on recency alone.
        let (v, p) = build("recency", &["alpha beta gamma", "delta epsilon zeta"]);
        let q: Vec<String> = vec!["alpha".into()];
        let hits = search(&v, "alpha", &q, None, &[300, 300], &[1.0, 1.0],
                          weights(), 260.0, 0.12, open_gate());
        assert!(hits.iter().all(|(i, _)| *i == 0), "doc 1 matched nothing: {hits:?}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_short_chunk_is_penalised_by_length_saturation() {
        let (v, p) = build("length", &["alpha beta gamma", "alpha beta gamma"]);
        let q: Vec<String> = vec!["alpha".into()];
        let hits = search(&v, "alpha", &q, None, &[26, 260], &[0.0, 0.0],
                          weights(), 260.0, 0.0, open_gate());
        let by: std::collections::HashMap<u32, f64> = hits.into_iter().collect();
        assert!(by[&1] > by[&0], "the longer chunk should outscore the short one");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn weights_renormalise_when_a_ranker_is_silent() {
        // With no vectors the embed weight drops and the two lexical rankers
        // share the whole budget -- otherwise every score would be scaled down
        // by 0.4 and fall under the floor.
        let (v, p) = build("renorm", &["alpha beta gamma delta", "alpha epsilon zeta eta",
                             "theta iota kappa lambda"]);
        let q: Vec<String> = vec!["alpha".into()];
        let hits = search(&v, "alpha", &q, None, &[300, 300, 300], &[0.0; 3],
                          weights(), 260.0, 0.12, open_gate());
        assert!(!hits.is_empty(), "renormalisation should keep scores above the floor");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn hits_come_back_in_descending_score_order() {
        let (v, p) = build("order", &["alpha beta", "alpha alpha beta gamma", "zeta"]);
        let q: Vec<String> = vec!["alpha".into()];
        let hits = search(&v, "alpha", &q, None, &[300; 3], &[0.0; 3],
                          weights(), 260.0, 0.0, open_gate());
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "not descending: {hits:?}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
