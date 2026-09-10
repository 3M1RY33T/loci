//! Does a delroy-sized `.lex` open in the time a page fault takes?
//!
//! The format exists to delete one measured cost: `joblib.load` on
//! `rankers/delroy.joblib` took **1.046s**, 35MB of pickled scipy, paid on
//! every one-shot invocation that touched that scope.
//!
//! This builds a model of delroy's real shape -- 8,273 documents, ~180,000
//! n-gram columns, 4.5M non-zeros -- writes it, and times opening it plus
//! probing both a CSR slice and a binary search of the n-gram table, so the
//! number is not measuring a lazy no-op.
//!
//!     cargo run --release -p loci-core --example lexbench
use loci_core::store::*;
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let n_docs = 8273usize;
    let n_ngrams = 180_000usize;
    let nnz = 4_555_199usize;

    let mut ngrams = BTreeMap::new();
    for i in 0..n_ngrams {
        ngrams.insert(format!("{:06}", i), 1.0 + (i % 7) as f32);
    }
    let mut bm25 = BTreeMap::new();
    for i in 0..40_000 {
        bm25.insert(format!("t{:06}", i), (1 + (i % 30) as u32, 0.5 + (i % 5) as f32));
    }
    let per = nnz / n_docs;
    let mut indptr = Vec::with_capacity(n_docs + 1);
    indptr.push(0u32);
    for d in 1..=n_docs { indptr.push((d * per) as u32); }
    let total = *indptr.last().unwrap() as usize;
    let m = LexModel {
        n_docs: n_docs as u32,
        avgdl: 120.0,
        doc_len: vec![120; n_docs],
        postings: (0..bm25.len()).map(|i| vec![(i as u32 % n_docs as u32, 3)]).collect(),
        bm25,
        vocab: (0..40_000).map(|i| format!("v{:06}", i)).collect(),
        ngrams,
        indptr,
        indices: (0..total).map(|i| (i % n_ngrams) as u32).collect(),
        data: vec![0.01f32; total],
    };

    let t = Instant::now();
    let bytes = serialize(&m);
    let ser = t.elapsed();
    let p = std::env::temp_dir().join("loci-lexbench.lex");
    std::fs::write(&p, &bytes).unwrap();
    println!("  size          {:.1} MB", bytes.len() as f64 / 1e6);
    println!("  serialize     {:?}", ser);

    let mut best = std::time::Duration::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let v = LexView::open(&p, Some(n_docs as u32)).expect("open");
        let (_, _, data) = v.csr();
        std::hint::black_box(data.len());
        std::hint::black_box(v.ngrams().find("100000"));
        best = best.min(t.elapsed());
    }
    println!("  open + probe  {:?}   (joblib baseline 1.046s)", best);
    let _ = std::fs::remove_file(&p);
}
