//! The `.lex` format: one scope's fitted lexical rankers, mmapped.
//!
//! It replaces `rankers/<scope>.joblib`, which cost 1.046s to `joblib.load`
//! for delroy -- 35MB of pickled scipy, paid on every one-shot invocation that
//! touched that scope. Here the numeric arrays are cast in place out of the
//! mapping, so opening is a page fault rather than a deserialization.
//!
//! Three decisions shape the layout:
//!
//! * **Sorted string tables, binary-searched.** A hash map would have to be
//!   built on open, which is the cost being removed. Sorted blobs are searched
//!   where they lie, so open stays O(1) in the vocabulary.
//! * **Columns renumbered into sorted order.** The n-gram vocabulary is stored
//!   sorted and the CSR indices renumbered to match, so a column id IS its
//!   position and no second lookup table exists. A permutation of columns does
//!   not change a dot product, so scores are unaffected.
//! * **float64 throughout.** sklearn and rank_bm25 both compute in float64.
//!   Storing f32 would put ~1e-7 of error into the data itself, and with the
//!   `--json` surface rounded to four decimals that is enough to straddle a
//!   rounding boundary -- the same failure the recency clock produced. The
//!   file is mmapped, so the extra bytes cost disk and not open time.
//! * **Refuse rather than misread.** A bad magic, an unknown version, a
//!   truncated section or a document count that disagrees with the caller all
//!   return None. That preserves the guard the Python had -- `_load_rankers`
//!   refit when `blob["n"] != n_chunks` -- because a misaligned ranking is
//!   worse than no ranking.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

pub const MAGIC: &[u8; 8] = b"LOCILEX\0";
pub const VERSION: u32 = 1;
const N_SECTIONS: usize = 9;
const HEADER_LEN: usize = 48 + N_SECTIONS * 16;

/// Section indices, named so the reader and writer cannot drift.
mod sec {
    pub const DOC_LEN: usize = 0;
    pub const BM25_TERMS: usize = 1;
    pub const BM25_DF: usize = 2;
    pub const BM25_IDF: usize = 3;
    pub const BM25_POSTINGS: usize = 4;
    pub const VOCAB: usize = 5;
    pub const NGRAMS: usize = 6;
    pub const TFIDF_IDF: usize = 7;
    pub const CSR: usize = 8;
}

/// What a fit produces, before it is written.
#[derive(Debug, Default, Clone)]
pub struct LexModel {
    pub n_docs: u32,
    pub avgdl: f64,
    /// Per-document token count, in document order.
    pub doc_len: Vec<u32>,
    /// BM25 term -> (document frequency, idf). Ordered, so the table is sorted
    /// by construction.
    pub bm25: BTreeMap<String, (u32, f64)>,
    /// BM25 postings per term, in the same order as `bm25`: (doc, term freq).
    pub postings: Vec<Vec<(u32, u32)>>,
    /// Token vocabulary, for the grounding check in `search`.
    pub vocab: Vec<String>,
    /// char n-gram -> idf. Ordered; the CSR is renumbered to match.
    pub ngrams: BTreeMap<String, f64>,
    /// CSR of the L2-normalised tf-idf matrix, columns in `ngrams` order.
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub data: Vec<f64>,
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------
fn pad_to_8(buf: &mut Vec<u8>) {
    while buf.len() % 8 != 0 {
        buf.push(0);
    }
}

/// offsets: u32[n + 1], then the concatenated bytes. Callers pass sorted
/// strings; `find` relies on it.
fn write_str_table(buf: &mut Vec<u8>, items: impl Iterator<Item = impl AsRef<str>>) {
    let items: Vec<String> = items.map(|s| s.as_ref().to_string()).collect();
    let mut offsets: Vec<u32> = Vec::with_capacity(items.len() + 1);
    let mut blob: Vec<u8> = Vec::new();
    offsets.push(0);
    for s in &items {
        blob.extend_from_slice(s.as_bytes());
        offsets.push(blob.len() as u32);
    }
    buf.extend_from_slice(bytemuck::cast_slice(&offsets));
    buf.extend_from_slice(&blob);
}

pub fn serialize(m: &LexModel) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();
    let mut sections = [(0u64, 0u64); N_SECTIONS];

    let mut push = |body: &mut Vec<u8>, idx: usize, f: &dyn Fn(&mut Vec<u8>)| {
        pad_to_8(body);
        let start = body.len();
        f(body);
        sections[idx] = ((HEADER_LEN + start) as u64, (body.len() - start) as u64);
    };

    push(&mut body, sec::DOC_LEN, &|b| {
        b.extend_from_slice(bytemuck::cast_slice(&m.doc_len))
    });
    push(&mut body, sec::BM25_TERMS, &|b| {
        write_str_table(b, m.bm25.keys())
    });
    push(&mut body, sec::BM25_DF, &|b| {
        let v: Vec<u32> = m.bm25.values().map(|(df, _)| *df).collect();
        b.extend_from_slice(bytemuck::cast_slice(&v));
    });
    push(&mut body, sec::BM25_IDF, &|b| {
        let v: Vec<f64> = m.bm25.values().map(|(_, idf)| *idf).collect();
        b.extend_from_slice(bytemuck::cast_slice(&v));
    });
    push(&mut body, sec::BM25_POSTINGS, &|b| {
        let mut indptr: Vec<u32> = Vec::with_capacity(m.postings.len() + 1);
        let mut flat: Vec<u32> = Vec::new();
        indptr.push(0);
        for p in &m.postings {
            for (doc, tf) in p {
                flat.push(*doc);
                flat.push(*tf);
            }
            indptr.push((flat.len() / 2) as u32);
        }
        b.extend_from_slice(bytemuck::cast_slice(&indptr));
        b.extend_from_slice(bytemuck::cast_slice(&flat));
    });
    push(&mut body, sec::VOCAB, &|b| write_str_table(b, m.vocab.iter()));
    push(&mut body, sec::NGRAMS, &|b| write_str_table(b, m.ngrams.keys()));
    push(&mut body, sec::TFIDF_IDF, &|b| {
        let v: Vec<f64> = m.ngrams.values().copied().collect();
        b.extend_from_slice(bytemuck::cast_slice(&v));
    });
    push(&mut body, sec::CSR, &|b| {
        b.extend_from_slice(bytemuck::cast_slice(&m.indptr));
        b.extend_from_slice(bytemuck::cast_slice(&m.indices));
        // indptr and indices are u32; `data` is f64 and bytemuck checks the
        // POINTER's alignment even for an empty slice. With an odd combined
        // count the f64 slice would start 4 mod 8 and the cast would panic --
        // which the empty model hit first, because 1 indptr entry and 0
        // indices is exactly that case.
        pad_to_8(b);
        b.extend_from_slice(bytemuck::cast_slice(&m.data));
    });

    // Header layout, fixed. avgdl and nnz are 8 bytes and sit at 8-byte
    // aligned offsets on purpose; the u32 fields are packed ahead of them so
    // no padding is needed to get there.
    //
    //   0  magic 8   8  version   12 n_docs   16 n_bm25
    //   20 n_vocab   24 n_ngrams  28 n_sections
    //   32 avgdl f64            40 nnz u64
    //   48 section table, N_SECTIONS x (offset u64, len u64)
    let mut out: Vec<u8> = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&m.n_docs.to_le_bytes());
    out.extend_from_slice(&(m.bm25.len() as u32).to_le_bytes());
    out.extend_from_slice(&(m.vocab.len() as u32).to_le_bytes());
    out.extend_from_slice(&(m.ngrams.len() as u32).to_le_bytes());
    out.extend_from_slice(&(N_SECTIONS as u32).to_le_bytes());
    out.extend_from_slice(&m.avgdl.to_le_bytes());
    out.extend_from_slice(&(m.indices.len() as u64).to_le_bytes());
    for (off, len) in sections {
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
    debug_assert_eq!(out.len(), HEADER_LEN);
    out.extend_from_slice(&body);
    out
}

// ---------------------------------------------------------------------------
// reading
// ---------------------------------------------------------------------------
/// A sorted string table, searched where it lies.
pub struct StrTable<'a> {
    offsets: &'a [u32],
    blob: &'a [u8],
}

impl<'a> StrTable<'a> {
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, i: usize) -> &'a str {
        let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        std::str::from_utf8(&self.blob[a..b]).unwrap_or("")
    }
    /// Position of `needle`, or None. Binary search, because building a hash
    /// map on open is the cost this format exists to remove.
    pub fn find(&self, needle: &str) -> Option<usize> {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.get(mid).cmp(needle) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }
}

/// Where a view's bytes live. A cached `.lex` is mapped; a fit that had no
/// file to read -- the in-process fallback `_fit` performs when the cache is
/// absent -- owns its buffer instead. Everything above this is identical.
enum Backing {
    Mapped(Mmap),
    Owned(Vec<u8>),
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mapped(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

pub struct LexView {
    map: Backing,
    pub n_docs: u32,
    pub avgdl: f64,
    n_bm25: usize,
    n_vocab: usize,
    n_ngrams: usize,
    nnz: usize,
    sections: [(u64, u64); N_SECTIONS],
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

impl LexView {
    /// Open a `.lex`, or None if it is not one this build can read.
    ///
    /// Every rejection path is deliberate: a misaligned ranking is worse than
    /// no ranking, and the caller's fallback is to refit.
    pub fn open(path: &Path, expect_docs: Option<u32>) -> Option<LexView> {
        let file = File::open(path).ok()?;
        let map = Backing::Mapped(unsafe { Mmap::map(&file).ok()? });
        Self::from_backing(map, expect_docs)
    }

    /// The same view over bytes already in memory.
    pub fn from_bytes(bytes: Vec<u8>, expect_docs: Option<u32>) -> Option<LexView> {
        Self::from_backing(Backing::Owned(bytes), expect_docs)
    }

    fn from_backing(map: Backing, expect_docs: Option<u32>) -> Option<LexView> {
        if map.len() < HEADER_LEN || &map[..8] != MAGIC {
            return None;
        }
        if u32_at(&map, 8) != VERSION {
            return None;
        }
        let n_docs = u32_at(&map, 12);
        if let Some(want) = expect_docs {
            if want != n_docs {
                return None; // the store changed under us; refit rather than misalign
            }
        }
        let n_bm25 = u32_at(&map, 16) as usize;
        let n_vocab = u32_at(&map, 20) as usize;
        let n_ngrams = u32_at(&map, 24) as usize;
        if u32_at(&map, 28) as usize != N_SECTIONS {
            return None;
        }
        let avgdl = f64::from_le_bytes(map[32..40].try_into().ok()?);
        let nnz = u64::from_le_bytes(map[40..48].try_into().ok()?) as usize;
        let mut sections = [(0u64, 0u64); N_SECTIONS];
        for (i, s) in sections.iter_mut().enumerate() {
            let base = 48 + i * 16;
            let off = u64::from_le_bytes(map[base..base + 8].try_into().ok()?);
            let len = u64::from_le_bytes(map[base + 8..base + 16].try_into().ok()?);
            if off as usize + len as usize > map.len() {
                return None; // truncated
            }
            *s = (off, len);
        }
        Some(LexView { map, n_docs, avgdl, n_bm25, n_vocab, n_ngrams, nnz, sections })
    }

    fn bytes(&self, i: usize) -> &[u8] {
        let (off, len) = self.sections[i];
        &self.map[off as usize..off as usize + len as usize]
    }

    fn table(&self, i: usize, n: usize) -> StrTable<'_> {
        let b = self.bytes(i);
        let split = (n + 1) * 4;
        StrTable { offsets: bytemuck::cast_slice(&b[..split]), blob: &b[split..] }
    }

    pub fn doc_len(&self) -> &[u32] {
        bytemuck::cast_slice(self.bytes(sec::DOC_LEN))
    }
    pub fn bm25_terms(&self) -> StrTable<'_> {
        self.table(sec::BM25_TERMS, self.n_bm25)
    }
    pub fn bm25_df(&self) -> &[u32] {
        bytemuck::cast_slice(self.bytes(sec::BM25_DF))
    }
    pub fn bm25_idf(&self) -> &[f64] {
        bytemuck::cast_slice(self.bytes(sec::BM25_IDF))
    }
    /// (indptr, flat pairs) -- postings for term `t` are
    /// `flat[indptr[t] * 2 .. indptr[t + 1] * 2]` as (doc, tf) couples.
    pub fn bm25_postings(&self) -> (&[u32], &[u32]) {
        let b: &[u32] = bytemuck::cast_slice(self.bytes(sec::BM25_POSTINGS));
        b.split_at(self.n_bm25 + 1)
    }
    pub fn vocab(&self) -> StrTable<'_> {
        self.table(sec::VOCAB, self.n_vocab)
    }
    pub fn ngrams(&self) -> StrTable<'_> {
        self.table(sec::NGRAMS, self.n_ngrams)
    }
    pub fn tfidf_idf(&self) -> &[f64] {
        bytemuck::cast_slice(self.bytes(sec::TFIDF_IDF))
    }
    /// (indptr, indices, data)
    pub fn csr(&self) -> (&[u32], &[u32], &[f64]) {
        let b = self.bytes(sec::CSR);
        let n_ptr = (self.n_docs as usize + 1) * 4;
        let n_idx = self.nnz * 4;
        // Skip the padding the writer inserted to land `data` 8-byte aligned.
        let head = n_ptr + n_idx;
        let head = head + (8 - head % 8) % 8;
        (
            bytemuck::cast_slice(&b[..n_ptr]),
            bytemuck::cast_slice(&b[n_ptr..n_ptr + n_idx]),
            bytemuck::cast_slice(&b[head..head + self.nnz * 8]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LexModel {
        let mut bm25 = BTreeMap::new();
        bm25.insert("alpha".to_string(), (2u32, 0.51f64));
        bm25.insert("beta".to_string(), (1u32, 1.20f64));
        let mut ngrams = BTreeMap::new();
        ngrams.insert(" al".to_string(), 1.4f64);
        ngrams.insert(" be".to_string(), 1.9f64);
        ngrams.insert("lph".to_string(), 2.1f64);
        LexModel {
            n_docs: 3,
            avgdl: 4.0,
            doc_len: vec![4, 5, 3],
            bm25,
            postings: vec![vec![(0, 2), (2, 1)], vec![(1, 3)]],
            vocab: vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()],
            ngrams,
            indptr: vec![0, 2, 3, 4],
            indices: vec![0, 2, 1, 0],
            data: vec![0.5, 0.5, 1.0, 1.0],
        }
    }

    fn write_tmp(bytes: &[u8], tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir()
            .join(format!("loci-lex-{}-{}.lex", std::process::id(), tag));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn a_written_model_reads_back_identically() {
        let m = sample();
        let p = write_tmp(&serialize(&m), "roundtrip");
        let v = LexView::open(&p, Some(3)).expect("should open");

        assert_eq!(v.n_docs, 3);
        assert_eq!(v.avgdl, 4.0);
        assert_eq!(v.doc_len(), &[4, 5, 3]);

        let terms = v.bm25_terms();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms.get(0), "alpha");
        assert_eq!(terms.find("beta"), Some(1));
        assert_eq!(terms.find("missing"), None);
        assert_eq!(v.bm25_df(), &[2, 1]);
        assert_eq!(v.bm25_idf(), &[0.51, 1.20]);

        let (indptr, flat) = v.bm25_postings();
        assert_eq!(indptr, &[0, 2, 3]);
        assert_eq!(&flat[0..4], &[0, 2, 2, 1]); // alpha: (0,2) (2,1)
        assert_eq!(&flat[4..6], &[1, 3]); // beta: (1,3)

        assert_eq!(v.vocab().find("gamma"), Some(2));
        assert_eq!(v.vocab().find("delta"), None);

        let ng = v.ngrams();
        assert_eq!(ng.len(), 3);
        assert_eq!(ng.find(" be"), Some(1));
        assert_eq!(v.tfidf_idf(), &[1.4, 1.9, 2.1]);

        let (ip, ix, dt) = v.csr();
        assert_eq!(ip, &[0, 2, 3, 4]);
        assert_eq!(ix, &[0, 2, 1, 0]);
        assert_eq!(dt, &[0.5, 0.5, 1.0, 1.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_doc_count_mismatch_is_refused() {
        // The guard the Python had: a store that changed underneath must
        // produce a refit, never a silently misaligned ranking.
        let p = write_tmp(&serialize(&sample()), "mismatch");
        assert!(LexView::open(&p, Some(3)).is_some());
        assert!(LexView::open(&p, Some(4)).is_none());
        assert!(LexView::open(&p, None).is_some());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn a_truncated_file_is_refused_rather_than_misread() {
        let full = serialize(&sample());
        for cut in [0, 8, HEADER_LEN - 1, HEADER_LEN, full.len() - 8] {
            let p = write_tmp(&full[..cut], &format!("trunc{cut}"));
            assert!(LexView::open(&p, None).is_none(), "cut at {cut} should be refused");
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn a_foreign_file_is_refused() {
        let p = write_tmp(b"not a lex file at all, but long enough to pass a length check....", "magic");
        assert!(LexView::open(&p, None).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_unknown_version_is_refused() {
        let mut bytes = serialize(&sample());
        bytes[8] = 99;
        let p = write_tmp(&bytes, "version");
        assert!(LexView::open(&p, None).is_none());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_empty_model_round_trips() {
        let m = LexModel { n_docs: 0, indptr: vec![0], ..Default::default() };
        let p = write_tmp(&serialize(&m), "empty");
        let v = LexView::open(&p, Some(0)).expect("an empty model is still a model");
        assert_eq!(v.bm25_terms().len(), 0);
        assert!(v.vocab().is_empty());
        assert_eq!(v.csr().1.len(), 0);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn an_odd_csr_head_still_yields_an_aligned_data_slice() {
        // indptr + indices are u32 and `data` is f64. With an odd combined
        // count the f64 slice starts 4 mod 8, and bytemuck checks the pointer
        // even when the slice is empty. Both parities are exercised.
        for nnz in [0usize, 1, 2, 3] {
            let m = LexModel {
                n_docs: 2,
                indptr: vec![0, nnz as u32, nnz as u32],
                indices: (0..nnz as u32).collect(),
                data: vec![0.25f64; nnz],
                ..Default::default()
            };
            let p = write_tmp(&serialize(&m), &format!("csr{nnz}"));
            let v = LexView::open(&p, Some(2)).expect("should open");
            let (ip, ix, dt) = v.csr();
            assert_eq!(ip.len(), 3);
            assert_eq!(ix.len(), nnz);
            assert_eq!(dt.len(), nnz);
            assert!(dt.iter().all(|x| *x == 0.25));
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn every_section_starts_eight_byte_aligned() {
        // bytemuck casts in place; a misaligned section would panic on read.
        let bytes = serialize(&sample());
        for i in 0..N_SECTIONS {
            let base = 48 + i * 16;
            let off = u64::from_le_bytes(bytes[base..base + 8].try_into().unwrap());
            assert_eq!(off % 8, 0, "section {i} at {off} is not 8-byte aligned");
        }
    }
}
