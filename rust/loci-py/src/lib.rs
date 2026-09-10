//! PyO3 bindings. Conversions only -- if logic appears here, it belongs in
//! loci-core, where `cargo test` can reach it.

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    loci_core::version()
}

// -- text ------------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (text, drop_stopwords=true))]
fn tokens(text: &str, drop_stopwords: bool) -> Vec<String> {
    loci_core::text::tokens(text, drop_stopwords)
}

#[pyfunction]
#[pyo3(signature = (text, drop_stopwords=true))]
fn unique_tokens(text: &str, drop_stopwords: bool) -> Vec<String> {
    loci_core::text::unique_tokens(text, drop_stopwords)
}

#[pyfunction]
fn rules_signature() -> String {
    loci_core::text::rules_signature()
}

#[pyfunction]
fn is_hex_blob(chunk: &str) -> bool {
    loci_core::text::is_hex_blob(chunk)
}

#[pyfunction]
fn is_unsegmented(text: &str) -> bool {
    loci_core::text::is_unsegmented(text)
}

#[pyfunction]
fn strip_diacritics(text: &str) -> String {
    loci_core::text::strip_diacritics(text)
}

// -- walk ------------------------------------------------------------------
#[pyfunction]
fn glob_matches(rel: &str, pattern: &str) -> bool {
    loci_core::walk::glob_matches(rel, pattern)
}

#[pyfunction]
fn iter_files(
    root: &str,
    patterns: Vec<String>,
    exclude: Vec<String>,
    skip_dirs: Vec<String>,
) -> Vec<String> {
    use std::path::{Path, PathBuf};
    let excl: Vec<PathBuf> = exclude.into_iter().map(PathBuf::from).collect();
    let skip: std::collections::HashSet<String> = skip_dirs.into_iter().collect();
    loci_core::walk::iter_files(Path::new(root), &patterns, &excl, &skip)
        .into_iter()
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// (files, visited_directories). A test seam -- see `walk::iter_files_traced`.
#[pyfunction]
fn walk_trace(
    root: &str,
    patterns: Vec<String>,
    exclude: Vec<String>,
    skip_dirs: Vec<String>,
) -> (Vec<String>, Vec<String>) {
    use std::path::{Path, PathBuf};
    let excl: Vec<PathBuf> = exclude.into_iter().map(PathBuf::from).collect();
    let skip: std::collections::HashSet<String> = skip_dirs.into_iter().collect();
    let (files, visited) =
        loci_core::walk::iter_files_traced(Path::new(root), &patterns, &excl, &skip);
    let s = |v: Vec<PathBuf>| v.into_iter().map(|p| p.to_string_lossy().into_owned()).collect();
    (s(files), s(visited))
}

// -- lexical ---------------------------------------------------------------
#[pyfunction]
#[pyo3(signature = (text, min_n=3, max_n=5))]
fn char_wb_ngrams(text: &str, min_n: usize, max_n: usize) -> Vec<String> {
    loci_core::lexical::char_wb_ngrams(text, min_n, max_n)
}

/// Fit both rankers and write the `.lex`. Returns the byte count written.
#[pyfunction]
fn lex_fit_write(path: &str, texts: Vec<String>, tokenized: Vec<Vec<String>>) -> PyResult<usize> {
    let model = loci_core::lexical::fit(&texts, &tokenized);
    let bytes = loci_core::store::serialize(&model);
    std::fs::write(path, &bytes)
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(e.to_string()))?;
    Ok(bytes.len())
}

/// An open `.lex`. Holds the mapping, so scoring never copies out of it.
#[pyclass]
struct Lex {
    inner: loci_core::store::LexView,
}

#[pymethods]
impl Lex {
    /// None when the file is absent, foreign, truncated, a version this build
    /// does not read, or fitted over a different number of documents.
    #[staticmethod]
    #[pyo3(signature = (path, n_docs=None))]
    fn open(path: &str, n_docs: Option<u32>) -> Option<Lex> {
        loci_core::store::LexView::open(std::path::Path::new(path), n_docs)
            .map(|inner| Lex { inner })
    }

    /// Fit in memory, with no file. The fallback `_fit` performs when the
    /// cached `.lex` is absent -- a query must not write to the rankers
    /// directory as a side effect of being asked.
    #[staticmethod]
    fn fit(texts: Vec<String>, tokenized: Vec<Vec<String>>) -> Option<Lex> {
        let model = loci_core::lexical::fit(&texts, &tokenized);
        let n = model.n_docs;
        loci_core::store::LexView::from_bytes(loci_core::store::serialize(&model), Some(n))
            .map(|inner| Lex { inner })
    }

    #[getter]
    fn n_docs(&self) -> u32 {
        self.inner.n_docs
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.inner.vocab().len()
    }

    fn bm25_scores(&self, query: Vec<String>) -> Vec<f64> {
        loci_core::lexical::bm25_scores(&self.inner, &query)
    }

    fn char_scores(&self, query: &str) -> Vec<f64> {
        loci_core::lexical::char_scores(&self.inner, query)
    }

    /// How many of these tokens the scope's vocabulary holds -- the grounding
    /// count `search` gates on.
    fn grounded(&self, tokens: Vec<String>) -> usize {
        let v = self.inner.vocab();
        let mut seen = std::collections::HashSet::new();
        tokens
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .filter(|t| v.find(t).is_some())
            .count()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tokens, m)?)?;
    m.add_function(wrap_pyfunction!(unique_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(rules_signature, m)?)?;
    m.add_function(wrap_pyfunction!(is_hex_blob, m)?)?;
    m.add_function(wrap_pyfunction!(is_unsegmented, m)?)?;
    m.add_function(wrap_pyfunction!(strip_diacritics, m)?)?;
    m.add_function(wrap_pyfunction!(glob_matches, m)?)?;
    m.add_function(wrap_pyfunction!(iter_files, m)?)?;
    m.add_function(wrap_pyfunction!(walk_trace, m)?)?;
    m.add_function(wrap_pyfunction!(char_wb_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(lex_fit_write, m)?)?;
    m.add_class::<Lex>()?;
    Ok(())
}
