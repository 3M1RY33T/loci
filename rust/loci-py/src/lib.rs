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
    Ok(())
}
