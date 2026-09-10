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

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(tokens, m)?)?;
    m.add_function(wrap_pyfunction!(unique_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(rules_signature, m)?)?;
    m.add_function(wrap_pyfunction!(is_hex_blob, m)?)?;
    m.add_function(wrap_pyfunction!(is_unsegmented, m)?)?;
    m.add_function(wrap_pyfunction!(strip_diacritics, m)?)?;
    Ok(())
}
