//! PyO3 bindings. Conversions only -- if logic appears here, it belongs in
//! loci-core, where `cargo test` can reach it.

use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    loci_core::version()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
