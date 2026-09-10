//! The arithmetic behind loci: tokenizer, walk, store, lexical ranking, fusion.
//!
//! No PyO3 here on purpose. This crate is testable with plain `cargo test` and
//! can be lifted into a standalone binary later without unpicking Python
//! types; `loci-py` holds every conversion.

pub mod text;
pub mod lexical;
pub mod store;
pub mod walk;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_the_crate_version() {
        assert_eq!(super::version(), env!("CARGO_PKG_VERSION"));
    }
}
