//! Shared syntax and product-owned unit binding.
pub use crate::units::Unit;
pub use query_language::ast::*;

/// A product output column spelling.
pub fn flat(field: &str) -> String {
    field.replace(['.', '-'], "_")
}
