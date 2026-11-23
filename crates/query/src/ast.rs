//! Shared syntax and product-owned unit binding.
pub use crate::units::Unit;
pub use query_language::ast::*;

/// A product output column spelling.
pub fn column_name(field: &str) -> String {
    field.replace(['.', '-'], "_")
}
