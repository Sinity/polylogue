//! Shared syntax and product-owned unit binding.
pub use query_language::ast::*;
pub use crate::units::Unit;

/// A product output column spelling.
pub fn flat(field: &str) -> String { field.replace(['.', '-'], "_") }

pub trait Comparison { fn sql(self) -> &'static str; }
impl Comparison for Op {
    fn sql(self) -> &'static str { if self == Op::Ne { "<>" } else { self.symbol() } }
}
