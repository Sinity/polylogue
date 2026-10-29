//! The one product SQL lowering backend and execution vocabulary.
pub mod compile;
pub mod explain;
pub mod facets;
pub mod fields;
pub mod time;
pub mod units;
pub mod ast {
    pub use polylogue_query::ast::*;
    pub trait Comparison {
        fn sql(self) -> &'static str;
    }
    impl Comparison for Op {
        fn sql(self) -> &'static str {
            if self == Op::Ne { "<>" } else { self.symbol() }
        }
    }
}
pub use compile::{
    Attach, ChatPage, ColKind, Column, ColumnReference, Definition, IdCheck, Kept, Measure, Near,
    OrderKey, Param, Program, RANK_FUSION_OFFSET, RowBasis, SavedQuery, Shape, WhyClause, compile,
    lower,
};
pub use polylogue_query::{Options, QueryError, catalog, parse};
pub fn quoted(value: &str) -> String {
    polylogue_query::quoted(value)
}
