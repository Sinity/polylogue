//! The one product SQL lowering backend and execution vocabulary.
pub mod compile;
pub mod continuation;
pub mod execute;
pub mod explain;
pub mod facets;
pub mod fields;
pub(crate) mod references;
pub mod time;
pub mod units;
pub mod ast {
    pub use polylogue_query::ast::*;
    /// The SQL spelling of a comparison operator.
    pub trait SqlOperator {
        fn sql(self) -> &'static str;
    }
    impl SqlOperator for Op {
        fn sql(self) -> &'static str {
            if self == Op::Ne { "<>" } else { self.symbol() }
        }
    }
}
pub use compile::{
    Attach, ColKind, Column, ColumnReference, Definition, DerivedRows, IdCheck, Kept, KeyedRows,
    Measure, Near, OrderKey, PageColumns, Param, Program, RANK_FUSION_OFFSET, RowBasis, SavedQuery,
    Shape, SnippetTerms, Statements, WhyClause, compile, lower,
};
pub use polylogue_query::{Options, QueryError, catalog, parse};
pub fn quoted(value: &str) -> String {
    polylogue_query::quoted(value)
}
