//! Compiling a checked query to SQL with bound parameters. Pages are bound
//! to the read they came from; there is no row cap.

use crate::ast::Query;

pub struct Compiled {
    pub sql: String,
    pub params: Vec<String>,
}

pub fn compile(_query: &Query) -> Compiled {
    // Architecture 7: one read program, composition, aggregates, sequence matching.
    todo!()
}
