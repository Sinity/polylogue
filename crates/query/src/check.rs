//! Checking a parsed query against the field table: unknown fields, wrong
//! value types and stages that do not apply to the unit fail before any SQL
//! runs.

use crate::QueryError;
use crate::ast::Query;

pub fn check(_query: &Query) -> Result<(), QueryError> {
    // Architecture 7: field table check with nearest alternative.
    todo!()
}
