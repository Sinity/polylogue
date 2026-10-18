//! The field table, as data: every field each unit accepts, its type and
//! the SQL it reads. Completion, help and error suggestions read this
//! table; nothing keeps a parallel list.

use crate::ast::Unit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Text,
    Count,
    Date,
    Enum(&'static [&'static str]),
    /// A path into a note's JSON value (`value.score`).
    JsonPath,
}

pub struct Field {
    pub name: &'static str,
    pub unit: Unit,
    pub ty: FieldType,
    /// SQL expression over the unit's table alias.
    pub sql: &'static str,
    pub help: &'static str,
}

pub static FIELDS: &[Field] = &[];

pub fn lookup(_unit: Unit, _name: &str) -> Option<&'static Field> {
    // Architecture 7: field table lookup and nearest-name suggestion.
    todo!()
}
