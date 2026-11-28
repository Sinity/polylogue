//! The listing's reply. Result rows follow the columns selected by the query.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Position {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freshness: Option<polylogue_model::freshness::ReadFreshness>,
    pub index_instance: String,
    pub index: u64,
    pub archive: u64,
    pub cursor: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    pub context: polylogue_model::context::RunContext,
    #[serde(default)]
    pub times: Vec<query_language::time::ResolvedTime>,
    /// The archive record of the original evaluation, shared by all its pages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run: Option<polylogue_model::Reference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<polylogue_model::Reference>,
}

pub use polylogue_store::query::execute::Explanation;

/// The query answer's shape is fixed. Row keys follow the declared columns;
/// measure data belongs to the named measure's own result contract. The
/// variants differ by their fields, and whether an answer is empty or
/// incomplete is the answer's outcome, never a field here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Answer {
    Rows(Box<Rows>),
    PagedMeasure(PagedMeasure),
    Measure(Measure),
}
impl Answer {
    #[cfg(test)]
    pub fn rows(&self) -> Option<&[Map<String, Value>]> {
        match self {
            Self::Rows(answer) => Some(&answer.rows),
            Self::PagedMeasure(_) | Self::Measure(_) => None,
        }
    }

    /// The rows, for callers that only ask row queries.
    pub fn into_rows(self) -> Result<Rows, crate::daemon::Failure> {
        match self {
            Self::Rows(rows) => Ok(*rows),
            Self::PagedMeasure(PagedMeasure { measure, .. })
            | Self::Measure(Measure { measure, .. }) => Err(crate::daemon::Failure::bad_request(
                format!("the query measures {measure}, where rows were expected"),
            )),
        }
    }

    pub fn next(&self) -> Option<&str> {
        match self {
            Self::Rows(rows) => rows.next.as_deref(),
            Self::PagedMeasure(measure) => measure.next.as_deref(),
            Self::Measure(_) => None,
        }
    }

    pub fn count(&self) -> Option<u64> {
        match self {
            Self::Rows(rows) => rows.count,
            Self::PagedMeasure(measure) => measure.count,
            Self::Measure(_) => None,
        }
    }

    pub fn set_count(&mut self, count: Option<u64>) {
        match self {
            Self::Rows(rows) => rows.count = count,
            Self::PagedMeasure(measure) => measure.count = count,
            Self::Measure(_) => {}
        }
    }

    /// The index position the answer was read at.
    pub fn read_position_mut(&mut self) -> &mut Position {
        match self {
            Self::Rows(rows) => &mut rows.read_position,
            Self::PagedMeasure(measure) => &mut measure.read_position,
            Self::Measure(measure) => &mut measure.read_position,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rows {
    pub read_position: Position,
    pub unit: String,
    pub shape: polylogue_query::Shape,
    pub columns: Vec<polylogue_query::Column>,
    pub rows: Vec<Map<String, Value>>,
    /// Native row identities in the same order, independent of displayed columns
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub references: Option<Vec<polylogue_model::Reference>>,
    pub count: Option<u64>,
    pub next: Option<String>,
    pub ranked: bool,
    pub notes: Vec<String>,
    pub incomplete: Vec<String>,
    pub resolved: Resolved,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub definitions: Vec<polylogue_query::Definition>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<Vec<Explanation>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PagedMeasure {
    pub measure: String,
    pub data: Value,
    pub read_position: Position,
    pub resolved: Resolved,
    #[serde(deserialize_with = "Option::deserialize")]
    #[schemars(required, schema_with = "Option::<u64>::json_schema")]
    pub count: Option<u64>,
    #[serde(deserialize_with = "Option::deserialize")]
    #[schemars(required, schema_with = "Option::<String>::json_schema")]
    pub next: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measure {
    pub measure: String,
    pub data: Value,
    pub read_position: Position,
    pub resolved: Resolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sql {
    /// One SQL statement
    #[schemars(extend("argument_form" = "positional"))]
    pub statement: String,
}
/// SQL determines the columns while the wire always carries a list of rows.
pub type SqlRows = Vec<Map<String, Value>>;
