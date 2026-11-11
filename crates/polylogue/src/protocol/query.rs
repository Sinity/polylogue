//! Query operation DTOs. Result rows follow the columns selected by the query.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    #[serde(default)]
    pub query: String,
    pub limit: Option<u64>,
    #[serde(default)]
    pub why: bool,
    pub session: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Position {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub currency: Option<polylogue_model::currency::IndexCurrency>,
    pub generation: String,
    pub index: u64,
    pub archive: u64,
    pub cursor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    pub context: polylogue_model::context::Provenance,
    #[serde(default)]
    pub times: Vec<query_language::time::ResolvedTime>,
}

pub use polylogue_store::query::execute::Explanation;

/// The query answer's shape is fixed. Row keys follow the declared columns;
/// measure data belongs to the named measure's own result contract. The
/// variants differ by their fields, and whether an answer is empty or
/// incomplete is the answer's outcome, never a field here.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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
            | Self::Measure(Measure { measure, .. }) => Err(crate::daemon::Failure::new(
                "bad_request",
                format!("the query measures {measure}, where rows were expected"),
            )),
        }
    }

    /// The index position the answer was read at.
    pub fn read_at_mut(&mut self) -> &mut Position {
        match self {
            Self::Rows(rows) => &mut rows.read_at,
            Self::PagedMeasure(measure) => &mut measure.read_at,
            Self::Measure(measure) => &mut measure.read_at,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rows {
    pub read_at: Position,
    pub unit: String,
    pub shape: polylogue_query::Shape,
    pub columns: Vec<polylogue_query::Column>,
    pub rows: Vec<Map<String, Value>>,
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PagedMeasure {
    pub measure: String,
    pub data: Value,
    pub read_at: Position,
    pub resolved: Resolved,
    pub count: Option<u64>,
    pub next: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measure {
    pub measure: String,
    pub data: Value,
    pub read_at: Position,
    pub resolved: Resolved,
}
