//! Query operation DTOs. Result rows follow the columns selected by the query.
use crate::daemon::Request;
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
    pub cursor: Option<String>,
    pub session: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}

impl Arguments {
    pub fn from_request(request: &Request) -> Option<Self> {
        let Request::Query {
            text,
            cursor,
            page_size,
            why,
            fields,
            session,
        } = request
        else {
            return None;
        };
        Some(Self {
            query: text.clone(),
            limit: *page_size,
            why: *why,
            cursor: cursor.clone(),
            session: session.clone(),
            fields: fields.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RowsOutcome {
    Rows,
    Empty,
    Incomplete,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MeasureOutcome {
    Measure,
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
    pub clock: query_language::time::ClockWire,
    #[serde(default)]
    pub times: Vec<query_language::time::ResolvedTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Explanation {
    pub clause: String,
    pub matches: u64,
    pub of: u64,
}

/// The query envelope is fixed. Row keys follow the declared columns;
/// measure data belongs to the named measure's own result contract.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Answer {
    Rows(Rows),
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
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rows {
    pub outcome: RowsOutcome,
    pub read_at: Position,
    pub unit: String,
    pub shape: polylogue_query::Shape,
    pub columns: Vec<polylogue_query::Column>,
    pub rows: Vec<Map<String, Value>>,
    pub count: u64,
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
    pub outcome: MeasureOutcome,
    pub measure: String,
    pub data: Value,
    pub read_at: Position,
    pub count: u64,
    pub next: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measure {
    pub outcome: MeasureOutcome,
    pub measure: String,
    pub data: Value,
    pub read_at: Position,
}
