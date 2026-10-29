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
pub enum Outcome {
    Rows,
    Empty,
    Incomplete,
    Measure,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Position {
    pub generation: String,
    pub index: u64,
    pub archive: u64,
    pub cursor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Resolved {
    pub now: Option<polylogue_model::Timestamp>,
    pub zone: String,
    pub day_start_minutes: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Explanation {
    pub clause: String,
    pub matches: u64,
    pub of: u64,
}

/// The query envelope is fixed. Row keys are the query's declared columns;
/// measure data is supplied by the named measure's own result contract.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub outcome: Outcome,
    pub read_at: Position,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<polylogue_query::Shape>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<polylogue_query::Column>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<Map<String, Value>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    #[serde(default)]
    pub next: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranked: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub incomplete: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved: Option<Resolved>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub definitions: Vec<polylogue_query::Definition>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<Vec<Explanation>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measure: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}
