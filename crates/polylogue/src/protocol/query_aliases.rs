//! Query conveniences keep their own precise arguments and reuse typed rows.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Live {
    #[serde(default)]
    pub why: bool,
    pub cursor: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Notices {
    pub limit: Option<u64>,
    #[serde(default)]
    pub why: bool,
    pub cursor: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sql {
    pub statement: String,
}
/// SQL determines the columns while the wire always carries a list of rows.
pub type SqlRows = Vec<Map<String, Value>>;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct List {
    pub query: Option<String>,
    pub limit: Option<u64>,
    #[serde(default)]
    pub why: bool,
    pub cursor: Option<String>,
    #[serde(default)]
    pub fields: Vec<String>,
}
