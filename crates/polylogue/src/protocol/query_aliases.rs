//! Query conveniences keep their own precise arguments and reuse typed rows.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Live {
    /// Explain an empty result: how many rows each condition keeps
    #[serde(default)]
    pub why: bool,
    /// The columns to show instead of the unit's defaults, comma-separated (`--fields id,cost,repo`)
    #[serde(default)]
    pub fields: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Notices {
    /// Rows per page
    #[serde(default = "notice_limit")]
    pub limit: u64,
    /// Explain an empty result: how many rows each condition keeps
    #[serde(default)]
    pub why: bool,
    /// The columns to show instead of the unit's defaults, comma-separated (`--fields id,cost,repo`)
    #[serde(default)]
    pub fields: Vec<String>,
}
fn notice_limit() -> u64 {
    50
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct List {
    /// A query in the query language
    #[schemars(extend("argument_form" = "words"))]
    pub query: Option<String>,
    /// Rows per page
    pub limit: Option<u64>,
    /// Explain an empty result: how many rows each condition keeps
    #[serde(default)]
    pub why: bool,
    /// The columns to show instead of the unit's defaults, comma-separated (`--fields id,cost,repo`)
    #[serde(default)]
    pub fields: Vec<String>,
}

/// `measure NAME`: a sessions query (none for a whole-archive measure).
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Measure {
    /// A sessions query (default: every session)
    #[schemars(extend("argument_form" = "words"))]
    pub query: Option<String>,
    /// The series, for a measure with several
    pub metric: Option<String>,
    /// The measure's own option
    pub option: Option<String>,
    /// Rows per page
    pub limit: Option<u64>,
    /// Explain an empty result: how many rows each condition keeps
    #[serde(default)]
    pub why: bool,
    /// The columns to show instead of the unit's defaults, comma-separated (`--fields id,cost,repo`)
    #[serde(default)]
    pub fields: Vec<String>,
}
