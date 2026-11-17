//! Saved queries, watches and kept results.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Inputs to the saved-question write used by the Save step.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Save {
    /// A query name; entirely hexadecimal names are reserved
    #[schemars(extend("argument_form" = "positional"))]
    pub name: String,
    /// The query
    #[schemars(extend("argument_form" = "rest"))]
    pub query: String,
    /// The denominator query required for a measure or aggregate figure
    #[schemars(extend("unlisted" = true))]
    pub denominator: Option<String>,
}

/// Inputs to the watch write used by the Save step.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Watch {
    /// What the notices are called (default: the query)
    pub name: Option<String>,
    /// How long a row must keep matching before it is a notice, such as 20m
    #[serde(rename = "for")]
    #[schemars(extend("unlisted" = true))]
    pub hold: Option<String>,
    /// Also show each notice as a desktop notification
    #[serde(default)]
    #[schemars(extend("unlisted" = true))]
    pub notify: bool,
    /// The query
    #[schemars(extend("argument_form" = "rest"))]
    pub query: String,
}
