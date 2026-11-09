//! Saved queries, watches and kept results.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `save NAME QUERY`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Save {
    pub name: String,
    pub query: String,
    /// The saved query a ratio divides by.
    pub denominator: Option<String>,
}

/// `watch QUERY`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Watch {
    pub name: Option<String>,
    /// How long a row must keep matching first, such as `20m`.
    #[serde(rename = "for")]
    pub hold: Option<String>,
    /// Also show each notice as a desktop notification.
    #[serde(default)]
    pub notify: bool,
    pub query: String,
}

/// `keep QUERY`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Keep {
    pub name: Option<String>,
    pub query: String,
}
