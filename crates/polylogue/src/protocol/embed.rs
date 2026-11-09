//! Embedding builds, their previews and their status.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `embed status` takes no arguments.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Status {}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// A sessions query limiting the texts (default: every prose text).
    pub query: Option<String>,
    /// At most this many texts.
    pub limit: Option<u64>,
    /// Confirm a build with a paid provider.
    #[serde(default)]
    pub yes: bool,
}
