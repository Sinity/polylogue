//! Schema operations validate the declaration's accepted selection arguments.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Infer {
    pub origin: Option<String>,
    #[serde(default)]
    pub all: bool,
    #[serde(default)]
    pub input: Vec<String>,
    pub since: Option<String>,
    pub revision: Option<u64>,
    #[serde(default)]
    pub field: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Unread {
    pub origin: Option<String>,
    #[serde(default)]
    pub all: bool,
}
