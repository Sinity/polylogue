//! Full query completion uses the shared replacement span and candidate type.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    #[serde(default)]
    pub text: String,
    pub position: Option<u64>,
}
pub type Answer = Vec<query_language::complete::Completion>;
