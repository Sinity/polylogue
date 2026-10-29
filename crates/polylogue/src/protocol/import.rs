//! Explicit file ingestion through the accepting daemon.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    #[serde(default)]
    pub paths: Vec<std::path::PathBuf>,
}
pub type Answer = polylogue_engine::ImportReport;
