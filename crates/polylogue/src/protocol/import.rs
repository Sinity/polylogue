//! Explicit file ingestion through the accepting daemon.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    #[serde(default)]
    pub paths: Vec<std::path::PathBuf>,
    /// The folder of an archive in the older format, imported once in place of paths.
    #[serde(default)]
    pub from_archive: Option<std::path::PathBuf>,
    /// With `from_archive`: read and check it, write nothing, and report counts per class.
    #[serde(default)]
    pub dry_run: bool,
}
pub type Answer = polylogue_engine::ImportReport;
