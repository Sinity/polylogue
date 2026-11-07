//! The on-demand check of stored bytes.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// Also hash every revision from its chain and write one line per
    /// revision to this new file on the daemon's host.
    pub manifest: Option<std::path::PathBuf>,
}
pub type Answer = polylogue_store::verify::Verification;
