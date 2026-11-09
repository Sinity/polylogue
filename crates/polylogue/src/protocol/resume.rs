//! Resuming a session through its tool's own command.
use polylogue_model::SessionId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Arguments {
    /// The session; without one, the sessions in `cwd` worth resuming.
    pub session: Option<SessionId>,
    pub cwd: Option<PathBuf>,
    /// At most this many sessions.
    pub limit: Option<u64>,
}
