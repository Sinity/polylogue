//! Public evidence of the immutable context used for one operation.
use crate::{Timestamp, authorship::Authorship};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub ran_at: Timestamp,
    pub clock: query_language::time::ClockWire,
    pub machine: String,
    pub authorship: Authorship,
    pub owner: Owner,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum Owner {
    Standalone,
    Primary {
        settings_identity: String,
        owner_revision: u64,
        observed_at: Timestamp,
        received_seconds_ago: f64,
    },
}
