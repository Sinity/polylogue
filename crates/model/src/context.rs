//! The immutable context used for one operation, as callers see it.
use crate::{Timestamp, authorship::Authorship};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunContext {
    pub ran_at: Timestamp,
    pub clock: query_language::time::ClockWire,
    pub machine: String,
    pub authorship: Authorship,
    pub sinex: SinexContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SinexContext {
    Standalone,
    Primary {
        settings_identity: String,
        settings_revision: u64,
        observed_at: Timestamp,
        received_seconds_ago: f64,
    },
}
