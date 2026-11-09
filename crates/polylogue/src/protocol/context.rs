//! The named context operations share one answer.
use crate::briefing::{Prepared, StartSource};
use polylogue_model::SessionId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Brief {
    pub cwd: Option<PathBuf>,
    pub session: Option<SessionId>,
    #[serde(default)]
    pub source: StartSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Advisory {
    pub prompt: String,
    pub cwd: Option<PathBuf>,
    pub session: SessionId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Takeover {
    pub session: SessionId,
    pub for_session: Option<SessionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Recall {
    #[serde(default)]
    pub query: String,
    pub cwd: Option<PathBuf>,
    pub for_session: Option<SessionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub delivery: Option<String>,
    pub prepared: Prepared,
}
