//! The named context operations share one answer.
use crate::briefing::{Prepared, StartSource};
use polylogue_model::SessionId;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Brief {
    /// The directory (default: the caller's working directory)
    pub cwd: Option<PathBuf>,
    /// The session asking, when known (`claude-code:<id>`)
    pub session: Option<SessionId>,
    /// Why the session starts
    #[serde(default)]
    pub source: StartSource,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Advisory {
    /// The prompt naming a file
    #[schemars(extend("argument_form" = "words"))]
    pub prompt: String,
    /// The directory (default: the caller's working directory)
    pub cwd: Option<PathBuf>,
    /// The receiving session
    pub session: SessionId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Takeover {
    /// `origin:native_id`
    #[schemars(extend("argument_form" = "positional"))]
    pub session: SessionId,
    /// The receiving session
    #[schemars(extend("unlisted" = true))]
    pub for_session: Option<SessionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Recall {
    /// What to look for
    #[serde(default)]
    #[schemars(extend("argument_form" = "words"))]
    pub query: String,
    /// The directory (default: the caller's working directory)
    pub cwd: Option<PathBuf>,
    /// The receiving session
    #[schemars(extend("unlisted" = true))]
    pub for_session: Option<SessionId>,
}

/// The context prepared for a hook, with the delivery that records its outcome.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    /// The delivery's ID, which the hook reports the outcome of; none when no
    /// delivery was queued
    pub delivery: Option<String>,
    pub prepared: Prepared,
}
