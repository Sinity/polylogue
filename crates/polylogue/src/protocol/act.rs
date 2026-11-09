//! Acts on live sessions: pings, answers, interrupts, new hosted sessions,
//! and the requests a session's hooks make.
use polylogue_model::{DeliveryOutcome, SessionId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// An addressed message to a session, or to every session a quoted query
/// (`@[sessions …]`) selects when it is sent.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Ping {
    pub target: String,
    pub text: String,
    /// Refuse rather than resume a session that is not running.
    #[serde(default)]
    pub live: bool,
    pub about: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Approve,
    Deny,
}

/// Approve or deny a session's pending request; `standing` answers the
/// same tool and input whenever the session asks next.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Answer {
    pub session: SessionId,
    pub decision: Decision,
    pub text: Option<String>,
    #[serde(default)]
    pub standing: bool,
}

/// Stop a session's current turn.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interrupt {
    pub session: SessionId,
}

/// Start a session Polylogue hosts, in `cwd`, with `text` as its first
/// turn: on `harness`, or as an instance of `role`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Start {
    pub text: String,
    pub harness: Option<String>,
    pub role: Option<String>,
    pub cwd: PathBuf,
}

/// Hand a session's work to a new hosted session on `to`, briefed with its
/// takeover pack (from message `at`), in `cwd` or the source's.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandOff {
    pub session: SessionId,
    pub to: String,
    pub at: Option<String>,
    pub cwd: Option<PathBuf>,
    /// Interrupt the source afterwards.
    #[serde(default)]
    pub stop_source: bool,
    /// Show the pack and start nothing.
    #[serde(default)]
    pub print: bool,
}

/// A hook takes the pings and notices pending for its session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub session: SessionId,
    pub event: String,
}

/// The synchronous permission hook asks for a decision given ahead.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Permission {
    pub session: SessionId,
    pub tool: String,
    #[serde(default)]
    pub input: serde_json::Value,
}

/// What happened to a delivery after it was handed out.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeliveryEvent {
    pub delivery: String,
    pub what: DeliveryOutcome,
    pub detail: Option<String>,
}
