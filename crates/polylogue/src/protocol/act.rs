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
    /// A session reference, @[role:NAME] for the role's running instance (or a new one), or @[QUERY] for every session the query selects now
    #[schemars(extend("argument_form" = "positional"))]
    pub target: String,
    /// The message
    #[schemars(extend("argument_form" = "words"))]
    pub text: String,
    /// Refuse rather than resume a session that is not running
    #[serde(default)]
    pub live: bool,
    /// A reference the ping is about
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
    /// The waiting session
    #[schemars(extend("argument_form" = "positional"))]
    pub session: SessionId,
    /// approve or deny
    #[schemars(extend("argument_form" = "positional"))]
    pub decision: Decision,
    /// A reason given with the answer
    pub text: Option<String>,
    /// Answer the same tool and input next time in this session
    #[serde(default)]
    pub standing: bool,
}

/// Stop a session's current turn.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interrupt {
    /// The session
    #[schemars(extend("argument_form" = "positional"))]
    pub session: SessionId,
}

/// Start a session Polylogue hosts, in `cwd`, with `text` as its first
/// turn: on `harness`, or as an instance of `role`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Start {
    /// The first turn
    #[schemars(extend("argument_form" = "words"))]
    pub text: String,
    /// The harness or web chat to start
    #[serde(rename = "in")]
    pub harness: Option<String>,
    /// Start an instance of this role
    pub role: Option<String>,
    /// The working directory (default: the caller's directory)
    pub cwd: Option<PathBuf>,
    /// Attach these files to the first turn
    #[serde(default)]
    pub attach: Vec<PathBuf>,
    /// The model to run
    pub model: Option<String>,
    /// The model's effort setting
    pub effort: Option<String>,
    /// The provider's project
    pub project: Option<String>,
    /// Send the new web chat's first turn
    #[serde(default)]
    pub submit: bool,
}

/// Hand a session's work to a new hosted session on `to`, briefed with its
/// takeover pack (from message `at`), in `cwd` or the source's.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandOff {
    /// The session whose work is handed off
    #[schemars(extend("argument_form" = "positional"))]
    pub session: SessionId,
    /// The harness of the new session
    pub to: String,
    /// The message to continue from
    pub at: Option<String>,
    /// The working directory, when not the source's
    pub cwd: Option<PathBuf>,
    /// Interrupt the source session once the new one runs
    #[serde(default)]
    pub stop_source: bool,
    /// Show the pack and start nothing
    #[serde(default)]
    pub print: bool,
}

/// A hook takes the pings and notices pending for its session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub session: SessionId,
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
