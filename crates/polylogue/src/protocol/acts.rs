//! Acts on live sessions: the arguments of the commands that start a
//! session or answer a session's hooks, and the replies of every act.
//! Each act is recorded as a delivery; a reply names it and says through
//! which channel it went and what came of it.
use crate::protocol::browser::ChatAnswer;
use polylogue_model::SessionId;
pub use polylogue_model::live::channel::Channel;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// A hook takes the messages and notices pending for its session.
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

/// What came of an act by the time it answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ActOutcome {
    Delivered,
    /// Recorded and held for its channel; its outcome is recorded later.
    Prepared,
    /// A resumed run started with it.
    Started,
    /// Queued behind the session's running resume.
    Queued,
    NotSent,
    Cancelled,
}

/// One session a message went to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Delivered {
    pub session: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<Channel>,
    pub outcome: ActOutcome,
    /// Why it was not sent
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<NotSentReason>,
    /// When a prepared message reaches its session
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The role whose new instance the message started
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started: Option<String>,
}

/// The cause of an unsent delivery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NotSentReason {
    #[error("session_missing")]
    SessionMissing,
    #[error("session_not_live: the session is {}", state.as_str())]
    SessionNotLive {
        state: polylogue_model::live::LiveState,
    },
    #[error("unsupported: {message}")]
    Unsupported { message: String },
    #[error("directory_unknown")]
    DirectoryUnknown,
    #[error("directory_missing({})", path.display())]
    DirectoryMissing { path: PathBuf },
    #[error("failed({message})")]
    Failed { message: String },
}

/// A message sent to one address: the message annotation (none when it
/// started a role's instance) and each session it went to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Sent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    pub deliveries: Vec<Delivered>,
    pub undo: String,
}

/// What one recipient of `send` got: a session's message, or a web chat's
/// turn held as a chat action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Reached {
    Session(Sent),
    WebChat(ChatAnswer),
}

/// Whom `send` reached, and the sessions it skipped with the line that sends to each alone.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SendReport {
    pub reached: Vec<Reached>,
    pub skipped: Vec<Skipped>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Skipped {
    pub session: String,
    pub reason: SkipReason,
    pub line: String,
}

/// Why a session in a multi-recipient send was skipped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SkipReason {
    SessionMissing,
    SessionNotRunning {
        state: polylogue_model::live::LiveState,
    },
}

/// Why a hosted permission request was cancelled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AnswerCancellation {
    /// The permission option kinds the harness offered.
    pub options: Vec<String>,
}

/// An answer to a session's pending request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Answered {
    pub delivery: String,
    /// The request answered, `request:SESSION#AT`
    pub request: String,
    pub channel: Channel,
    pub outcome: ActOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cancellation: Option<AnswerCancellation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub undo: String,
}

/// A stopped turn.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Interrupted {
    pub session: SessionId,
    pub delivery: String,
    pub channel: Channel,
    pub outcome: ActOutcome,
    pub undo: String,
}

/// A session Polylogue started and hosts, over the protocol it holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Started {
    pub session: SessionId,
    pub delivery: String,
    pub channel: Channel,
    /// The role it is an instance of
    pub role: Option<String>,
    pub undo: String,
}

/// What `start` began: a hosted session, or a web chat's new conversation
/// held as a chat action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum StartReply {
    Hosted(Started),
    WebChat(ChatAnswer),
}

/// A session's work continued in a new hosted session of another harness.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Continued {
    pub session: SessionId,
    pub delivery: String,
    pub channel: Channel,
    pub briefed_from: SessionId,
    /// The message the new session continues from
    pub at: Option<String>,
    /// The interrupt of the source session, with `--stop-source`
    pub stop_source: Option<command_surface::Answer<Interrupted>>,
    pub undo: String,
}

/// What `answer`, `interrupt` and `resume --in` did to each session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Acted {
    Answered(Answered),
    Interrupted(Interrupted),
    Continued(Box<Continued>),
    WebChat(ChatAnswer),
}

/// The messages and notices a hook takes for its session, each once.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Claimed {
    pub deliveries: Vec<ClaimedDelivery>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClaimedDelivery {
    pub id: String,
    pub text: String,
}

/// A decision given ahead for a session's pending tool request; none when
/// no standing answer applies, and the harness shows its own prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Decided {
    /// `approve` or `deny`
    pub decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery: Option<String>,
}
