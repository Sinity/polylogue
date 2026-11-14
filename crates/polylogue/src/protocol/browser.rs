//! Chat actions and backfill jobs the browser extension runs, and what the
//! extension asks the archive about the conversations it sees.
use crate::browser::actions::{ChatFile, Update};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Queues a chat action with the files it attaches; only the operator does.
/// The request ID is the action's idempotency key.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatSend {
    pub provider: String,
    pub text: String,
    pub model: String,
    pub effort: Option<String>,
    /// The conversation to reply in; none starts a new one.
    pub conversation: Option<String>,
    pub project: Option<String>,
    #[serde(default)]
    pub submit: bool,
    /// The attached files with their bytes; the command line reads `--attach`.
    #[serde(default)]
    pub files: Vec<ChatFile>,
}

/// Every queued chat action.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatList {}

/// The operator's approval or refusal of a queued chat action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatDecide {
    pub id: String,
}

/// The extension instance `by` takes the oldest queued chat action, while
/// chat actions are on.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatClaim {
    pub by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatReport {
    pub id: String,
    pub update: Update,
}

/// A chat action's decision as the extension's page reports it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatDecision {
    pub id: String,
    pub approve: bool,
    pub by: String,
    pub detail: Option<String>,
}

/// A chat action attachment's bytes, base64.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatAttachment {
    pub id: String,
    pub attachment: String,
}

/// Starts capturing a web chat's older conversations; the provider's job
/// that has not ended is answered instead of a second one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillStart {
    pub provider: String,
    pub since: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillList {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillCancel {
    pub id: String,
}

/// The extension instance `by` takes the backfill job it holds or the
/// oldest waiting one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillClaim {
    pub by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillReport {
    pub id: String,
    pub by: String,
    pub report: polylogue_model::user::BackfillReport,
}

/// What the archive holds of some of a web chat's conversations: the
/// session, its message count and when it was last captured.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Conversations {
    pub provider: String,
    pub ids: Vec<String>,
}

/// The files of a web chat's conversation whose bytes the archive holds,
/// so the extension does not download them again.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoredFiles {
    pub provider: String,
    pub conversation: String,
}

/// A selection in a chat web page saved as the operator's note on that
/// message. The request ID is the draft's ID.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectionNote {
    pub body: String,
    pub quote: String,
    pub provider: String,
    pub conversation_id: String,
    pub message_id: String,
    pub page_url: String,
    pub selected_at: String,
}

/// A blob is durable in the local primary spool, before a chat action exists.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PendingCapture {
    pub identity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCurrent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    pub action: Option<serde_json::Value>,
}

/// Confirmed state, a durable authored receipt, or file custody before enqueue.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ChatAnswer {
    Current(ChatCurrent),
    Receipt(polylogue_model::user::UserOutcome),
    Capture(PendingCapture),
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackfillCurrent {
    pub job: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum BackfillAnswer {
    Current(BackfillCurrent),
    Receipt(polylogue_model::user::UserOutcome),
}
