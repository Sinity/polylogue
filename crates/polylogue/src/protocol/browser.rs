//! Chat actions and import jobs the browser extension runs, and what the
//! extension asks the archive about the conversations it sees.
use crate::browser::actions::{ActionSummary, Update};
use crate::browser::import_job::JobSummary;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The extension instance `by` takes the oldest queued chat action, while
/// chat actions are on.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatClaim {
    /// The extension instance taking the action
    pub by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatReport {
    /// The chat action
    pub id: String,
    /// Its progress or outcome
    pub update: Update,
}

/// The operator's answer to a held turn from the extension's popup: the
/// same answer as `answer allow|deny` on its web-chat session.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HeldTurnAnswer {
    /// The chat action
    pub id: String,
    /// allow or deny
    pub value: String,
    /// A reason given with the answer
    pub reason: Option<String>,
}

/// A chat action attachment's bytes, base64.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatAttachment {
    /// The chat action
    pub id: String,
    /// The attachment's ID within the action
    pub attachment: String,
}

/// The extension instance `by` takes the import job it holds or the
/// oldest waiting one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobClaim {
    /// The extension instance taking the job
    pub by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobReport {
    /// The job
    pub id: String,
    /// The extension instance holding the job's claim
    pub by: String,
    /// Its progress or outcome
    pub report: polylogue_model::user::ImportJobReport,
}

/// Asks what the archive holds of some of a web chat's conversations (the
/// session, its message count and when it was last captured).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Conversations {
    /// The web chat, by the extension's name for it
    pub provider: String,
    /// The provider's conversation IDs
    pub ids: Vec<String>,
}

/// Asks which files of a web chat's conversation the archive already holds
/// bytes for, so the extension does not download them again.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoredFiles {
    /// The web chat, by the extension's name for it
    pub provider: String,
    /// The provider's conversation ID
    pub conversation: String,
}

/// A selection in a chat web page saved as the operator's note on that
/// message. The request ID is the note's idempotency key.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectionNote {
    /// The note's text
    pub body: String,
    /// The selected text it quotes
    pub quote: String,
    /// The web chat, by the extension's name for it
    pub provider: String,
    /// The provider's conversation ID
    pub conversation_id: String,
    /// The provider's message ID the selection is in
    pub message_id: String,
    /// The page the selection was made on
    pub page_url: String,
    /// When the selection was made
    pub selected_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatCurrent {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    pub action: Option<Box<ActionSummary>>,
}

/// The current state, or a saved write result.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ChatAnswer {
    Current(ChatCurrent),
    Outcome(polylogue_model::user::UserOutcome),
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobCurrent {
    pub job: Option<Box<JobSummary>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ImportJobAnswer {
    Current(ImportJobCurrent),
    Outcome(polylogue_model::user::UserOutcome),
}

/// `chat attachment`: an attached file's bytes.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatAttachmentBytes {
    pub media_type: String,
    /// The bytes, standard base64.
    pub bytes: String,
}

/// `conversations`: what the archive holds of each asked conversation, in
/// the order asked.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConversationsAnswer {
    pub provider: String,
    pub conversations: Vec<ConversationState>,
}

/// One conversation as the archive holds it; a conversation it lacks has
/// no message count.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConversationState {
    /// The provider's conversation ID.
    pub id: String,
    /// The session it is stored as; none for an ID no session can name.
    pub session: Option<String>,
    pub messages: Option<u64>,
    /// Its latest capture said files were still downloading; none when no
    /// stored session can name the conversation.
    pub downloading: Option<bool>,
    /// When the latest stored capture of it was taken (ms).
    pub last_captured_at: Option<i64>,
}

/// `stored-files`: the conversation's files whose bytes the archive holds.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoredFilesAnswer {
    pub files: Vec<StoredFile>,
    /// Its latest capture said files were still downloading; absent when no
    /// session can name the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downloading: Option<bool>,
}

/// A file of a conversation whose bytes the blob store holds.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StoredFile {
    /// The provider's ID of the file.
    pub id: String,
    #[schemars(with = "String")]
    pub blob: polylogue_model::BlobHash,
    pub size: Option<u64>,
}

/// A capture whose attachment names the extension asks the parser to read.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CaptureFiles {
    /// The capture JSON, preserving the submitted bytes
    pub capture: String,
}
