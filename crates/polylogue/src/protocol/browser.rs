//! Chat actions and import jobs the browser extension runs, and what the
//! extension asks the archive about the conversations it sees.
use crate::browser::actions::{ActionSummary, ChatFile, Update};
use crate::browser::import_job::JobSummary;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Queues a chat action with the files it attaches; only the operator does.
/// The request ID is the action's idempotency key.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatSend {
    /// The web chat: chatgpt
    #[schemars(extend("argument_form" = "positional"))]
    pub provider: String,
    /// What to write
    #[schemars(extend("argument_form" = "words"))]
    pub text: String,
    /// The model, as the provider's model menu names it
    pub model: String,
    /// The reasoning effort, as the provider's menu names it
    #[schemars(extend("unlisted" = true))]
    pub effort: Option<String>,
    /// Reply in this conversation (its ID); without it a new conversation starts
    pub conversation: Option<String>,
    /// Start the conversation in this project
    #[schemars(extend("unlisted" = true))]
    pub project: Option<String>,
    /// Send the turn instead of leaving a draft
    #[serde(default)]
    pub submit: bool,
    /// The attached files with their bytes; the command line reads `--attach`
    #[serde(default)]
    #[schemars(extend("hidden" = true, "unlisted" = true))]
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
    /// The chat action
    #[schemars(extend("argument_form" = "positional"))]
    pub id: String,
}

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

/// A chat action's decision as the extension's page reports it.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatDecision {
    /// The chat action
    pub id: String,
    /// Approve it; without this it is declined
    pub approve: bool,
    /// Who decided, as the extension reports it
    pub by: String,
    /// Why, in the decider's words
    pub detail: Option<String>,
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

/// Starts capturing a web chat's older conversations; the provider's job
/// that has not ended is answered instead of a second one.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobStart {
    /// The web chat: chatgpt, claude-ai or grok
    #[schemars(extend("argument_form" = "positional"))]
    pub provider: String,
    /// Only conversations changed on or after this date or time
    pub since: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobList {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobCancel {
    /// The job
    #[schemars(extend("argument_form" = "positional"))]
    pub id: String,
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

/// A blob is stored in the local primary-mode spool before a chat action exists.
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
    pub action: Option<Box<ActionSummary>>,
}

/// Confirmed state, a saved write result, or files held in the spool before enqueue.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ChatAnswer {
    Current(ChatCurrent),
    Outcome(polylogue_model::user::UserOutcome),
    Capture(PendingCapture),
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

/// `chat list`: every chat action and where it stands.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChatActionList {
    pub actions: Vec<ActionSummary>,
}

/// `backfill list`: every import job and how far it got.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportJobs {
    pub jobs: Vec<JobSummary>,
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
