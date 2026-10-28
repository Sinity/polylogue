//! One request shape shared by every client: the CLI, MCP, the hook
//! command, the browser host and the web reader. A write carries a request
//! ID so a resent write is a no-op.

use crate::briefing::Ask;
use polylogue_model::DeliveryOutcome;
use polylogue_model::{Author, DocumentId, NewNote, SessionId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Selects the first page at a message reference or at the latest message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadStart {
    Message(polylogue_model::MessageId),
    Latest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Queues a chat action with the files it attaches; only the operator
    /// does. The request ID is the action's idempotency key.
    ChatSend {
        request_id: String,
        action: crate::browser::actions::ChatRequest,
        #[serde(default)]
        files: Vec<crate::browser::actions::ChatFile>,
    },
    ChatActions,
    ChatAction {
        id: String,
    },
    /// The extension instance `by` takes the oldest queued chat action,
    /// while chat actions are on.
    ChatClaim {
        by: String,
    },
    ChatReport {
        id: String,
        update: crate::browser::actions::Update,
    },
    ChatDecide {
        id: String,
        approve: bool,
        by: String,
        #[serde(default)]
        detail: Option<String>,
    },
    /// A chat action attachment's bytes, base64.
    ChatAttachment {
        id: String,
        attachment: String,
    },
    /// Starts capturing a web chat's older conversations; the provider's
    /// job that has not ended is answered instead of a second one.
    BackfillStart {
        provider: String,
        #[serde(default)]
        since: Option<String>,
    },
    BackfillJobs,
    BackfillCancel {
        id: String,
    },
    /// The extension instance `by` takes the backfill job it holds or the
    /// oldest waiting one.
    BackfillClaim {
        by: String,
    },
    BackfillReport {
        id: String,
        by: String,
        report: polylogue_store::backfill_jobs::BackfillReport,
    },
    /// What the archive holds of some of a web chat's conversations: the
    /// session, its message count and when it was last captured.
    Conversations {
        provider: String,
        ids: Vec<String>,
    },
    /// The files of a web chat's conversation whose bytes the archive
    /// holds, so the extension does not download them again.
    StoredFiles {
        provider: String,
        conversation: String,
    },
    Status,
    Explain {
        subject: String,
        expression: Option<String>,
        unit: Option<String>,
        page_size: Option<u64>,
        cursor: Option<String>,
        session: Option<String>,
    },
    /// Completions for one word of a query being typed.
    Complete {
        word: String,
    },
    Attachment {
        message: polylogue_model::MessageId,
        block: usize,
        #[serde(default)]
        attachment: usize,
        #[serde(default)]
        input: Option<String>,
        #[serde(default)]
        reveal: bool,
        #[serde(default = "attachment_chunk")]
        length: usize,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// Finite JSON-lines change-reference batch over the existing socket.
    Changes {
        since: Option<String>,
    },
    /// A query in the query language. `cursor` continues a previous page;
    /// `page_size` overrides the unit's default page; `why` explains an
    /// empty result clause by clause.
    Query {
        text: String,
        cursor: Option<String>,
        #[serde(default)]
        page_size: Option<u64>,
        #[serde(default)]
        why: bool,
        /// The calling session, which `self` names.
        #[serde(default)]
        session: Option<String>,
        /// The columns to show instead of the unit's defaults.
        #[serde(default)]
        fields: Vec<String>,
    },
    /// Reads one session, document or event in a view and format.
    /// `with_parents` (sessions only): the parent sessions' messages first.
    Read {
        target: ReadTarget,
        view: String,
        format: String,
        #[serde(default)]
        with_parents: bool,
        budget: Option<usize>,
        cursor: Option<String>,
        #[serde(default)]
        start: Option<ReadStart>,
    },
    /// One exact note revision, including a superseded or withdrawn revision.
    Note {
        id: String,
    },
    Outline {
        session: SessionId,
        #[serde(default)]
        with_parents: bool,
        page_size: usize,
        cursor: Option<String>,
    },
    Import {
        request_id: String,
        paths: Vec<PathBuf>,
    },
    Export {
        targets: Vec<ReadTarget>,
        format: String,
        /// Also the archive's user state: every request and excision.
        #[serde(default)]
        user_state: bool,
    },
    AddNote {
        request_id: String,
        note: NewNote,
    },
    /// A note on every row a query returns (sessions, messages, documents
    /// or notes); `note.targets` is filled from the rows.
    AnnotateQuery {
        request_id: String,
        query: String,
        note: NewNote,
    },
    /// The operator accepts or rejects one revision of an agent's note.
    Judge {
        request_id: String,
        note: String,
        accepted: bool,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Saves a query under a name, to run as `query:NAME`; saving a name
    /// again revises its definition.
    Save {
        request_id: String,
        name: String,
        text: String,
        author: Author,
    },
    /// Watches a query: each row that starts matching becomes a notice
    /// (a signal rule delivering notices).
    Watch {
        request_id: String,
        query: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// How long a row must keep matching first, such as `20m`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        hold: Option<String>,
        /// Also show each notice as a desktop notification.
        #[serde(default)]
        notify: bool,
        author: Author,
    },
    /// Runs a query and keeps the rows it returned, as `result:<id>`.
    Keep {
        request_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Withdraw {
        request_id: String,
        note: String,
        author: Author,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Prepares context: a brief, a takeover or a recall. With a `sender`
    /// (hook, mcp, ...) it is recorded as a delivery to the recipient
    /// before it is returned; the request ID becomes the delivery ID.
    Context {
        request_id: String,
        ask: Ask,
        #[serde(default)]
        sender: Option<String>,
    },
    /// Advice printed by a hook. Preparation is read-only.
    AdviceDelivered {
        request_id: String,
        session: Option<SessionId>,
        cwd: PathBuf,
        text: String,
    },
    /// What happened to a delivery after it was handed out.
    DeliveryEvent {
        delivery: String,
        what: DeliveryOutcome,
        #[serde(default)]
        detail: Option<String>,
    },
    /// Sessions in a directory worth resuming, with what each left open.
    Resumable {
        cwd: PathBuf,
        #[serde(default)]
        limit: Option<u64>,
    },
    Continue {
        session: SessionId,
    },
    /// Reads (`value: None`) or sets a setting.
    Setting {
        #[serde(default)]
        request_id: Option<String>,
        key: String,
        #[serde(default)]
        value: Option<String>,
        /// Per machine instead of archive-wide.
        #[serde(default)]
        machine: bool,
    },
    Sql {
        statement: String,
    },
    /// `session:<id>`, `document:<id>` or `input:<id>`. A dry run counts
    /// what would go and changes nothing.
    Excise {
        request_id: String,
        target: String,
        #[serde(default)]
        reason: Option<String>,
        #[serde(default)]
        dry_run: bool,
    },
    /// Embeddings: `status`, `preview` (what a build would embed and cost)
    /// or `build`. `query` limits them to the sessions it finds; `limit`
    /// to that many texts. A paid provider builds only with `yes`.
    Embed {
        what: String,
        #[serde(default)]
        query: Option<String>,
        #[serde(default)]
        limit: Option<u64>,
        #[serde(default)]
        yes: bool,
    },
}

impl Request {
    /// Whether the request can run without writing to the archive.
    /// Conditional delivery and setting writes are resolved from their values.
    pub(crate) fn read_only(&self) -> bool {
        match self {
            Self::Status
            | Self::Complete { .. }
            | Self::Explain { .. }
            | Self::Attachment { .. }
            | Self::Query { .. }
            | Self::Changes { .. }
            | Self::Read { .. }
            | Self::Outline { .. }
            | Self::Note { .. }
            | Self::Export { .. }
            | Self::Resumable { .. }
            | Self::Continue { .. }
            | Self::Sql { .. }
            | Self::ChatActions
            | Self::ChatAction { .. }
            | Self::ChatAttachment { .. }
            | Self::BackfillJobs
            | Self::Conversations { .. }
            | Self::StoredFiles { .. } => true,
            Self::Setting { value, .. } => value.is_none(),
            Self::Context { sender, .. } => sender.is_none(),
            Self::Embed { what, .. } => matches!(what.as_str(), "status" | "preview"),
            Self::ChatSend { .. }
            | Self::ChatClaim { .. }
            | Self::BackfillStart { .. }
            | Self::BackfillCancel { .. }
            | Self::BackfillClaim { .. }
            | Self::BackfillReport { .. }
            | Self::ChatReport { .. }
            | Self::ChatDecide { .. }
            | Self::Import { .. }
            | Self::AddNote { .. }
            | Self::AnnotateQuery { .. }
            | Self::Judge { .. }
            | Self::Keep { .. }
            | Self::Save { .. }
            | Self::Watch { .. }
            | Self::Withdraw { .. }
            | Self::AdviceDelivered { .. }
            | Self::DeliveryEvent { .. }
            | Self::Excise { .. } => false,
        }
    }
}

/// What `read` and `export` name: `document:ID`, `event:ID`, or a session
/// ID (with or without `session:`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum ReadTarget {
    Session(SessionId),
    Document(DocumentId),
    Event(String),
}

impl ReadTarget {
    pub fn parse(text: &str) -> Result<Self, String> {
        if let Some(id) = text.strip_prefix("event:") {
            if id.is_empty() {
                return Err("event reference requires an ID".into());
            }
            return Ok(Self::Event(id.to_owned()));
        }
        if let Some(id) = text.strip_prefix("document:") {
            return DocumentId::parse(id)
                .map(Self::Document)
                .map_err(|e| e.to_string());
        }
        SessionId::parse(text.strip_prefix("session:").unwrap_or(text))
            .map(Self::Session)
            .map_err(|e| e.to_string())
    }
}

fn attachment_chunk() -> usize {
    64 * 1024
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum Response {
    Ok {
        data: serde_json::Value,
    },
    Error {
        kind: String,
        message: String,
        /// Structured detail: for a query error, the position, similar
        /// field names and a corrected query that runs as given.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<serde_json::Value>,
    },
}

impl Response {
    /// The application response boundary shared by socket and HTTP consumers.
    /// Mask decoded strings before JSON escaping, keeping every structured field.
    pub(crate) fn masked(mut self) -> Self {
        match &mut self {
            Self::Ok { data } => polylogue_render::mask_json(data),
            Self::Error {
                kind,
                message,
                detail,
            } => {
                *kind = polylogue_render::mask(kind);
                *message = polylogue_render::mask(message);
                if let Some(detail) = detail {
                    polylogue_render::mask_json(detail);
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests;
