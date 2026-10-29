//! One request shape shared by every client: the CLI, MCP, the hook
//! command, the browser host and the web reader. A write carries a request
//! ID so a resent write is a no-op.

mod references;

use crate::briefing::Ask;
use polylogue_model::DeliveryOutcome;
use polylogue_model::{Author, DocumentId, NewNote, SessionId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Selects the first page at a message reference or at the latest message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageSelection {
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
        report: polylogue_model::user::BackfillReport,
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
    SchemaInfer {
        #[serde(flatten)]
        selection: super::schema::Infer,
    },
    SchemaUnread {
        origin: Option<String>,
    },
    Explain {
        subject: String,
        expression: Option<String>,
        unit: Option<String>,
        page_size: Option<u64>,
        cursor: Option<String>,
        session: Option<String>,
    },
    /// Chronological recorded activity and a logical-day calendar.
    Timeline {
        date: Option<String>,
        #[serde(default)]
        across_years: bool,
        #[serde(default)]
        undated: bool,
        cursor: Option<String>,
        page_size: Option<u64>,
    },
    /// Session-scoped usage, optionally bounded by a logical day, with response pages.
    Cost {
        month: Option<String>,
        #[serde(default)]
        undated: bool,
        #[serde(default)]
        unknown_model: bool,
        query: String,
        date: Option<String>,
        model: Option<String>,
        cursor: Option<String>,
        page_size: Option<u64>,
    },
    /// Grammar completions at a UTF-8 byte offset in the full query text.
    Complete {
        text: String,
        position: usize,
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
        #[serde(default)]
        slices: bool,
        view: String,
        format: String,
        #[serde(default)]
        with_parents: bool,
        budget: Option<usize>,
        cursor: Option<String>,
        #[serde(default)]
        selection: Option<PageSelection>,
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
        /// Also the archive's user state: every request.
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
            | Self::SchemaInfer { .. }
            | Self::SchemaUnread { .. }
            | Self::Timeline { .. }
            | Self::Cost { .. }
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
            | Self::DeliveryEvent { .. } => false,
        }
    }

    /// A context has bounded read preparation even when it also records a receipt.
    /// Write authorization still follows read_only, independently of this phase.
    pub(crate) fn has_read_phase(&self) -> bool {
        self.read_only() || matches!(self, Self::Context { .. })
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
#[serde(
    into = "command_surface::Answer<serde_json::Value>",
    from = "command_surface::Answer<serde_json::Value>"
)]
pub enum Response {
    Ok {
        data: serde_json::Value,
        /// References supplied by the response constructor, before transport cleaning.
        #[serde(skip)]
        references: References,
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
    pub fn answer(&self) -> command_surface::Answer<serde_json::Value> {
        match self {
            Self::Ok { data, .. } => crate::protocol::answer(data.clone()),
            Self::Error {
                kind,
                message,
                detail,
            } => command_surface::Answer::Error {
                error: crate::protocol::failure(kind, message.clone(), detail.as_ref()),
                notes: detail
                    .as_ref()
                    .and_then(|d| serde_json::from_value(d["notes"].clone()).ok())
                    .unwrap_or_default(),
                next: detail
                    .as_ref()
                    .and_then(|d| serde_json::from_value(d["next"].clone()).ok())
                    .unwrap_or_default(),
            },
        }
    }

    /// The application response boundary shared by socket and HTTP consumers.
    /// Mask decoded strings before JSON escaping, keeping every structured field.
    pub(crate) fn masked(mut self) -> Self {
        match &mut self {
            Self::Ok { data, references } => references.clean(data, &mut polylogue_render::mask),
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

impl From<Response> for command_surface::Answer<serde_json::Value> {
    fn from(response: Response) -> Self {
        response.answer()
    }
}

impl From<command_surface::Answer<serde_json::Value>> for Response {
    fn from(answer: command_surface::Answer<serde_json::Value>) -> Self {
        use command_surface::Answer;
        match answer {
            Answer::Ok { value, .. }
            | Answer::Empty { value, .. }
            | Answer::Degraded { value, .. } => Self::Ok {
                data: value,
                references: References::default(),
            },
            Answer::Error { error, notes, next } => {
                let detail = serde_json::json!({
                    "position": error.position,
                    "suggestion": error.suggestion,
                    "hint": error.hint,
                    "retryable": error.retryable,
                    "notes": notes,
                    "next": next,
                });
                Self::Error {
                    kind: error.code.unwrap_or_else(|| {
                        serde_json::to_value(error.kind)
                            .expect("failure kind serializes")
                            .as_str()
                            .expect("kind is text")
                            .to_owned()
                    }),
                    message: error.message,
                    detail: Some(detail),
                }
            }
        }
    }
}

/// Typed identifiers and declared command continuations excluded from text cleaning.
#[derive(Debug, Clone, Default)]
pub struct References(Vec<String>);

impl References {
    pub(crate) fn response(mut self, data: impl Serialize) -> Response {
        match references::value(&mut self, "", &data) {
            Ok(data) => {
                // These root fields belong to the command response, not its authored values.
                for field in ["cursor", "next", "previous"] {
                    self.opaque_field(&data, format!("/{field}"));
                }
                Response::Ok {
                    data,
                    references: self,
                }
            }
            Err(error) => crate::daemon::service::error("internal", error),
        }
    }
    pub(crate) fn value(&mut self, path: &str, data: &impl Serialize) -> serde_json::Value {
        references::value(self, path, data).expect("model serializes")
    }

    pub(crate) fn observe(&mut self, path: &str, data: &impl Serialize) {
        references::observe(self, path, data);
    }

    pub(crate) fn add(&mut self, path: impl Into<String>, _value: &impl Serialize) {
        self.0.push(path.into());
    }

    pub(crate) fn opaque_field(&mut self, value: &serde_json::Value, path: impl Into<String>) {
        let path = path.into();
        if let Some(value) = value.pointer(&path).and_then(serde_json::Value::as_str) {
            self.add(path, &value);
        }
    }

    pub(crate) fn field<T: serde::de::DeserializeOwned + Serialize>(
        &mut self,
        value: &serde_json::Value,
        path: impl Into<String>,
    ) {
        let path = path.into();
        if let Some(value) = value
            .pointer(&path)
            .and_then(|v| serde_json::from_value::<T>(v.clone()).ok())
        {
            self.value(&path, &value);
        }
    }

    fn clean(&mut self, value: &mut serde_json::Value, string: &mut dyn FnMut(&str) -> String) {
        polylogue_model::masking::mask_json_values_except(value, string, &mut self.0);
    }
}

impl Response {
    pub(crate) fn clean(mut self, mut string: impl FnMut(&str) -> String) -> Self {
        match &mut self {
            Self::Ok { data, references } => references.clean(data, &mut string),
            Self::Error {
                message, detail, ..
            } => {
                *message = string(message);
                if let Some(detail) = detail {
                    polylogue_model::masking::mask_json_values(detail, string);
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests;
