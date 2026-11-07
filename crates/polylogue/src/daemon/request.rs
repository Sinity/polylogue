//! One request shape shared by every client: the CLI, MCP, the hook
//! command, the browser host and the web reader. A write carries a request
//! ID so a resent write is a no-op.

use crate::briefing::Ask;
use polylogue_model::DeliveryOutcome;
use polylogue_model::SessionId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    /// The on-demand check of stored bytes, with an optional manifest file.
    Verify {
        manifest: Option<PathBuf>,
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
        #[serde(default = "attachment_chunk")]
        length: usize,
        #[serde(default)]
        cursor: Option<String>,
    },
    /// Finite change-reference batch using shared SSE framing.
    Changes {
        since: Option<String>,
    },
    /// Lines for a session or a sessions query as its live state changes
    /// and its rows arrive, until the deadline; `marks` continues a
    /// previous follow.
    Follow {
        query: String,
        #[serde(default)]
        marks: Option<crate::protocol::follow::Marks>,
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
    /// Read one canonical reference or a query through the shared selection.
    Read {
        target: String,
        #[serde(default)]
        options: polylogue_model::read::ReadOptions,
        #[serde(default)]
        delivery: polylogue_model::read::ReadDelivery,
        format: String,
        budget: Option<usize>,
        #[serde(default)]
        with_blobs: bool,
        /// A page cursor issued by an earlier page of this same reading.
        #[serde(default)]
        cursor: Option<String>,
    },
    Import {
        request_id: String,
        paths: Vec<PathBuf>,
    },
    /// An archive in the older format, imported once; with `dry_run` only
    /// read and checked.
    ImportArchive {
        request_id: String,
        path: PathBuf,
        #[serde(default)]
        dry_run: bool,
    },
    /// Validate and record an extension selection in one pinned operation.
    BrowserSelectionNote {
        request_id: String,
        body: String,
        quote: String,
        provider: String,
        conversation_id: String,
        message_id: String,
        page_url: String,
        selected_at: String,
    },
    Assert {
        request_id: String,
        assertions: Vec<polylogue_model::assertion::NewAssertion>,
    },
    AssertQuery {
        request_id: String,
        query: String,
        assertion: polylogue_model::assertion::NewAssertion,
    },
    Revise {
        request_id: String,
        #[serde(with = "polylogue_model::assertion_revisions")]
        bases: Vec<polylogue_model::RevisionReference>,
        patch: polylogue_model::assertion::AssertionPatch,
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
        #[serde(default)]
        denominator: Option<String>,
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
        targets: Vec<polylogue_model::Reference>,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Internal plan for the canonical brief, takeover and recall operations.
    /// With a `sender`
    /// (hook, mcp, ...) its record as a delivery to the recipient is queued
    /// for the writer and the answer does not wait for it; the request ID
    /// becomes the delivery ID.
    #[serde(skip)]
    Context {
        request_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        budget_tokens: Option<u32>,
        ask: Ask,
        #[serde(default)]
        sender: Option<String>,
    },
    /// An addressed message to a session, or to every session a quoted
    /// query (`@[sessions …]`) selects when it is sent.
    Ping {
        request_id: String,
        target: String,
        text: String,
        /// Refuse rather than resume a session that is not running.
        #[serde(default)]
        live: bool,
        #[serde(default)]
        about: Option<String>,
    },
    /// A hook takes the pings and notices pending for its session.
    Claim {
        session: SessionId,
        event: String,
    },
    /// Approve or deny a session's pending request; `standing` answers the
    /// same tool and input whenever the session asks next.
    Answer {
        request_id: String,
        session: SessionId,
        approve: bool,
        #[serde(default)]
        text: Option<String>,
        #[serde(default)]
        standing: bool,
    },
    /// The synchronous permission hook asks for a decision given ahead.
    Permission {
        session: SessionId,
        tool: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    /// Stop a session's current turn.
    Interrupt {
        request_id: String,
        session: SessionId,
    },
    /// Start a session Polylogue hosts, in `cwd`, with `text` as its first
    /// turn: on `harness`, or as an instance of `role` (its harness and
    /// instructions).
    Start {
        request_id: String,
        #[serde(default)]
        harness: Option<String>,
        #[serde(default)]
        role: Option<String>,
        cwd: PathBuf,
        text: String,
    },
    /// Hand a session's work to a new hosted session on `to`, briefed with
    /// its takeover pack (from message `at`), in `cwd` or the source's; `stop_source` interrupts the
    /// source afterwards; `print` shows the pack and starts nothing.
    HandOff {
        request_id: String,
        session: SessionId,
        to: String,
        #[serde(default)]
        cwd: Option<PathBuf>,
        #[serde(default)]
        at: Option<String>,
        #[serde(default)]
        stop_source: bool,
        #[serde(default)]
        print: bool,
    },
    /// What happened to a delivery after it was handed out.
    DeliveryEvent {
        request_id: String,
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
        key: Option<String>,
        #[serde(default)]
        value: Option<String>,
        /// Explicit host identity for host-scoped settings.
        #[serde(default)]
        machine: Option<String>,
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
    pub(crate) fn streaming(&self) -> bool {
        matches!(
            self,
            Self::Changes { .. }
                | Self::Follow { .. }
                | Self::Read {
                    delivery: polylogue_model::read::ReadDelivery::Stream,
                    ..
                }
        )
    }

    /// Whether the request can run without writing to the archive.
    /// Conditional delivery and setting writes are resolved from their values.
    pub(crate) fn read_only(&self) -> bool {
        match self {
            Self::Status
            | Self::SchemaInfer { .. }
            | Self::SchemaUnread { .. }
            | Self::Verify { .. }
            | Self::Timeline { .. }
            | Self::Cost { .. }
            | Self::Complete { .. }
            | Self::Explain { .. }
            | Self::Attachment { .. }
            | Self::Query { .. }
            | Self::Changes { .. }
            | Self::Follow { .. }
            | Self::Read { .. }
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
            | Self::ImportArchive { .. }
            | Self::BrowserSelectionNote { .. }
            | Self::Assert { .. }
            | Self::AssertQuery { .. }
            | Self::Revise { .. }
            | Self::Judge { .. }
            | Self::Keep { .. }
            | Self::Save { .. }
            | Self::Watch { .. }
            | Self::Withdraw { .. }
            | Self::DeliveryEvent { .. }
            | Self::Ping { .. }
            | Self::Claim { .. }
            | Self::Answer { .. }
            | Self::Permission { .. }
            | Self::Interrupt { .. }
            | Self::Start { .. }
            | Self::HandOff { .. } => false,
        }
    }

    /// A context has bounded read preparation even when it also records a receipt.
    /// Write authorization still follows read_only, independently of this phase.
    pub(crate) fn has_read_phase(&self) -> bool {
        self.read_only()
            || matches!(
                self,
                Self::Context { .. } | Self::AssertQuery { .. } | Self::BrowserSelectionNote { .. }
            )
    }
}

fn attachment_chunk() -> usize {
    64 * 1024
}

#[cfg(test)]
mod tests;
