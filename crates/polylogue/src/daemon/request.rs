//! One request shape shared by every client: the CLI, MCP, the hook
//! command, the browser host and the web reader. A write carries a request
//! ID so a resent write is a no-op.

use crate::briefing::Ask;
use polylogue_model::{Author, NewNote, SessionId};
use polylogue_store::{DeliveryOutcome, NoteFilter};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Status,
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
    Read {
        session: SessionId,
        view: String,
        format: String,
        logical: bool,
        budget: Option<usize>,
        cursor: Option<String>,
    },
    Import {
        request_id: String,
        paths: Vec<PathBuf>,
    },
    Export {
        sessions: Vec<SessionId>,
        format: String,
    },
    AddNote {
        request_id: String,
        note: NewNote,
    },
    /// Runs a query and keeps the rows it returned, as `result:<id>`.
    Keep {
        request_id: String,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Notes {
        filter: NoteFilter,
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
    /// `session:<id>`, `input:<id>` or `text:<string>`. A dry run counts
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
    /// One harness hook event, as the hook command received it.
    Hook {
        event: String,
        payload: serde_json::Value,
    },
    Cancel {
        request_id: String,
    },
}

impl Request {
    /// Whether the request can run without admitting archive writes.
    /// Conditional delivery and setting writes are resolved from their values.
    pub(crate) fn read_only(&self) -> bool {
        match self {
            Self::Status
            | Self::Query { .. }
            | Self::Read { .. }
            | Self::Export { .. }
            | Self::Notes { .. }
            | Self::Resumable { .. }
            | Self::Continue { .. }
            | Self::Sql { .. } => true,
            Self::Setting { value, .. } => value.is_none(),
            Self::Context { sender, .. } => sender.is_none(),
            Self::Embed { what, .. } => matches!(what.as_str(), "status" | "preview"),
            Self::Import { .. }
            | Self::AddNote { .. }
            | Self::Keep { .. }
            | Self::Withdraw { .. }
            | Self::DeliveryEvent { .. }
            | Self::Excise { .. }
            | Self::Hook { .. }
            | Self::Cancel { .. } => false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum Response {
    Ok {
        data: serde_json::Value,
    },
    /// Accepted; progress and the outcome are separate records.
    Accepted {
        request_id: String,
    },
    Error {
        kind: String,
        message: String,
        /// Structured detail: for a query error, the position, the
        /// candidates and a corrected query that runs as given.
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
            Self::Accepted { request_id } => *request_id = polylogue_render::mask(request_id),
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
