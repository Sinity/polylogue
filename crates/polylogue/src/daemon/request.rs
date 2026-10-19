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
    /// Sessions by latest activity, newest first. `before` continues after
    /// the previous page's last activity time (Unix ms).
    List {
        limit: u64,
        before: Option<i64>,
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
    Candidates {
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
    Measure {
        what: String,
        query: Option<String>,
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
    /// Sessions with live evidence and their live state at `now` (the
    /// daemon's clock when absent).
    Live {
        #[serde(default)]
        now: Option<i64>,
    },
    /// The most recent signal notices and what happened to each.
    Notices {
        #[serde(default)]
        limit: Option<u64>,
    },
    /// The change feed: what changed after `since` (a cursor from a
    /// previous answer's `next`; everything when absent).
    Changes {
        #[serde(default)]
        since: Option<String>,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_round_trip_with_an_op_tag() {
        let r = Request::Query {
            text: "repo:x".into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["op"], "query");
        let back: Request = serde_json::from_value(v).unwrap();
        assert!(matches!(back, Request::Query { .. }));
    }

    #[test]
    fn a_context_request_carries_its_ask() {
        let v = serde_json::json!({
            "op": "context",
            "request_id": "r1",
            "ask": {"ask": "session_start", "cwd": "/x", "source": "resume"},
            "sender": "hook"
        });
        let r: Request = serde_json::from_value(v).unwrap();
        assert!(matches!(
            r,
            Request::Context {
                ask: Ask::SessionStart { .. },
                ..
            }
        ));
    }
}
