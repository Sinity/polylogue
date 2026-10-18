//! One request shape shared by every client: the CLI, MCP, the hook
//! command, the browser host and the web reader. A write carries a request
//! ID so a resent write is a no-op.

use polylogue_model::{Note, SessionId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Status,
    Query {
        text: String,
        cursor: Option<String>,
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
        note: Note,
    },
    Brief {
        session: Option<SessionId>,
        cwd: Option<PathBuf>,
    },
    Recall {
        text: String,
    },
    Continue {
        session: SessionId,
    },
    Measure {
        what: String,
        query: Option<String>,
    },
    Sql {
        statement: String,
    },
    Excise {
        request_id: String,
        target: String,
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
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["op"], "query");
        let back: Request = serde_json::from_value(v).unwrap();
        assert!(matches!(back, Request::Query { .. }));
    }
}
