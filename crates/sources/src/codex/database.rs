//! Codex SQLite databases, parsed from their stored row-dump records.
//! Lines are a schema line, row lines `{t, k, r}` and gone lines
//! `{t, k, gone}`.
//!
//! What becomes rows: thread names, kinds and settings (`threads`), spawn
//! edges (`thread_spawn_edges`), goals (`thread_goals`) and memories
//! (`stage1_outputs`). The thread-history tables are derived from the
//! rollouts, which are parsed; they stay in the stored row dump, as do the
//! rows of the `queue_*` databases.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 4;

use super::session_id;
use crate::lines::Record;
use crate::origin::Root;
use crate::parse::{Output, Parse, Parser};
use crate::time;
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json, Link, LinkKind, Session, SessionKind, TitleSource};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ROOT: Root = super::ROOT;

/// The databases captured, by file name prefix (`state_5.sqlite`). The
/// Diagnostic logs and thread-history projections are outside this declaration.
const DATABASES: &[&str] = &["state_", "goals_", "memories_", "queue_"];

pub const LOCATIONS: &[&str] = &[
    "state_*.sqlite",
    "goals_*.sqlite",
    "memories_*.sqlite",
    "queue_*.sqlite",
];

/// The database a file is, or the database its row dump is of.
fn database(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    let db = name.strip_suffix(".jsonl").unwrap_or(name);
    let stem = db.strip_suffix(".sqlite")?;
    DATABASES
        .iter()
        .any(|p| stem.starts_with(p) && stem[p.len()..].bytes().all(|b| b.is_ascii_digit()))
        .then_some(db)
}

/// Recognizes a row dump file; the database itself is never parsed.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    let head = head.bytes;
    path.extension().is_some_and(|e| e == "jsonl")
        && database(path).is_some()
        && crate::document::starts_object_lines(head)
}

/// Whether `path` is a database this origin exports.
pub fn is_database(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "sqlite") && database(path).is_some()
}

pub fn is_row_dump(path: &Path) -> bool {
    is_database(path) || recognize(path, &crate::document::Head::new(b"{"))
}

pub fn open(path: &Path) -> Option<Parse> {
    is_row_dump(path).then(|| Parse::Lines(Box::new(Database)))
}

pub struct Database;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

fn row_session(row: &Json, field: &str) -> Option<polylogue_model::SessionId> {
    row.str(field)
        .filter(|s| !s.is_empty())
        .and_then(session_id)
}

static THREAD_FIELDS: crate::fields::Fields = crate::fields::Fields::new(
    "thread",
    &[
        (
            "id",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "name",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "title",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "first_user_message",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "agent_nickname",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "cwd",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "git_branch",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "cli_version",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "thread_source",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "source",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "created_at_ms",
            crate::fields::Rule::Read(crate::fields::ExpectedType::Number),
        ),
        (
            "created_at",
            crate::fields::Rule::Read(crate::fields::ExpectedType::Number),
        ),
    ],
);
static EDGE_FIELDS: crate::fields::Fields = crate::fields::Fields::new(
    "spawn_edge",
    &[
        (
            "parent_thread_id",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
        (
            "child_thread_id",
            crate::fields::Rule::Read(crate::fields::ExpectedType::String),
        ),
    ],
);
static EVENT_FIELDS: crate::fields::Fields = crate::fields::Fields::new(
    "thread_event",
    &[(
        "",
        crate::fields::Rule::Kept("complete captured row in the event data"),
    )],
);

impl Parser for Database {
    type ParserState = NoState;

    fn kind(&self, v: &Json) -> String {
        if v.get("schema").is_some() {
            return "schema".into();
        }
        let t = v.str("t").filter(|s| !s.is_empty()).unwrap_or("?");
        if v.get("gone").is_some() {
            format!("{t}/gone")
        } else {
            t.to_owned()
        }
    }

    fn parse_record(
        &self,
        _: &mut NoState,
        record: Record<'_>,
        v: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        if v.get("schema").is_some() {
            return RecordOutcome::Used;
        }
        let Some(table) = v.str("t").filter(|s| !s.is_empty()) else {
            return RecordOutcome::unreadable("row dump line without a table");
        };
        if v.get("gone").is_some() {
            return RecordOutcome::skipped(
                "row deleted in the tool; the row dump keeps its last value",
            );
        }
        let row = v.get("r").unwrap_or(&Json::Null);
        records.fields(
            table,
            &crate::fields::header(v, &["r"]),
            &crate::row_dump::ROW_ENVELOPE,
        );
        match table {
            "threads" => {
                records.fields("threads", row, &THREAD_FIELDS);
                let Some(id) = row_session(row, "id") else {
                    return RecordOutcome::unreadable("thread row without an ID");
                };
                let kind = if row.str("thread_source").filter(|s| !s.is_empty()) == Some("subagent")
                    || row
                        .str("source")
                        .filter(|s| !s.is_empty())
                        .is_some_and(|s| s.contains("\"subagent\""))
                {
                    SessionKind::Subagent
                } else {
                    SessionKind::Main
                };
                let mut s = Session::new(id, kind);
                if let Some(name) = row.str("name").filter(|s| !s.is_empty()) {
                    s.offer_title(name, TitleSource::Generated);
                }
                // Codex fills `title` with the whole first prompt until a
                // thread is named; that copy is not a title, and the derived
                // title is its first line.
                if let Some(title) = row.str("title").filter(|s| !s.is_empty()).filter(|title| {
                    Some(*title) != row.str("first_user_message").filter(|s| !s.is_empty())
                }) {
                    s.offer_title(title, TitleSource::Summary);
                }
                if let Some(nick) = row.str("agent_nickname").filter(|s| !s.is_empty()) {
                    s.offer_title(nick, TitleSource::AgentName);
                }
                if let Some(cwd) = row.str("cwd").filter(|s| !s.is_empty()) {
                    s.offer_cwd(cwd);
                }
                if let Some(b) = row.str("git_branch").filter(|s| !s.is_empty()) {
                    s.git_branch = Some(b.to_owned());
                }
                if let Some(v) = row.str("cli_version").filter(|s| !s.is_empty()) {
                    s.tool_version = Some(v.to_owned());
                }
                if let Some(at) = row
                    .get("created_at_ms")
                    .and_then(time::epoch_millis)
                    .or_else(|| row.get("created_at").and_then(time::epoch_seconds))
                {
                    s.offer_start(at.timestamp(), at.accuracy())
                        .expect("decoded source time is valid");
                }
                out.sessions.push(s);
                RecordOutcome::Used
            }
            "thread_spawn_edges" => {
                records.fields("thread_spawn_edges", row, &EDGE_FIELDS);
                let (Some(parent), Some(child)) = (
                    row_session(row, "parent_thread_id"),
                    row_session(row, "child_thread_id"),
                ) else {
                    return RecordOutcome::unreadable("spawn edge without both threads");
                };
                out.links.push(Link {
                    kind: LinkKind::Spawn,
                    copies_prefix: false,
                    how: None,
                    parent,
                    child,
                    branch_point: None,
                    call_id: None,
                    evidence: "thread_spawn_edges".into(),
                });
                RecordOutcome::Used
            }
            "thread_goals" | "stage1_outputs" => {
                records.fields(table, row, &EVENT_FIELDS);
                let Some(session) = row_session(row, "thread_id") else {
                    return RecordOutcome::unreadable("row without a thread ID");
                };
                let (kind, at) = if table == "thread_goals" {
                    (
                        "goal",
                        row.get("updated_at_ms").and_then(time::epoch_millis),
                    )
                } else {
                    (
                        "memory",
                        row.get("generated_at").and_then(time::epoch_seconds),
                    )
                };
                out.events.push({
                    let (at, time_accuracy) = crate::time::parts(at);
                    Event {
                        session: Some(session),
                        kind: kind.into(),
                        at,
                        at_quality: time_accuracy.quality,
                        at_precision: time_accuracy.precision,
                        at_until: time_accuracy.until,
                        after: None,
                        range: Some((record.offset, record.offset + record.bytes.len() as u64)),
                        data: row.clone(),
                    }
                });
                RecordOutcome::Used
            }
            "thread_items"
            | "thread_turns"
            | "thread_realtime_items"
            | "thread_history_projection_state" => RecordOutcome::skipped(
                "derived from the rollout, which is parsed; kept in the stored row dump",
            ),
            _ => {
                records.unknown_table(table, table, row);
                RecordOutcome::skipped("kept in the stored row dump; not indexed")
            }
        }
    }

    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

#[cfg(test)]
mod tests;
