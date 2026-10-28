//! Codex SQLite databases, parsed from their stored row-dump records.
//! Lines are a schema line, row lines `{t, k, r}` and gone lines
//! `{t, k, gone}`.
//!
//! What becomes rows: thread names, kinds and settings (`threads`), spawn
//! edges (`thread_spawn_edges`), goals (`thread_goals`) and memories
//! (`stage1_outputs`). The thread-history tables are derived from the
//! rollouts, which are parsed; they stay in the stored export.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use super::{items::str_field, session_id};
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

/// The database a file is, or the database its export is of.
fn database(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    let db = name.strip_suffix(".jsonl").unwrap_or(name);
    let stem = db.strip_suffix(".sqlite")?;
    DATABASES
        .iter()
        .any(|p| stem.starts_with(p) && stem[p.len()..].bytes().all(|b| b.is_ascii_digit()))
        .then_some(db)
}

/// Recognizes an export file; the database itself is never parsed.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension().is_some_and(|e| e == "jsonl")
        && database(path).is_some()
        && head
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_none_or(|&b| b == b'{')
}

/// Whether `path` is a database this origin exports.
pub fn is_database(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "sqlite") && database(path).is_some()
}

pub fn is_row_dump(path: &Path) -> bool {
    is_database(path) || recognize(path, b"{")
}

pub fn open(path: &Path) -> Option<Parse> {
    is_row_dump(path).then(|| Parse::Lines(Box::new(Database)))
}

pub struct Database;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoState {}

fn row_session(row: &Json, field: &str) -> Option<polylogue_model::SessionId> {
    str_field(row, field).and_then(session_id)
}

impl Parser for Database {
    type ParserState = NoState;

    fn kind(&self, v: &Json) -> String {
        if v.get("schema").is_some() {
            return "schema".into();
        }
        let t = str_field(v, "t").unwrap_or("?");
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
        _: &mut Records,
    ) -> RecordOutcome {
        if v.get("schema").is_some() {
            return RecordOutcome::Used;
        }
        let Some(table) = str_field(v, "t") else {
            return RecordOutcome::unreadable("export line without a table");
        };
        if v.get("gone").is_some() {
            return RecordOutcome::skipped(
                "row deleted in the tool; the export keeps its last value",
            );
        }
        let row = v.get("r").unwrap_or(&Json::Null);
        match table {
            "threads" => {
                let Some(id) = row_session(row, "id") else {
                    return RecordOutcome::unreadable("thread row without an ID");
                };
                let kind = if str_field(row, "thread_source") == Some("subagent")
                    || str_field(row, "source").is_some_and(|s| s.contains("\"subagent\""))
                {
                    SessionKind::Subagent
                } else {
                    SessionKind::Main
                };
                let mut s = Session::new(id, kind);
                if let Some(name) = str_field(row, "name") {
                    s.offer_title(name, TitleSource::Generated);
                }
                // Codex fills `title` with the whole first prompt until a
                // thread is named; that copy is not a title, and the derived
                // title is its first line.
                if let Some(title) = str_field(row, "title")
                    .filter(|title| Some(*title) != str_field(row, "first_user_message"))
                {
                    s.offer_title(title, TitleSource::Summary);
                }
                if let Some(nick) = str_field(row, "agent_nickname") {
                    s.offer_title(nick, TitleSource::AgentName);
                }
                if let Some(cwd) = str_field(row, "cwd") {
                    s.offer_cwd(cwd);
                }
                if let Some(b) = str_field(row, "git_branch") {
                    s.git_branch = Some(b.to_owned());
                }
                if let Some(v) = str_field(row, "cli_version") {
                    s.tool_version = Some(v.to_owned());
                }
                if let Some(at) = row
                    .get("created_at_ms")
                    .and_then(time::epoch_millis)
                    .or_else(|| row.get("created_at").and_then(time::epoch_seconds))
                {
                    s.offer_start(at);
                }
                out.sessions.push(s);
                RecordOutcome::Used
            }
            "thread_spawn_edges" => {
                let (Some(parent), Some(child)) = (
                    row_session(row, "parent_thread_id"),
                    row_session(row, "child_thread_id"),
                ) else {
                    return RecordOutcome::unreadable("spawn edge without both threads");
                };
                out.links.push(Link {
                    kind: LinkKind::Spawn,
                    parent,
                    child,
                    at: None,
                    call_id: None,
                    evidence: "thread_spawn_edges".into(),
                });
                RecordOutcome::Used
            }
            "thread_goals" | "stage1_outputs" => {
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
                out.events.push(Event {
                    session: Some(session),
                    kind: kind.into(),
                    at,
                    after: None,
                    range: Some((record.offset, record.offset + record.bytes.len() as u64)),
                    data: row.clone(),
                });
                RecordOutcome::Used
            }
            "thread_items"
            | "thread_turns"
            | "thread_realtime_items"
            | "thread_history_projection_state" => RecordOutcome::skipped(
                "derived from the rollout, which is parsed; kept in the stored export",
            ),
            _ => RecordOutcome::skipped("kept in the stored export; not indexed"),
        }
    }

    fn flush(&self, _: &mut NoState, _: &mut Output) {}
}

#[cfg(test)]
mod tests;
