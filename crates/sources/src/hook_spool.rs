//! The hook spool: one JSON line per harness hook event (session start,
//! prompt submitted, tool about to run, tool finished, compaction, stop),
//! appended without a sync. Two layouts share the format: a file
//! per session (`<provider>-<session>.jsonl`) and dated files
//! (`carriers/<provider>/<date>/*.ndjson`, each holding the hook lines one
//! process wrote). The `polylogue hook` command writes a
//! third: one file per harness process (`<pid>.ndjson`), lines
//! `{"at_ns", "event", "harness_pid", "payload"}`, for Claude Code.
//!
//! Each line becomes an event on the session it names, kept whole: what
//! the hook saw (tool input, permission mode, effort, working directory)
//! is evidence the transcript does not always hold. The session ID is
//! the harness's own, so the events land on the same session as its
//! transcript.

use crate::document::{first_key, session_id, str_of};
use crate::fields::{Fields, Rule};
use crate::lines::Record;
use crate::parse::{Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json, Session, SessionKind};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ORIGIN: &str = "hook-spool";

pub const LOCATIONS: &[&str] = &[
    "polylogue/hooks/*.jsonl",
    "polylogue/hooks/carriers/*/*/*.ndjson",
    "polylogue/hooks/*.ndjson",
];

fn is_spool_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e == "jsonl" || e == "ndjson")
}

/// A spool line starts `{"event_type"` (per-session files), `{"event_id"`
/// (dated files) or `{"at_ns"` (the hook command's files).
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    is_spool_file(path) && matches!(first_key(head), Some("event_type" | "event_id" | "at_ns"))
}

pub fn open(path: &Path) -> Option<Parse> {
    is_spool_file(path).then(|| Parse::Lines(Box::new(HookSpool)))
}

pub struct HookSpool;

/// Session facts already sent, so each step resends them once.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {}

static FIELDS: Fields = Fields::new(&[
    ("event_id", Rule::Read),
    ("event_type", Rule::Read),
    ("event", Rule::Read),
    ("at_ns", Rule::Read),
    (
        "harness_pid",
        Rule::Kept("the harness process, which the file name also carries"),
    ),
    ("observed_at_ms", Rule::Read),
    ("timestamp", Rule::Read),
    ("provider", Rule::Read),
    ("session_id", Rule::Read),
    (
        "payload",
        Rule::Kept("the hook's input, kept whole as the event's data"),
    ),
]);

/// `PreToolUse` → `hook_pre_tool_use`.
fn event_kind(event_type: &str) -> String {
    let mut snake = String::new();
    for c in event_type.chars() {
        if c.is_ascii_uppercase() {
            if !snake.is_empty() && !snake.ends_with('_') {
                snake.push('_');
            }
            snake.push(c.to_ascii_lowercase());
        } else if matches!(c, '-' | ' ' | '.') {
            snake.push('_');
        } else {
            snake.push(c);
        }
    }
    format!("hook_{snake}")
}

impl Parser for HookSpool {
    type ParserState = ParserState;

    fn kind(&self, value: &Json) -> String {
        str_of(value, "event_type")
            .or_else(|| str_of(value, "event"))
            .unwrap_or("?")
            .to_owned()
    }

    fn parse_record(
        &self,
        _carry: &mut ParserState,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome {
        let kind = self.kind(value);
        records.fields(&kind, value, &FIELDS);
        let Some(event_type) = str_of(value, "event_type").or_else(|| str_of(value, "event"))
        else {
            return RecordOutcome::unreadable("no event_type");
        };
        // The hook command's lines name no provider: it is installed for
        // Claude Code only.
        let provider = match str_of(value, "provider") {
            Some(p) => p,
            None if value.get("at_ns").is_some() => "claude-code",
            None => return RecordOutcome::unreadable("no provider"),
        };
        let native = str_of(value, "session_id")
            .or_else(|| value.pointer("/payload/session_id").and_then(Json::as_str))
            .filter(|s| !s.is_empty());
        let Some(native) = native else {
            return RecordOutcome::skipped("event names no session");
        };
        let Some(sid) = session_id(provider, native) else {
            return RecordOutcome::unreadable("unusable provider or session ID");
        };
        let at = value
            .get("timestamp")
            .and_then(crate::time::rfc3339)
            .or_else(|| {
                value
                    .get("observed_at_ms")
                    .and_then(crate::time::epoch_millis)
            })
            .or_else(|| {
                let ns = value.get("at_ns")?;
                let ns = ns
                    .as_str()
                    .and_then(|s| s.parse::<u64>().ok())
                    .or_else(|| ns.as_u64())?;
                polylogue_model::Timestamp::from_unix_ms((ns / 1_000_000) as i64)
            });
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(cwd) = value.pointer("/payload/cwd").and_then(Json::as_str) {
            session.offer_cwd(cwd);
        }
        out.sessions.push(session);
        let mut data = value.get("payload").cloned().unwrap_or(Json::Null);
        if let (Some(id), Json::Object(map)) = (value.get("event_id"), &mut data) {
            map.insert("event_id".into(), id.clone());
        }
        out.events.push(Event {
            session: Some(sid),
            kind: event_kind(event_type),
            at,
            after: None,
            range: Some((record.offset, record.offset + record.bytes.len() as u64)),
            data,
        });
        RecordOutcome::Used
    }

    fn flush(&self, _carry: &mut ParserState, _out: &mut Output) {}
}

#[cfg(test)]
mod tests;
