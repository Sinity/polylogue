//! The hook spool: one JSON line per harness hook event (session start,
//! prompt submitted, tool about to run, tool finished, compaction, stop),
//! appended without a sync. Two layouts share the format: a file
//! per session (`<provider>-<session>.jsonl`) and dated files
//! (`carriers/<provider>/<date>/*.ndjson`, each holding the hook lines one
//! process wrote). The `polylogue hook` command writes a
//! third: one file per harness process (`<pid>.ndjson`), lines
//! `{"at_ns", "event", "harness_pid", "provider", "session_id", "payload"}`
//! for Claude Code, Codex and Hermes; lines written before it named the
//! harness carry no `provider` and are Claude Code's.
//!
//! Each line becomes an event on the session it names, kept whole: what
//! the hook saw (tool input, permission mode, effort, working directory)
//! is evidence the transcript does not always hold. The session ID is
//! the harness's own, so the events land on the same session as its
//! transcript. A hook line never makes a session: until the transcript or
//! an export provides it, its events wait on a placeholder that reads do
//! not show, and a transcript a payload carries is not read.
//!
//! An event is one hook invocation. The producer gives each invocation its
//! own `event_id`, so equal payloads and timestamps still name two events.
//! That ID, the line's time and its process stay in its data; storing the
//! same file again adds none.

/// Version of the rows this module produces.
pub const VERSION: u32 = 2;

use crate::document::{first_key, session_id, str_of};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::lines::Record;
use crate::parse::{Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ORIGIN: &str = "hook-spool";

/// Written only after the native hook output was successfully handed over.
pub const CONTEXT_EMITTED: &str = "HookContextEmitted";

// These native envelope fields are copied alongside, rather than inside, the
// payload in Event.data. Keep decoding and event retention on the same list.
const ENVELOPE_FIELDS: &[&str] = &[
    "event_id",
    "at_ns",
    "harness_pid",
    "git_identity",
    "terminal_identity",
    "repository",
    "boot_id",
    "native_unavailable",
];

/// Decode an emission claim retained by this parser. This is not native
/// admission and confers no authority on imported bytes. The receipt proves no
/// transcript-message attribution without an exact native message link.
pub fn context_emission(event: &Event) -> Option<polylogue_model::HookContextEmission> {
    if event.kind != event_kind(CONTEXT_EMITTED) {
        return None;
    }
    let mut payload = event.data.as_object()?.clone();
    let receipt = payload.get("event_id")?.as_str()?.to_owned();
    for key in ENVELOPE_FIELDS {
        payload.remove(*key);
    }
    let claim: polylogue_model::HookContextEmission =
        serde_json::from_value(Json::Object(payload)).ok()?;
    if receipt.is_empty() || claim.invocation.is_empty() || receipt == claim.invocation
        || claim.hook_event.is_empty() || event.session.as_ref() != Some(&claim.session)
        || claim.deliveries.is_empty()
        || !claim.deliveries.iter().all(|reference| {
            matches!(polylogue_model::Reference::parse(reference),
                Ok(parsed @ polylogue_model::Reference::Delivery(_)) if parsed.to_string() == *reference)
        })
    {
        return None;
    }
    Some(claim)
}

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

/// Each line stands alone; nothing carries from one step to the next.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {}

static FIELDS: Fields = Fields::new(&[
    ("event_id", Rule::Read(T::String)),
    ("event_type", Rule::Read(T::String)),
    ("event", Rule::Read(T::String)),
    ("at_ns", Rule::Read(T::Number)),
    ("harness_pid", Rule::Read(T::Number)),
    ("observed_at_ms", Rule::Read(T::Number)),
    ("timestamp", Rule::Read(T::StringOrNumber)),
    ("provider", Rule::Read(T::String)),
    ("session_id", Rule::Read(T::String)),
    ("git_identity", Rule::Kept("native Git snapshot")),
    (
        "terminal_identity",
        Rule::Kept("native terminal adapter identity"),
    ),
    (
        "repository",
        Rule::Kept("repository named by native Git configuration"),
    ),
    ("boot_id", Rule::Kept("the source boot identity")),
    (
        "native_unavailable",
        Rule::Kept("unavailable optional native observations"),
    ),
    (
        "payload",
        Rule::Kept("the hook's input, kept whole as the event's data"),
    ),
])
.for_kind("record");

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
        // The hook command's lines from before it named the harness are
        // Claude Code's.
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
            .or_else(|| value.get("at_ns").and_then(crate::time::epoch_nanos))
            .map(crate::time::SourceTime::captured);
        let mut data = match value.get("payload") {
            Some(Json::Object(map)) => Json::Object(map.clone()),
            Some(Json::Null) | None => Json::Object(Default::default()),
            Some(other) => serde_json::json!({ "payload": other }),
        };
        if let Json::Object(map) = &mut data {
            for key in ENVELOPE_FIELDS {
                if let Some(v) = value.get(*key) {
                    map.insert((*key).into(), v.clone());
                }
            }
        }
        out.events.push({
            let (at, time_evidence) = crate::time::parts(at);
            Event {
                session: Some(sid),
                kind: event_kind(event_type),
                at,
                at_quality: time_evidence.quality,
                at_precision: time_evidence.precision,
                at_until: time_evidence.until,
                after: None,
                range: Some((record.offset, record.offset + record.bytes.len() as u64)),
                data,
            }
        });
        RecordOutcome::Used
    }

    fn flush(&self, _carry: &mut ParserState, _out: &mut Output) {}
}

#[cfg(test)]
mod tests;
