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
//! is information the transcript does not always hold. The session ID is
//! the harness's own, so the events land on the same session as its
//! transcript. A hook line never makes a session: until the transcript or
//! an export provides it, its events wait on a placeholder that reads do
//! not show, and a transcript a payload carries is not read.
//!
//! The daemon's process watcher, harness registry readers and commit
//! watcher write lines of the same shape under `process/`, `registry/` and
//! `git/`, with `route` naming them; their events are `process_<event>`,
//! `registry_<event>` and `git_<event>` instead of `hook_<event>`, and
//! together with hook events they are a session's live events
//! (`polylogue_model::live`).
//!
//! An event is one hook invocation. The producer gives each invocation its
//! own `event_id`, so equal payloads and timestamps still name two events.
//! That ID, the line's time and its process stay in its data; storing the
//! same file again adds none.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 5;

parser_modules! {}

use crate::document::{first_key, session_id};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::lines::Record;
use crate::parse::{Output, Parse, Parser};
use crate::{RecordOutcome, Records};
use polylogue_model::{Event, Json};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const ORIGIN: &str = "hook-spool";

/// Written only after the hook output was successfully handed over.
pub const CONTEXT_EMITTED: &str = "ContextEmitted";

// These record wrapper fields are copied alongside, rather than inside, the
// payload in Event.data. Keep decoding and event copying on the same list.
const WRAPPER_FIELDS: &[&str] = &[
    "event_id",
    "at_ns",
    "harness_pid",
    "harness_start",
    "git_identity",
    "terminal_identity",
    "repository",
    "boot_id",
    "unstated",
];

pub const LOCATIONS: &[&str] = &[
    "polylogue/hooks/*.jsonl",
    "polylogue/hooks/carriers/*/*/*.ndjson",
    "polylogue/hooks/*.ndjson",
    "polylogue/hooks/process/*.ndjson",
    "polylogue/hooks/registry/*.ndjson",
    "polylogue/hooks/git/*.ndjson",
];

/// Per-session and per-process spools are `.jsonl` or `.ndjson`.
fn is_spool_file(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e == "jsonl" || e == "ndjson")
}

/// A spool line starts `{"event_type"` (per-session files), `{"event_id"`
/// (dated files) or `{"at_ns"` (the hook command's files).
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    let head = head.bytes;
    is_spool_file(path) && matches!(first_key(head), Some("event_type" | "event_id" | "at_ns"))
}

pub fn open(path: &Path) -> Option<Parse> {
    is_spool_file(path).then(|| {
        Parse::Lines(Box::new(HookSpool {
            provider: provider_of(path),
        }))
    })
}

/// The harness a spool file's name gives: `<provider>-<session>.jsonl`
/// directly under the spool, or `carriers/<provider>/<date>/…`. Lines
/// that name their own provider keep it.
fn provider_of(path: &Path) -> Option<String> {
    fn name(path: Option<&Path>) -> Option<&str> {
        path?.file_name()?.to_str()
    }
    let parent = path.parent();
    let date_dir = parent.and_then(Path::parent);
    if name(date_dir.and_then(Path::parent)) == Some("carriers") {
        return name(date_dir).map(str::to_owned);
    }
    if name(parent) != Some("hooks") || path.extension().is_none_or(|e| e != "jsonl") {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    crate::origin::ORIGINS
        .iter()
        .map(|origin| origin.name)
        .filter(|origin| {
            stem.strip_prefix(origin)
                .and_then(|rest| rest.strip_prefix('-'))
                .is_some_and(|session| !session.is_empty())
        })
        .max_by_key(|origin| origin.len())
        .map(str::to_owned)
}

#[derive(Default)]
pub struct HookSpool {
    /// The provider the file's name gives, for lines that name none.
    provider: Option<String>,
}

/// Each line stands alone; nothing carries from one step to the next.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParserState {}

static FIELDS: Fields = Fields::new(
    "record",
    &[
        ("event_id", Rule::Read(T::String)),
        ("event_type", Rule::Read(T::String)),
        ("event", Rule::Read(T::String)),
        ("at_ns", Rule::Read(T::Number)),
        ("harness_pid", Rule::Read(T::Number)),
        ("harness_start", Rule::Read(T::Number)),
        ("route", Rule::Read(T::String)),
        ("observed_at_ms", Rule::Read(T::Number)),
        ("timestamp", Rule::Read(T::StringOrNumber)),
        ("provider", Rule::Read(T::String)),
        ("session_id", Rule::Read(T::String)),
        ("git_identity", Rule::Kept("stated Git snapshot")),
        (
            "terminal_identity",
            Rule::Kept("stated terminal adapter identity"),
        ),
        (
            "repository",
            Rule::Kept("repository named by the Git configuration"),
        ),
        ("boot_id", Rule::Kept("the source boot identity")),
        (
            "unstated",
            Rule::Kept("optional observations that were unavailable"),
        ),
        (
            "payload",
            Rule::Kept("the hook's input, kept whole as the event's data"),
        ),
    ],
);

/// The event kind of a line of `route`, or `None` for an unknown route.
fn route_prefix(route: Option<&str>) -> Option<&'static str> {
    match route {
        None | Some("hook") => Some(polylogue_model::live::HOOK_PREFIX),
        Some("process") => Some(polylogue_model::live::PROCESS_PREFIX),
        Some("registry") => Some(polylogue_model::live::REGISTRY_PREFIX),
        Some("git") => Some(polylogue_model::live::GIT_PREFIX),
        Some(_) => None,
    }
}

fn routed_kind(prefix: &str, event_type: &str) -> String {
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
    format!("{prefix}{snake}")
}

/// The hook event a line records: `event_type` in the per-session and dated
/// files, `event` in the lines the hook command writes.
fn event_type(value: &Json) -> Option<&str> {
    value
        .str("event_type")
        .filter(|s| !s.is_empty())
        .or_else(|| value.str("event").filter(|s| !s.is_empty()))
}

impl Parser for HookSpool {
    type ParserState = ParserState;

    fn kind(&self, value: &Json) -> String {
        event_type(value).unwrap_or("?").to_owned()
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
        let Some(event_type) = event_type(value) else {
            return RecordOutcome::unreadable("no event_type");
        };
        // The hook command's lines from before it named the harness are
        // Claude Code's.
        let route = match value.get("route") {
            None | Some(Json::Null) => None,
            Some(Json::String(route)) => Some(route.as_str()).filter(|s| !s.is_empty()),
            Some(_) => return RecordOutcome::unreadable("route is not a string"),
        };
        let Some(prefix) = route_prefix(route) else {
            return RecordOutcome::unreadable("unknown route");
        };
        let provider = match value.get("provider") {
            None | Some(Json::Null) => None,
            Some(Json::String(provider)) => Some(provider.as_str()).filter(|s| !s.is_empty()),
            Some(_) => return RecordOutcome::unreadable("provider is not a string"),
        };
        let provider = match provider.or(self.provider.as_deref()) {
            Some(p) => p,
            None if value.get("at_ns").is_some() => "claude-code",
            None => return RecordOutcome::unreadable("no provider"),
        };
        let native = value
            .str("session_id")
            .filter(|s| !s.is_empty())
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
            for key in WRAPPER_FIELDS {
                if let Some(v) = value.get(*key) {
                    map.insert((*key).into(), v.clone());
                }
            }
        }
        out.events.push({
            let (at, time_accuracy) = crate::time::parts(at);
            Event {
                session: Some(sid),
                kind: routed_kind(prefix, event_type),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
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
