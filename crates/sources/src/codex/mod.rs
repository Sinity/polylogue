//! Codex rollout interpretation. SQLite metadata and paginated-history adapters
//! are not registered here. Unknown native kinds remain explicitly unrecognized.
use crate::frame::{IgnoredKind, Record, RecordOutcome, RecordParser};
use polylogue_model::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
pub mod keyed;
mod keyed_codec;
pub mod projection;
mod raw;
pub const PARSER_VERSION: u32 = 2;

/// Record kinds that carry nothing to keep in the index (their bytes stay stored).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ignored {
    LegacyStateMarker,
    InternalPassthrough,
}

impl IgnoredKind for Ignored {
    fn name(self) -> &'static str {
        match self {
            Self::LegacyStateMarker => "record_type: state",
            Self::InternalPassthrough => "internal_chat_message_metadata_passthrough",
        }
    }
    fn reason(self) -> &'static str {
        match self {
            Self::LegacyStateMarker => "a 2025 state marker with no content",
            Self::InternalPassthrough => "opaque provider metadata; counted, not parsed",
        }
    }
}

/// Item types that belong to the model request that produced them.
fn is_model_item(t: &str) -> bool {
    matches!(
        t,
        "reasoning" | "function_call" | "custom_tool_call" | "web_search_call" | "tool_search_call"
    )
}

fn is_shell_tool(name: &str) -> bool {
    matches!(name, "shell" | "shell_command" | "exec_command")
}

/// How a Codex session ran (`session_meta.source`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum SessionMode {
    /// `cli`, `vscode`: a person types the prompts.
    Interactive,
    /// `exec`: a non-interactive run whose prompts come from the invoking program.
    Exec,
    /// `source.subagent`: spawned by another thread, whose instructions arrive as user items.
    Subagent,
    #[default]
    Unknown,
}

/// A harness wrapper rather than a person's words: tagged context, notices and instructions.
/// An `<image …>` marker begins a typed prompt that carries an image, so it is not one.
fn is_harness_wrapper(text: &str) -> bool {
    let t = text.trim_start();
    (t.starts_with('<') && !t.starts_with("<image")) || t.starts_with("# AGENTS.md")
}

/// A tool call as later records need it: where its block is, and the fields an output reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct CallInfo {
    /// (message index, block index) of the call's block.
    at: (usize, usize),
    tool: String,
    /// For `write_stdin`, the process session the call writes to.
    stdin_session: Option<String>,
}

fn outcome_words(o: &ToolOutcome) -> (String, Option<String>) {
    let reason = match o {
        ToolOutcome::Unknown { reason } => Some(reason.as_str().to_owned()),
        ToolOutcome::Error { reason } => reason.as_ref().map(polylogue_model::ErrorReason::code),
        ToolOutcome::Ok => None,
    };
    (o.as_str().to_owned(), reason)
}

fn outcome_of_words(o: &str, reason: Option<&str>) -> ToolOutcome {
    match (o, reason) {
        ("ok", _) => ToolOutcome::Ok,
        ("error", r) => ToolOutcome::Error {
            reason: r.map(polylogue_model::ErrorReason::parse),
        },
        (_, Some(r)) => ToolOutcome::Unknown {
            reason: UnknownReason::parse(r),
        },
        (_, None) => ToolOutcome::Unknown {
            reason: UnknownReason::ResultWithoutStatus,
        },
    }
}

fn texts(content: Option<&Value>) -> Vec<String> {
    match content {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|i| match i {
                Value::String(s) => Some(s.clone()),
                _ => i.get("text").and_then(Value::as_str).map(str::to_owned),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn native_duration(item: &Value) -> Option<u64> {
    if let Some(milliseconds) = item.get("duration_ms").and_then(Value::as_u64) {
        return Some(milliseconds);
    }
    let duration = item.get("duration")?;
    let seconds = duration.get("secs")?.as_u64()?;
    let nanos = duration.get("nanos")?.as_u64()?;
    if nanos >= 1_000_000_000 {
        return None;
    }
    seconds.checked_mul(1000)?.checked_add(nanos / 1_000_000)
}
fn unit_item(unit: &keyed::EvidenceUnit) -> Option<Value> {
    let record: Value = serde_json::from_slice(&unit.canonical).ok()?;
    record.get("payload")?.get("item").cloned()
}

fn native_outcome(item: &Value) -> (ToolOutcome, Option<i64>) {
    let exit = item.get("exit_code").and_then(Value::as_i64);
    let status = item.get("status").and_then(Value::as_str);
    let outcome = match (exit, status) {
        (Some(0), _) | (None, Some("completed")) => ToolOutcome::Ok,
        (Some(_), _) | (None, Some("failed")) => ToolOutcome::Error { reason: None },
        (_, Some("declined")) => ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("declined".into())),
        },
        _ => ToolOutcome::Unknown {
            reason: UnknownReason::ResultWithoutStatus,
        },
    };
    (outcome, exit)
}

fn recorded_output(value: Option<&Value>) -> Option<ToolOutput> {
    match value? {
        Value::String(text) => Some(ToolOutput::Text(text.clone())),
        Value::Array(items)
            if items
                .iter()
                .all(|item| item.is_string() || item.get("text").is_some_and(Value::is_string)) =>
        {
            Some(ToolOutput::Text(texts(value).join("\n")))
        }
        Value::Object(object) => object
            .get("output")
            .and_then(Value::as_str)
            .map(|text| ToolOutput::Text(text.into())),
        _ => None,
    }
}

/// The outcome a tool output states (`spec/sources/codex.md`, tool calls and outcomes, sources 2 and 3).
fn output_outcome(tool: &str, output: &Value, text: &str) -> (ToolOutcome, Option<i64>) {
    let mut candidates: Vec<Value> = Vec::new();
    match output {
        Value::String(s) => {
            if let Ok(v) = serde_json::from_str::<Value>(s) {
                candidates.push(v);
            }
        }
        Value::Object(_) => candidates.push(output.clone()),
        Value::Array(items) => {
            for i in items {
                if let Some(t) = i.get("text").and_then(Value::as_str)
                    && let Ok(v) = serde_json::from_str::<Value>(t)
                {
                    candidates.push(v);
                }
            }
        }
        _ => {}
    }
    for c in &candidates {
        let code = c
            .get("exit_code")
            .or_else(|| c.get("metadata").and_then(|m| m.get("exit_code")))
            .and_then(Value::as_i64);
        if let Some(code) = code {
            return (
                if code == 0 {
                    ToolOutcome::Ok
                } else {
                    ToolOutcome::Error { reason: None }
                },
                Some(code),
            );
        }
    }
    for candidate in &candidates {
        if let Some(is_error) = candidate
            .get("is_error")
            .or_else(|| candidate.get("metadata").and_then(|m| m.get("is_error")))
            .and_then(Value::as_bool)
        {
            return (
                if is_error {
                    ToolOutcome::Error { reason: None }
                } else {
                    ToolOutcome::Ok
                },
                None,
            );
        }
    }
    for line in text.lines() {
        let rest = line
            .strip_prefix("Exit code: ")
            .or_else(|| line.strip_prefix("Process exited with code "));
        if let Some(code) = rest.and_then(|r| r.trim().parse::<i64>().ok()) {
            return (
                if code == 0 {
                    ToolOutcome::Ok
                } else {
                    ToolOutcome::Error { reason: None }
                },
                Some(code),
            );
        }
    }
    if text.starts_with("Command blocked by PreToolUse hook") {
        return (
            ToolOutcome::Error {
                reason: Some(polylogue_model::ErrorReason::BlockedByHook),
            },
            None,
        );
    }
    // Fixed result lines the tools format themselves (spec/sources/codex.md, outcome source 5;
    // `exec` status lines measured on real rollouts, evidence/codex-real-accounting.md).
    if tool == "apply_patch" {
        if text.starts_with("Success.") {
            return (ToolOutcome::Ok, None);
        }
        if text.starts_with("Failed to apply patch")
            || text.starts_with("apply_patch verification failed")
        {
            return (ToolOutcome::Error { reason: None }, None);
        }
    }
    if tool == "exec" {
        let first = text.lines().next().unwrap_or("");
        if first.starts_with("Script completed") {
            return (ToolOutcome::Ok, None);
        }
        if first.starts_with("Script failed") {
            return (ToolOutcome::Error { reason: None }, None);
        }
        if first.starts_with("aborted by user") {
            return (
                ToolOutcome::Error {
                    reason: Some(polylogue_model::ErrorReason::Interrupted),
                },
                None,
            );
        }
        if first.starts_with("Script running with cell ID") {
            // A launch acknowledgement, not a result; later `wait` calls on the cell report it.
            return (
                ToolOutcome::Unknown {
                    reason: UnknownReason::NoResult,
                },
                None,
            );
        }
    }
    (
        ToolOutcome::Unknown {
            reason: UnknownReason::ResultWithoutStatus,
        },
        None,
    )
}

fn item_id_kind(item: &Value) -> IdKind {
    if item
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        IdKind::Provider
    } else {
        IdKind::Provisional
    }
}
fn provider_time(at: Timestamp) -> ProviderTime {
    ProviderTime::Intrinsic {
        at,
        precision: Precision::Instant,
        until: None,
    }
}
fn parse_timestamp(text: &str) -> Option<Timestamp> {
    let at = text.parse::<jiff::Timestamp>().ok()?;
    Some(Timestamp(at.as_millisecond()))
}
