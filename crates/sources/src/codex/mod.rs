//! Codex rollout interpretation. SQLite metadata and paginated-history adapters
//! are not registered here. Unknown native kinds remain explicitly unrecognized.
use crate::frame::{IgnoredKind, Record, RecordOutcome, RecordParser};
use polylogue_model::RetainedJson as Value;
use polylogue_model::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub mod history;
pub mod keyed;
mod keyed_codec;
pub mod origins;
pub mod presence;
pub mod projection;
pub mod prompt_history;
mod raw;
pub mod selection;
pub const PARSER_VERSION: u32 = 11;

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

fn text_parts_with_paths(
    content: Option<&Value>,
) -> impl Iterator<Item = (Vec<selection::PathPart>, &str)> {
    use selection::PathPart::{Index, Key};
    let single = content
        .and_then(Value::as_str)
        .map(|text| (Vec::new(), text))
        .into_iter();
    let array = content
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(i, item)| {
            item.as_str()
                .map(|text| (vec![Index(i as u64)], text))
                .or_else(|| {
                    item.get("text")
                        .and_then(Value::as_str)
                        .map(|text| (vec![Index(i as u64), Key("text".into())], text))
                })
        });
    single.chain(array)
}
fn text_parts(content: Option<&Value>) -> impl Iterator<Item = &str> {
    text_parts_with_paths(content).map(|(_, text)| text)
}
fn texts(content: Option<&Value>) -> Vec<String> {
    text_parts(content).map(str::to_owned).collect()
}
fn has_nonempty_text(content: Option<&Value>) -> bool {
    let mut parts = text_parts(content);
    parts
        .next()
        .is_some_and(|first| !first.is_empty() || parts.next().is_some())
}

fn native_duration(item: &Value) -> Option<u64> {
    if let Some(milliseconds) = item.get("duration_ms").and_then(Value::as_u64) {
        return Some(milliseconds);
    }
    if let Some(duration) = item.get("duration") {
        let seconds = duration.get("secs")?.as_u64()?;
        let nanos = duration.get("nanos")?.as_u64()?;
        if nanos >= 1_000_000_000 {
            return None;
        }
        return seconds.checked_mul(1000)?.checked_add(nanos / 1_000_000);
    }
    output_duration(item.get("output")?)
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

fn has_recorded_output(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(_)) => true,
        Some(Value::Array(items)) => items
            .iter()
            .all(|item| item.is_string() || item.get("text").is_some_and(Value::is_string)),
        Some(Value::Object(object)) => object.get(&"output".into()).is_some_and(Value::is_string),
        _ => false,
    }
}
fn recorded_output(value: Option<&Value>) -> Option<ToolOutput> {
    if !has_recorded_output(value) {
        return None;
    }
    match value? {
        Value::String(text) => Some(ToolOutput::Text(text.text().to_owned())),
        Value::Array(_) => Some(ToolOutput::Text(texts(value).join("\n"))),
        Value::Object(object) => object
            .get(&"output".into())
            .and_then(Value::as_str)
            .map(|text| ToolOutput::Text(text.into())),
        _ => None,
    }
}

fn wall_seconds(line: &str, prefix: &str) -> bool {
    line.strip_prefix(prefix)
        .and_then(|s| s.strip_suffix(" seconds"))
        .is_some_and(|n| {
            !n.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit() || b == b'.')
                && n.parse::<f64>().is_ok()
        })
}

/// Only provider-written headers carry verdicts. A command preamble occupies
/// one line; its embedded newline cannot move the header's anchored position.
fn result_header_exit(text: &str) -> Option<i64> {
    let mut lines = text.lines();
    let mut first = lines.next()?;
    if first.starts_with("Command: ") {
        first = lines.next()?;
    }
    if first.starts_with("Chunk ID: ") {
        first = lines.next()?;
    }
    if let Some(code) = first.strip_prefix("Exit code: ") {
        let code = code.parse().ok()?;
        let wall = lines.next()?;
        if !wall_seconds(wall, "Wall time: ") {
            return None;
        }
        let next = lines.next()?;
        return (next == "Output:"
            || next
                .strip_prefix("Total output lines: ")
                .is_some_and(|n| n.parse::<u64>().is_ok()))
        .then_some(code);
    }
    command_header_status(text)?
        .strip_prefix("Process exited with code ")
        .or_else(|| command_header_status(text)?.strip_prefix("Process completed with exit code "))?
        .parse()
        .ok()
}
fn command_header_status(text: &str) -> Option<&str> {
    let mut lines = text.lines();
    let mut first = lines.next()?;
    if first.starts_with("Command: ") {
        first = lines.next()?;
    }
    if first.starts_with("Chunk ID: ") {
        first = lines.next()?;
    }
    wall_seconds(first, "Wall time: ")
        .then(|| lines.next())
        .flatten()
}
fn running_session(text: &str) -> Option<&str> {
    let status = if text.starts_with("Process running with session ID ") {
        text.lines().next()?
    } else {
        command_header_status(text)?
    };
    let session = status.strip_prefix("Process running with session ID ")?;
    (!session.is_empty() && session.bytes().all(|c| c.is_ascii_digit())).then_some(session)
}
/// For content-item output, Codex emits the status as a complete separate
/// item. Text following its Output marker is user data, not that status item.
fn complete_script_status(output: &Value) -> bool {
    let Some(items) = output.as_array() else {
        return true;
    };
    let Some(header) = items
        .first()
        .and_then(|item| item.get("text"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    let mut lines = header.lines();
    let Some(_) = lines.next() else {
        return false;
    };
    let Some(wall) = lines.next() else {
        return false;
    };
    wall_seconds(wall, "Wall time ") && lines.next() == Some("Output:") && lines.next().is_none()
}

fn output_duration(output: &Value) -> Option<u64> {
    let decoded = match output {
        Value::String(text) => scalar_json(text.text()),
        Value::Object(_) => Some(output.clone()),
        _ => None,
    };
    if let Some(seconds) = decoded
        .as_ref()
        .and_then(|v| v.get("metadata"))
        .and_then(|v| v.get("duration_seconds"))
    {
        return crate::decimal::scaled_integer(seconds, 3)?.try_into().ok();
    }
    let (text, prefix) = match output {
        Value::String(text)
            if result_header_exit(text.text()).is_some()
                || running_session(text.text()).is_some() =>
        {
            (text.text(), "Wall time: ")
        }
        Value::Array(items) if complete_script_status(output) => {
            (items.first()?.get("text")?.as_str()?, "Wall time ")
        }
        _ => return None,
    };
    let wall = text.lines().find(|line| line.starts_with(prefix))?;
    let seconds = wall.strip_prefix(prefix)?.strip_suffix(" seconds")?;
    let value = Value::parse(seconds.as_bytes()).ok()?;
    crate::decimal::scaled_integer(&value, 3)?.try_into().ok()
}

/// The outcome a tool output states (`spec/sources/codex.md`, tool calls and outcomes, sources 2 and 3).
fn output_outcome(tool: &str, output: &Value, text: &str) -> (ToolOutcome, Option<i64>) {
    let mut candidates: Vec<Value> = Vec::new();
    match output {
        Value::String(s) => {
            if let Some(v) = scalar_json(s.text()) {
                candidates.push(v);
            }
        }
        Value::Object(_) => candidates.push(output.clone()),
        Value::Array(items) => {
            for i in items {
                if let Some(t) = i.get("text").and_then(Value::as_str)
                    && let Some(v) = scalar_json(t)
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
    if let Some(code) = result_header_exit(text) {
        return (
            if code == 0 {
                ToolOutcome::Ok
            } else {
                ToolOutcome::Error { reason: None }
            },
            Some(code),
        );
    }
    if (is_shell_tool(tool) || tool == "write_stdin") && running_session(text).is_some() {
        return (
            ToolOutcome::Unknown {
                reason: UnknownReason::NoResult,
            },
            None,
        );
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
    if tool == "exec" && complete_script_status(output) {
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

/// Codex's current mapping remains scalar-only. Reject before keyed state or
/// identity construction; lexical opaque numbers still use the retained tree.
fn require_scalar_json(value: &Value) -> Result<(), String> {
    match value {
        Value::String(value) if value.native().encoding().is_some() => {
            Err("Codex non-scalar JSON strings are not yet supported".into())
        }
        Value::Array(values) => values.iter().try_for_each(require_scalar_json),
        Value::Object(fields) => {
            for (key, value) in fields {
                if key.encoding().is_some() {
                    return Err("Codex non-scalar JSON keys are not yet supported".into());
                }
                require_scalar_json(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn scalar_json(text: &str) -> Option<Value> {
    let value = Value::parse(text.as_bytes()).ok()?;
    require_scalar_json(&value).ok()?;
    Some(value)
}

#[cfg(test)]
#[path = "scalar_compatibility_tests.rs"]
mod scalar_compatibility_tests;
