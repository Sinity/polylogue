//! Codex response items as blocks, and the outcome of a tool call read
//! from its output.
//!
//! Codex writes each item complete: a message, a reasoning summary, a tool
//! call, a tool output. Outputs carry no status field; the outcome is in
//! the text the harness writes (`Process exited with code 0`, `Exit code:
//! 1`, a JSON `metadata.exit_code`, `Script failed`). A tool whose output
//! has no such marker and that runs commands has an unknown outcome, never
//! a success.

use crate::parse::InlineBlob;
use base64::Engine;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, ErrorReason, Json, PromptKind, ToolOutcome,
    UnknownReason,
};

pub fn str_field<'a>(v: &'a Json, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Json::as_str).filter(|s| !s.is_empty())
}

fn ty(item: &Json) -> &str {
    item.get("type").and_then(Json::as_str).unwrap_or("")
}

/// The blocks of a message item's `content`.
pub fn message_blocks(content: &Json, blobs: &mut Vec<InlineBlob>) -> Vec<Block> {
    match content {
        Json::String(text) => vec![Block::Text { text: text.clone() }],
        Json::Array(items) => items.iter().map(|i| content_block(i, blobs)).collect(),
        _ => Vec::new(),
    }
}

fn content_block(item: &Json, blobs: &mut Vec<InlineBlob>) -> Block {
    match ty(item) {
        "input_text" | "output_text" | "text" | "Text" => Block::Text {
            text: str_field(item, "text").unwrap_or_default().to_owned(),
        },
        "input_image" | "image" => Block::Image(image(item, AttachmentDirection::Input, blobs)),
        "input_file" => Block::File(Attachment {
            name: str_field(item, "filename").map(str::to_owned),
            url: str_field(item, "file_url").map(str::to_owned),
            direction: AttachmentDirection::Input,
            ..Attachment::default()
        }),
        other => Block::Other {
            type_name: other.to_owned(),
            data: item.clone(),
        },
    }
}

/// An image item: a `data:` URL becomes a blob; any other URL is kept as a
/// reference.
fn image(item: &Json, direction: AttachmentDirection, blobs: &mut Vec<InlineBlob>) -> Attachment {
    let url = str_field(item, "image_url").or_else(|| str_field(item, "url"));
    let mut attachment = Attachment {
        direction,
        ..Attachment::default()
    };
    match url.and_then(data_url) {
        Some((media_type, bytes)) => {
            let hash = BlobHash::of(&bytes);
            attachment.media_type = Some(media_type);
            attachment.size = Some(bytes.len() as u64);
            attachment.blob = Some(hash);
            blobs.push(InlineBlob {
                hash,
                size: bytes.len() as u64,
                bytes,
            });
        }
        None => attachment.url = url.map(str::to_owned),
    }
    attachment
}

fn data_url(url: &str) -> Option<(String, Vec<u8>)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta.strip_suffix(";base64")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .ok()?;
    Some((media_type.to_owned(), bytes))
}

/// Text that the harness, not the operator, put in a user message. Codex
/// sends each of these as its own message.
const HARNESS_MARKERS: &[&str] = &[
    "<environment_context>",
    "# AGENTS.md instructions",
    "<user_instructions>",
    "<subagent_notification>",
    "<recommended_plugins>",
    "<codex_internal_context",
    "<turn_aborted>",
    "<skill>",
    "<user_action>",
    "<permissions instructions>",
];

pub fn prompt_kind(content: &Json) -> PromptKind {
    let first = content
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .find_map(|c| c.get("text").and_then(Json::as_str))
        })
        .map(str::trim_start)
        .unwrap_or("");
    if first.starts_with("<user_shell_command>") {
        PromptKind::Command
    } else if HARNESS_MARKERS.iter().any(|m| first.starts_with(m)) {
        PromptKind::Harness
    } else {
        PromptKind::Prose
    }
}

/// The reasoning summary or raw reasoning text; `None` when only the
/// encrypted content was written.
pub fn reasoning_block(item: &Json) -> Block {
    let mut parts: Vec<&str> = Vec::new();
    for key in ["summary", "content"] {
        if let Some(items) = item.get(key).and_then(Json::as_array) {
            parts.extend(
                items
                    .iter()
                    .filter_map(|c| c.get("text").and_then(Json::as_str)),
            );
        }
    }
    let encrypted = str_field(item, "encrypted_content").is_some();
    Block::Thinking {
        text: (!parts.is_empty()).then(|| parts.join("\n\n")),
        signature: None,
        redacted: parts.is_empty() && encrypted,
    }
}

/// The tool name a call is filed under: an MCP namespace joins its name
/// (`mcp__serena__find_symbol`); Codex's own namespaces (`collaboration`)
/// stay apart from the name with a dot.
pub fn tool_name(item: &Json) -> Option<String> {
    let name = str_field(item, "name")?;
    Some(match str_field(item, "namespace") {
        Some(ns) if ns.starts_with("mcp__") && !name.starts_with(ns) => {
            format!("{}__{name}", ns.trim_end_matches('_'))
        }
        Some(ns) if !ns.starts_with("mcp__") => format!("{ns}.{name}"),
        _ => name.to_owned(),
    })
}

/// A call's arguments: `arguments` holds JSON in a string; custom tools
/// send free text in `input`.
pub fn call_input(item: &Json) -> Json {
    if let Some(args) = item.get("arguments") {
        return match args {
            Json::String(s) => serde_json::from_str(s).unwrap_or_else(|_| args.clone()),
            other => other.clone(),
        };
    }
    item.get("input")
        .or_else(|| item.get("action"))
        .cloned()
        .unwrap_or(Json::Null)
}

/// The shell command a call runs, for search over commands.
pub fn call_command(tool: &str, input: &Json) -> Option<String> {
    let field = |k: &str| input.get(k);
    let joined = |v: &Json| match v {
        Json::String(s) => Some(s.clone()),
        Json::Array(parts) => {
            let words: Vec<&str> = parts.iter().filter_map(Json::as_str).collect();
            (!words.is_empty()).then(|| shell_line(&words))
        }
        _ => None,
    };
    match tool {
        "exec_command" | "shell_command" => {
            field("cmd").or_else(|| field("command")).and_then(joined)
        }
        "shell" | "local_shell" => field("command").and_then(joined),
        "exec" => input.as_str().map(str::to_owned),
        _ => None,
    }
}

/// `["zsh", "-lc", "git status"]` reads as the script it runs.
fn shell_line(words: &[&str]) -> String {
    match words {
        [_, flag, script] if flag.starts_with('-') && flag.ends_with('c') => (*script).to_owned(),
        _ => words.join(" "),
    }
}

/// The file a call edits, for patch tools.
pub fn call_path(tool: &str, input: &Json) -> Option<String> {
    if tool != "apply_patch" {
        return input.get("path").and_then(Json::as_str).map(str::to_owned);
    }
    let patch = input
        .as_str()
        .or_else(|| input.get("input").and_then(Json::as_str))?;
    patch.lines().find_map(|l| {
        ["*** Update File: ", "*** Add File: ", "*** Delete File: "]
            .iter()
            .find_map(|p| l.strip_prefix(p))
            .map(|p| p.trim().to_owned())
    })
}

/// A command still running when its output was written; a later
/// `write_stdin` or `wait` reports how it ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Running {
    /// A unified-exec process, continued by `write_stdin { session_id }`.
    Process(String),
    /// A code-mode script cell, continued by `wait { cell_id }`.
    Cell(String),
}

impl Running {
    pub fn key(&self) -> String {
        match self {
            Self::Process(id) => format!("process {id}"),
            Self::Cell(id) => format!("cell {id}"),
        }
    }
}

/// Tools whose output reports how a command ended. For other tools an
/// output with no failure marker is a success.
const COMMAND_TOOLS: &[&str] = &[
    "exec_command",
    "shell",
    "shell_command",
    "local_shell",
    "exec",
    "write_stdin",
    "wait",
    "apply_patch",
];

fn exit(code: i64) -> ToolOutcome {
    if code == 0 {
        ToolOutcome::Ok
    } else {
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(code)),
        }
    }
}

fn number_after<'a>(text: &'a str, marker: &str) -> Option<&'a str> {
    let at = text.find(marker)? + marker.len();
    let rest = text[at..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

/// The outcome a tool output reports, and the command it left running.
pub fn output_outcome(tool: &str, output: &str) -> (ToolOutcome, Option<Running>) {
    let head: String = output.lines().take(8).collect::<Vec<_>>().join("\n");
    if let Some(code) = number_after(&head, "Process exited with code")
        .or_else(|| number_after(&head, "Exit code:"))
        .and_then(|n| n.parse().ok())
    {
        return (exit(code), None);
    }
    if let Some(id) = number_after(&head, "Process running with session ID") {
        return (running(), Some(Running::Process(id.to_owned())));
    }
    if let Some(id) = number_after(&head, "Script running with cell ID") {
        return (running(), Some(Running::Cell(id.to_owned())));
    }
    let first = output.lines().next().unwrap_or("").trim();
    if first == "Script completed" {
        return (ToolOutcome::Ok, None);
    }
    if first == "Script failed" || first.starts_with("Script error") {
        return (ToolOutcome::Error { reason: None }, None);
    }
    if first.starts_with("aborted") || first.contains("interrupted by the user") {
        return (
            ToolOutcome::Error {
                reason: Some(ErrorReason::Interrupted),
            },
            None,
        );
    }
    if let Ok(Json::Object(map)) = serde_json::from_str::<Json>(output.trim()) {
        if let Some(code) = map
            .get("metadata")
            .and_then(|m| m.get("exit_code"))
            .and_then(Json::as_i64)
        {
            return (exit(code), None);
        }
        if map.get("success").and_then(Json::as_bool) == Some(false) {
            return (ToolOutcome::Error { reason: None }, None);
        }
    }
    if is_failure_line(first) {
        return (ToolOutcome::Error { reason: None }, None);
    }
    if COMMAND_TOOLS.contains(&tool) {
        (
            ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            },
            None,
        )
    } else {
        (ToolOutcome::Ok, None)
    }
}

fn running() -> ToolOutcome {
    ToolOutcome::Unknown {
        reason: UnknownReason::RunningInBackground,
    }
}

/// `apply_patch verification failed: …`, `write_stdin failed: …`,
/// `collab spawn failed: …`: the harness names the failed step first.
fn is_failure_line(first: &str) -> bool {
    let Some((head, _)) = first.split_once(" failed") else {
        return false;
    };
    head.len() <= 40
        && head
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ' ')
}

/// The text of a tool output, which is a string, a list of content items
/// (text and images) or a JSON value.
pub fn output_parts(output: &Json, blobs: &mut Vec<InlineBlob>) -> (String, Vec<Attachment>) {
    match output {
        Json::String(s) => (s.clone(), Vec::new()),
        Json::Array(items) => {
            let mut attachments = Vec::new();
            let mut texts = Vec::new();
            for item in items {
                match ty(item) {
                    "input_image" | "output_image" | "image" => {
                        attachments.push(image(item, AttachmentDirection::Output, blobs));
                    }
                    _ => {
                        if let Some(t) = item.get("text").and_then(Json::as_str) {
                            texts.push(t);
                        }
                    }
                }
            }
            (texts.join("\n"), attachments)
        }
        Json::Object(map) => match map.get("content").or_else(|| map.get("output")) {
            Some(inner) if !inner.is_null() => output_parts(inner, blobs),
            _ => (output.to_string(), Vec::new()),
        },
        Json::Null => (String::new(), Vec::new()),
        other => (other.to_string(), Vec::new()),
    }
}

/// How a command or tool item recorded by the thread ended, from its
/// `item_completed` record.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Completed {
    pub outcome: ToolOutcome,
    pub duration_ms: Option<u64>,
}

pub fn completed(item: &Json) -> Option<Completed> {
    let status = str_field(item, "status");
    let code = item.get("exit_code").and_then(Json::as_i64);
    let outcome = match (status, code) {
        (_, Some(code)) => exit(code),
        (Some("completed"), None) => ToolOutcome::Ok,
        (Some("failed" | "error"), None) => ToolOutcome::Error { reason: None },
        (Some("declined" | "rejected"), None) => ToolOutcome::Error {
            reason: Some(ErrorReason::Denied(status.unwrap_or_default().to_owned())),
        },
        (Some("interrupted" | "cancelled" | "aborted"), None) => ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        },
        (Some("inProgress" | "in_progress" | "running"), None) => running(),
        _ => return None,
    };
    let duration_ms = item
        .get("duration")
        .and_then(duration_ms)
        .or_else(|| item.get("durationMs").and_then(Json::as_u64));
    Some(Completed {
        outcome,
        duration_ms,
    })
}

/// `{"secs": 1, "nanos": 500000000}` or a number of milliseconds.
fn duration_ms(v: &Json) -> Option<u64> {
    if let Some(ms) = v.as_u64() {
        return Some(ms);
    }
    let secs = v.get("secs")?.as_u64()?;
    let nanos = v.get("nanos").and_then(Json::as_u64).unwrap_or(0);
    Some(secs * 1000 + nanos / 1_000_000)
}

/// The tool call and result a thread item records, for a command the
/// harness ran with no call item of its own (code-mode scripts run
/// commands this way; the Python version recovered them the same way).
pub fn recovered_call(item: &Json, blobs: &mut Vec<InlineBlob>) -> Option<Vec<Block>> {
    let id = str_field(item, "id")?.to_owned();
    let (tool, input, command, path, output) = match ty(item) {
        "CommandExecution" => {
            let command = match item.get("command") {
                Some(Json::Array(words)) => {
                    let w: Vec<&str> = words.iter().filter_map(Json::as_str).collect();
                    Some(shell_line(&w))
                }
                Some(Json::String(s)) => Some(s.clone()),
                _ => None,
            };
            let output = str_field(item, "aggregated_output")
                .or_else(|| str_field(item, "formatted_output"))
                .or_else(|| str_field(item, "stdout"))
                .map(str::to_owned);
            let input = serde_json::json!({
                "command": item.get("command"),
                "cwd": item.get("cwd"),
                "parsed_cmd": item.get("parsed_cmd"),
            });
            // A code-mode script's command: the `exec_command` tool the
            // script called (spec sources/codex.md, code mode).
            ("exec_command", input, command, None, output)
        }
        "FileChange" => {
            let changes = item.get("changes").cloned().unwrap_or(Json::Null);
            // Keyed by path; older items wrote a list of `{path, …}`.
            let path = match &changes {
                Json::Object(map) => map.keys().next().cloned(),
                Json::Array(list) => list
                    .first()
                    .and_then(|c| str_field(c, "path"))
                    .map(str::to_owned),
                _ => None,
            };
            let output = str_field(item, "stdout").map(str::to_owned);
            ("file_change", changes, None, path, output)
        }
        "McpToolCall" => {
            let server = str_field(item, "server").unwrap_or("mcp");
            let tool = str_field(item, "tool").unwrap_or("?");
            let output = item
                .get("result")
                .filter(|r| !r.is_null())
                .or_else(|| item.get("error"))
                .map(|r| output_parts(r, blobs).0);
            let input = call_input(item);
            let name = format!("mcp__{server}__{tool}");
            return Some(vec![
                Block::ToolCall {
                    call_id: Some(id.clone()),
                    tool: Some(name),
                    input,
                    command: None,
                    path: None,
                },
                result_block(&id, item, output, Vec::new()),
            ]);
        }
        "WebSearch" => {
            let input =
                serde_json::json!({ "query": item.get("query"), "action": item.get("action") });
            ("web_search", input, None, None, None)
        }
        "ImageView" => {
            let path = str_field(item, "path").map(str::to_owned);
            let input = serde_json::json!({ "path": item.get("path") });
            ("view_image", input, None, path, None)
        }
        _ => return None,
    };
    Some(vec![
        Block::ToolCall {
            call_id: Some(id.clone()),
            tool: Some(tool.to_owned()),
            input,
            command,
            path,
        },
        result_block(&id, item, output, Vec::new()),
    ])
}

fn result_block(
    id: &str,
    item: &Json,
    output: Option<String>,
    attachments: Vec<Attachment>,
) -> Block {
    let outcome = completed(item).map_or(
        ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        },
        |c| c.outcome,
    );
    Block::ToolResult {
        call_id: Some(id.to_owned()),
        outcome,
        output,
        data: None,
        attachments,
    }
}

#[cfg(test)]
mod tests;
