//! Message content: blocks, tool-result outcomes and what kind of user
//! message a record is.

use crate::parse::ExtractedBlob;
use base64::Engine;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, ErrorReason, Json, PromptKind, ToolAction,
    ToolOutcome, UnknownReason,
};

/// The blocks of `message.content`, which is a string or an array.
pub fn blocks(record: &Json, content: &Json, blobs: &mut Vec<ExtractedBlob>) -> Vec<Block> {
    match content {
        Json::String(text) => vec![Block::Text { text: text.clone() }],
        Json::Array(items) => {
            let results = items.iter().filter(|i| ty(i) == "tool_result").count();
            items
                .iter()
                .map(|item| block(record, item, results == 1, blobs))
                .collect()
        }
        _ => Vec::new(),
    }
}

fn ty(item: &Json) -> &str {
    item.get("type").and_then(Json::as_str).unwrap_or("")
}

fn string(v: &Json, key: &str) -> Option<String> {
    v.get(key).and_then(Json::as_str).map(str::to_owned)
}

/// What a Claude Code tool does, by its name. Server tools (`web_search`,
/// `code_execution`) run on the API side under their own names.
pub fn action(tool: &str) -> ToolAction {
    match tool {
        "Bash" | "BashOutput" | "KillShell" => ToolAction::Shell,
        "Read" | "NotebookRead" => ToolAction::FileRead,
        "Write" => ToolAction::FileWrite,
        "Edit" | "MultiEdit" | "NotebookEdit" => ToolAction::FileEdit,
        "Grep" | "Glob" | "LS" | "ToolSearch" => ToolAction::Search,
        "WebFetch" | "WebSearch" | "web_search" | "web_fetch" => ToolAction::Web,
        "code_execution" | "bash_code_execution" => ToolAction::Code,
        "Task" | "Agent" => ToolAction::Subagent,
        t if t.starts_with("mcp__") => ToolAction::Mcp,
        _ => ToolAction::Other,
    }
}

fn block(record: &Json, item: &Json, sole_result: bool, blobs: &mut Vec<ExtractedBlob>) -> Block {
    match ty(item) {
        "text" => Block::Text {
            text: string(item, "text").unwrap_or_default(),
        },
        "thinking" => Block::Thinking {
            text: string(item, "thinking"),
            signature: string(item, "signature"),
            redacted: false,
        },
        "redacted_thinking" => Block::Thinking {
            text: None,
            signature: string(item, "data"),
            redacted: true,
        },
        "tool_use" | "server_tool_use" => {
            let input = item.get("input").cloned().unwrap_or(Json::Null);
            let tool = string(item, "name");
            Block::ToolCall {
                run_by_script: false,
                call_id: string(item, "id"),
                action: action(tool.as_deref().unwrap_or("")),
                tool,
                command: string(&input, "command"),
                path: ["file_path", "notebook_path", "path"]
                    .iter()
                    .find_map(|k| string(&input, k)),
                input,
            }
        }
        "tool_result" => tool_result(record, item, sole_result, blobs),
        "image" => Block::Image(image(item, AttachmentDirection::Input, blobs)),
        "document" => Block::File(image(item, AttachmentDirection::Input, blobs)),
        other => Block::Other {
            type_name: other.to_owned(),
            data: item.clone(),
        },
    }
}

/// An image or document block. Inline base64 bytes become a blob.
fn image(item: &Json, direction: AttachmentDirection, blobs: &mut Vec<ExtractedBlob>) -> Attachment {
    let source = item.get("source").unwrap_or(&Json::Null);
    let mut attachment = Attachment {
        media_type: string(source, "media_type"),
        url: string(source, "url"),
        direction,
        ..Attachment::default()
    };
    if string(source, "type").as_deref() == Some("base64")
        && let Some(data) = source.get("data").and_then(Json::as_str)
        && let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data)
    {
        let hash = BlobHash::of(&bytes);
        attachment.size = Some(bytes.len() as u64);
        attachment.blob = Some(hash);
        blobs.push(ExtractedBlob {
            hash,
            size: bytes.len() as u64,
            bytes,
        });
    }
    attachment
}

fn tool_result(
    record: &Json,
    item: &Json,
    sole_result: bool,
    blobs: &mut Vec<ExtractedBlob>,
) -> Block {
    // `toolUseResult` describes the record's result only when there is
    // exactly one.
    let meta = record
        .get("toolUseResult")
        .filter(|_| sole_result)
        .filter(|m| m.is_object());
    let mut attachments = Vec::new();
    let output = match item.get("content") {
        Some(Json::String(s)) => Some(s.clone()),
        Some(Json::Array(parts)) => {
            for part in parts.iter().filter(|p| ty(p) == "image") {
                attachments.push(image(part, AttachmentDirection::Output, blobs));
            }
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Json::as_str))
                .collect();
            Some(texts.join("\n"))
        }
        _ => None,
    };
    Block::ToolResult {
        call_id: string(item, "tool_use_id"),
        outcome: outcome(record, item, meta, output.as_deref()),
        output,
        data: meta.cloned(),
        attachments,
    }
}

/// The result's outcome, from the most specific evidence present: a denial,
/// an interruption, a timeout, a background launch, then `is_error`.
pub fn outcome(
    record: &Json,
    item: &Json,
    meta: Option<&Json>,
    output: Option<&str>,
) -> ToolOutcome {
    let denial = item
        .get("toolDenialKind")
        .or_else(|| record.get("toolDenialKind"))
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty());
    if let Some(kind) = denial {
        return ToolOutcome::Error {
            reason: Some(ErrorReason::Denied(kind.to_owned())),
        };
    }
    let meta_bool = |k| meta.and_then(|m| m.get(k)).and_then(Json::as_bool);
    if meta_bool("interrupted") == Some(true) {
        return ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        };
    }
    if meta
        .and_then(|m| m.get("timedOutAfterMs"))
        .is_some_and(|v| v.as_u64().is_some())
    {
        return ToolOutcome::Error {
            reason: Some(ErrorReason::TimedOut),
        };
    }
    let background = item.get("backgroundTaskId").is_some()
        || meta.is_some_and(|m| m.get("backgroundTaskId").is_some());
    if background {
        return ToolOutcome::Unknown {
            reason: UnknownReason::RunningInBackground,
        };
    }
    match item.get("is_error") {
        None | Some(Json::Null) => ToolOutcome::Ok,
        Some(Json::Bool(false)) => ToolOutcome::Ok,
        Some(Json::Bool(true)) => ToolOutcome::Error {
            reason: output.and_then(exit_code).map(ErrorReason::ExitCode),
        },
        Some(_) => ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        },
    }
}

/// The exit code a shell result prints (`Exit code 2`); the last one wins.
pub fn exit_code(text: &str) -> Option<i64> {
    text.lines().rev().find_map(|line| {
        let l = line.trim().to_ascii_lowercase();
        let rest = l
            .strip_prefix("exit code:")
            .or_else(|| l.strip_prefix("exit code"))?;
        rest.trim().parse().ok()
    })
}

/// Markers the harness writes into user messages; a message starting with
/// one was not typed by a person.
const HARNESS_MARKERS: &[&str] = &[
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
    "<system-reminder>",
    "<task-notification>",
    "[Request interrupted by user",
    "Caveat: The messages below were generated by the user while running local commands",
];

pub fn prompt_kind(record: &Json, content: &Json) -> PromptKind {
    if record.get("isMeta").and_then(Json::as_bool) == Some(true) {
        return PromptKind::Harness;
    }
    // Newer versions state who wrote the prompt.
    if let Some(kind) = record
        .get("origin")
        .and_then(|o| o.get("kind"))
        .and_then(Json::as_str)
        && kind != "human"
    {
        return if kind == "peer" {
            PromptKind::Model
        } else {
            PromptKind::Harness
        };
    }
    let text = match content {
        Json::String(s) => s.as_str(),
        Json::Array(items) => {
            if !items.is_empty() && items.iter().all(|i| ty(i) == "tool_result") {
                return PromptKind::Harness;
            }
            items
                .iter()
                .find_map(|i| i.get("text").and_then(Json::as_str))
                .unwrap_or("")
        }
        _ => "",
    };
    let t = text.trim_start();
    if HARNESS_MARKERS.iter().any(|m| t.starts_with(m)) {
        return PromptKind::Harness;
    }
    let slash = t.strip_prefix('/').is_some_and(|rest| {
        !t.trim_end().contains('\n')
            && rest.split_whitespace().next().is_some_and(|w| {
                w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':'))
            })
    });
    if slash
        || t.starts_with("<command-name>")
        || t.starts_with("<command-message>")
        || t.starts_with("<bash-input>")
        || t.starts_with('!')
        || t.starts_with('#')
    {
        PromptKind::Command
    } else {
        PromptKind::Prose
    }
}

#[cfg(test)]
mod tests;
