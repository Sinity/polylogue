use super::{decoded, identity, native, text};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use polylogue_model::{Block, Json, Message, MessageId, PromptKind, Role, SessionId, ToolAction};
use std::collections::HashSet;

pub(super) static FIELDS: Fields = Fields::new(&[
    ("id", Rule::Read(T::Number)),
    ("session_id", Rule::Read(T::String)),
    ("role", Rule::Read(T::String)),
    ("content", Rule::Read(T::StringOrArray)),
    ("platform_message_id", Rule::Read(T::String)),
    ("tool_calls", Rule::Read(T::Array)),
    ("tool_call_id", Rule::Read(T::String)),
    ("tool_name", Rule::Read(T::String)),
    ("reasoning_content", Rule::Read(T::StringOrArray)),
    ("reasoning", Rule::Read(T::StringOrArray)),
    ("reasoning_details", Rule::Read(T::Array)),
    ("codex_reasoning_items", Rule::Read(T::Array)),
    ("codex_message_items", Rule::Read(T::Array)),
    ("timestamp", Rule::Read(T::StringOrNumber)),
    ("active", Rule::Read(T::Number)),
    ("compacted", Rule::Read(T::Number)),
    ("observed", Rule::Read(T::Number)),
    ("token_count", Rule::Read(T::Number)),
    ("finish_reason", Rule::Read(T::String)),
    ("_empty_recovery_synthetic", Rule::Read(T::Number)),
]);

pub(super) fn active(row: &Json) -> Option<bool> {
    match (
        row.get("active").and_then(Json::as_i64),
        row.get("compacted").and_then(Json::as_i64),
    ) {
        (Some(1), _) | (_, Some(1)) => Some(true),
        (Some(0), Some(0)) => Some(false),
        _ => None,
    }
}

pub(super) fn seed(row: &Json) -> Option<String> {
    match row.get("role").and_then(Json::as_str)? {
        "user" | "assistant" | "tool" | "system" => {}
        _ => return None,
    }
    if let Some(id) = native(row, "platform_message_id") {
        return Some(format!("platform:{}", identity::component(&id)));
    }
    if let Some(id) = native(row, "tool_call_id") {
        return Some(format!("result:{}", identity::component(&id)));
    }
    if let Some(calls) = decoded(row, "tool_calls")
        && let Some(call) = calls.as_array().and_then(|calls| calls.first())
        && let Some(id) = native(call, "id")
    {
        return Some(format!("calls:{}", identity::component(&id)));
    }
    let mut value = row.as_object()?.clone();
    for field in ["id", "session_id", "active", "compacted", "observed"] {
        value.remove(field);
    }
    Some(format!("~{}", identity::hash(&Json::Object(value))))
}

pub(super) fn system(id: MessageId, session: SessionId, text: String) -> Message {
    Message {
        id,
        session,
        parent: None,
        role: Role::System,
        at: None,
        model: None,
        prompt_kind: Some(PromptKind::Harness),
        blocks: vec![Block::Text { text }],
        usage: Default::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        attrs: Default::default(),
        production: Default::default(),
    }
}

type Diagnostics = Vec<(String, Json)>;

pub(super) fn parse(
    row: &Json,
    id: MessageId,
    session: SessionId,
    model: Option<String>,
) -> Result<(Message, Diagnostics), &'static str> {
    let role = match row.get("role").and_then(Json::as_str) {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool" | "system") => Role::System,
        _ => return Err("unrecognized Hermes role; authorship was not assigned"),
    };
    let mut diagnostics = Vec::new();
    let mut blocks = Vec::new();
    for field in ["content", "reasoning", "reasoning_content", "finish_reason"] {
        if row
            .get(field)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            diagnostics.push((
                "hermes_message_diagnostic".into(),
                serde_json::json!({"field":field,"reason":"expected text; original value kept"}),
            ));
        }
    }
    if row
        .get("timestamp")
        .is_some_and(|value| !value.is_null() && crate::time::epoch_seconds(value).is_none())
    {
        diagnostics.push(("hermes_message_diagnostic".into(), serde_json::json!({"field":"timestamp","reason":"declared epoch seconds are unreadable or out of range; time remains unknown"})));
    }
    let tool = row.get("role").and_then(Json::as_str) == Some("tool");
    let content = text(row, "content");
    let harness = role == Role::System
        || row
            .get("_empty_recovery_synthetic")
            .is_some_and(|value| value == &Json::Bool(true) || value.as_i64() == Some(1))
        || content
            .as_deref()
            .is_some_and(|text| text.starts_with("[CONTEXT COMPACTION"));
    let mut covered = HashSet::<String>::new();
    let mut thinking_covered = HashSet::<String>::new();
    if let Some(reasoning) = text(row, "reasoning_content").or_else(|| text(row, "reasoning")) {
        thinking_covered.insert(reasoning.trim().to_owned());
        blocks.push(Block::Thinking {
            text: Some(reasoning),
            signature: None,
            redacted: false,
        });
    }
    let mut evidence = serde_json::Map::new();
    for field in [
        "reasoning_details",
        "codex_reasoning_items",
        "codex_message_items",
    ] {
        if let Some(raw) = row.get(field).filter(|value| !value.is_null()) {
            let decoded = raw
                .as_str()
                .and_then(|text| serde_json::from_str::<Json>(text).ok());
            if decoded.as_ref().is_none_or(|value| !value.is_array()) {
                diagnostics.push(("hermes_message_diagnostic".into(), serde_json::json!({"field":field,"reason":"structured item array unreadable; original value kept"})));
            }
            evidence.insert(
                field.into(),
                serde_json::json!({"kept":true,"json_decoded":decoded.is_some()}),
            );
        }
    }
    if !tool {
        let mut native_calls = Vec::new();
        let codex = decoded(row, "codex_message_items");
        let mut prose = Vec::new();
        if let Some(items) = codex.as_ref().and_then(Json::as_array) {
            for item in items {
                if item.get("type").and_then(Json::as_str) == Some("function_call") {
                    native_calls.push(tool_call(item));
                } else if item.get("type").and_then(Json::as_str) == Some("message") {
                    if let Some(parts) = item.get("content").and_then(Json::as_array) {
                        for part in parts {
                            if matches!(
                                part.get("type").and_then(Json::as_str),
                                Some("output_text" | "text")
                            ) && let Some(text) = text(part, "text")
                            {
                                prose.push(text);
                            }
                        }
                    }
                } else {
                    diagnostics.push(("hermes_message_diagnostic".into(), serde_json::json!({"field":"codex_message_items","reason":"unrecognized native item type; original value kept"})));
                }
            }
        }
        if let Some(content) = content
            .as_ref()
            .filter(|content| !content.is_empty() || prose.is_empty())
        {
            covered.insert(content.trim().to_owned());
            covered.extend(content.lines().map(|line| line.trim().to_owned()));
            blocks.push(Block::Text {
                text: content.clone(),
            });
        }
        for text in prose {
            if !covered.contains(text.trim()) {
                covered.insert(text.trim().to_owned());
                blocks.push(Block::Text { text });
            }
        }
        if let Some(items) = decoded(row, "codex_reasoning_items")
            .as_ref()
            .and_then(Json::as_array)
        {
            for item in items {
                if item.get("type").and_then(Json::as_str) != Some("reasoning") {
                    continue;
                }
                let mut parts = Vec::new();
                for field in ["summary", "content"] {
                    if let Some(values) = item.get(field).and_then(Json::as_array) {
                        for part in values {
                            if let Some(text) = text(part, "text")
                                && !thinking_covered.contains(text.trim())
                            {
                                thinking_covered.insert(text.trim().to_owned());
                                parts.push(text);
                            }
                        }
                    }
                }
                let signature = text(item, "encrypted_content");
                if !parts.is_empty() || signature.is_some() {
                    blocks.push(Block::Thinking {
                        text: (!parts.is_empty()).then(|| parts.join("\n\n")),
                        redacted: signature.is_some(),
                        signature,
                    });
                }
            }
        }
        let mut projected_calls = Vec::new();
        if let Some(calls) = decoded(row, "tool_calls") {
            if let Some(calls) = calls.as_array() {
                for call in calls {
                    let block = tool_call(call);
                    projected_calls.push(block.clone());
                    blocks.push(block);
                }
            } else {
                diagnostics.push((
                    "hermes_message_diagnostic".into(),
                    serde_json::json!({"reason":"tool_calls is not an array"}),
                ));
            }
        } else if row.get("tool_calls").is_some_and(|value| !value.is_null()) {
            diagnostics.push((
                "hermes_message_diagnostic".into(),
                serde_json::json!({"reason":"tool_calls JSON is unreadable"}),
            ));
        }
        for block in native_calls {
            if projected_calls.contains(&block) {
                continue;
            }
            blocks.push(block);
        }
    } else {
        let (result, diagnostic) =
            super::outcome::result(row.get("content"), native(row, "tool_call_id"));
        blocks.extend(result);
        if let Some(reason) = diagnostic {
            diagnostics.push((
                "hermes_tool_status".into(),
                serde_json::json!({"reason":reason}),
            ));
        }
    }
    if !evidence.is_empty() {
        diagnostics.push(("hermes_reasoning_evidence".into(), Json::Object(evidence)));
    }
    if row.get("active").is_some()
        || row.get("compacted").is_some()
        || row.get("observed").is_some()
    {
        let state = ["active", "compacted", "observed"]
            .into_iter()
            .filter_map(|key| row.get(key).map(|value| (key.to_owned(), value.clone())))
            .collect();
        diagnostics.push(("hermes_message_state".into(), Json::Object(state)));
    }
    let mut message = system(id, session, String::new());
    message.role = role;
    message.blocks = blocks;
    message.prompt_kind = if harness {
        Some(PromptKind::Harness)
    } else if role == Role::User {
        Some(PromptKind::Prose)
    } else {
        None
    };
    message.at = row.get("timestamp").and_then(crate::time::epoch_seconds);
    message.model = (role == Role::Assistant && !harness)
        .then_some(model)
        .flatten();
    message.stop_reason = text(row, "finish_reason");
    if let Some(tokens) = row.get("token_count").and_then(Json::as_u64) {
        if role == Role::User {
            message.usage.input = Some(tokens);
        } else if role == Role::Assistant {
            message.usage.output = Some(tokens);
        }
    }
    Ok((message, diagnostics))
}

fn tool_call(call: &Json) -> Block {
    let function = call.get("function").unwrap_or(call);
    let tool = text(function, "name");
    let input = match function.get("arguments") {
        Some(Json::String(text)) => {
            serde_json::from_str(text).unwrap_or_else(|_| Json::String(text.clone()))
        }
        Some(value) => value.clone(),
        None => Json::Null,
    };
    let action = action(tool.as_deref());
    let command = input
        .get("command")
        .or_else(|| input.get("cmd"))
        .and_then(Json::as_str)
        .map(str::to_owned);
    let path = text(&input, "path");
    Block::ToolCall {
        run_by_script: false,
        call_id: native(call, "id").or_else(|| native(call, "call_id")),
        tool,
        action,
        input,
        command,
        path,
    }
}

pub(super) fn action(tool: Option<&str>) -> ToolAction {
    match tool {
        Some("terminal" | "shell" | "exec_command") => ToolAction::Shell,
        Some("python" | "run_code") => ToolAction::Code,
        Some("read_file") => ToolAction::FileRead,
        Some("write_file") => ToolAction::FileWrite,
        Some("edit_file") => ToolAction::FileEdit,
        Some("search_files" | "list_directory") => ToolAction::Search,
        Some("web_search" | "web_fetch") => ToolAction::Web,
        Some("delegate_task") => ToolAction::Subagent,
        _ => ToolAction::Other,
    }
}
