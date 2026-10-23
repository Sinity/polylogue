//! Keep the model-facing streams beside the provider's original display data.

use crate::document::string_of;
use polylogue_model::{Block, ErrorReason, Json, ToolOutcome, UnknownReason};

fn streams(output: Option<&str>, error: Option<&str>) -> Option<String> {
    match (output, error) {
        (Some(output), Some(error))
            if !output.is_empty() && !error.is_empty() && output != error =>
        {
            Some(format!("{output}\n{error}"))
        }
        (Some(""), Some(error)) => Some(error.to_owned()),
        (Some(output), _) => Some(output.to_owned()),
        (_, Some(error)) => Some(error.to_owned()),
        _ => None,
    }
}

pub(super) fn result(call: &Json) -> Block {
    let response = call
        .get("result")
        .and_then(Json::as_array)
        .and_then(|r| r.first())
        .and_then(|r| r.pointer("/functionResponse/response"));
    let output = response
        .and_then(|r| r.get("output"))
        .and_then(Json::as_str);
    let error = response.and_then(|r| r.get("error")).and_then(Json::as_str);
    let display = call.get("resultDisplay");
    let text = streams(output, error);
    let has_result = call.get("result").is_some_and(|value| {
        !value.is_null() && value.as_array().is_none_or(|items| !items.is_empty())
    }) || display.is_some_and(|value| !value.is_null());
    let outcome = match call.get("status").and_then(Json::as_str) {
        Some("success") => ToolOutcome::Ok,
        Some("error") => ToolOutcome::Error { reason: None },
        Some("cancelled") => ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        },
        _ if has_result => ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        },
        _ => ToolOutcome::NoResult,
    };
    let mut data = serde_json::Map::new();
    for key in ["result", "resultDisplay", "status"] {
        if let Some(value) = call.get(key) {
            data.insert(key.into(), value.clone());
        }
    }
    Block::ToolResult {
        call_id: string_of(call, "id"),
        outcome,
        output: text,
        data: (!data.is_empty()).then_some(Json::Object(data)),
        attachments: Vec::new(),
    }
}
