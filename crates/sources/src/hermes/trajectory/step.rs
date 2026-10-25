//! ATIF step text and actions. Recorded source roles are descriptors; they do
//! not establish model authorship or delivery to a model.
use polylogue_model::{Attachment, AttachmentDirection, Block, Json, ToolOutcome, UnknownReason};

pub(super) fn blocks(step: &Json) -> (Vec<Block>, Vec<&'static str>) {
    let mut out = Vec::new();
    let mut limits = Vec::new();
    if let Some(text) = step.get("reasoning_content").and_then(Json::as_str) {
        out.push(Block::Thinking {
            text: Some(text.into()),
            signature: None,
            redacted: false,
        });
    } else if step.get("reasoning_content").is_some_and(|v| !v.is_null()) {
        limits.push("reasoning_content is not text");
    }
    match step.get("message") {
        Some(value) => content(value, &mut out, &mut limits),
        None => limits.push("missing required message; other supplied content remains readable"),
    }
    if let Some(calls) = step.get("tool_calls") {
        if let Some(calls) = calls.as_array() {
            for call in calls {
                let id = native(call, "tool_call_id");
                let tool = native(call, "function_name");
                let args = call.get("arguments").filter(|v| v.is_object());
                if let (Some(id), Some(tool), Some(args)) = (id, tool, args) {
                    out.push(Block::ToolCall {
                        call_id: Some(id.into()),
                        tool: Some(tool.into()),
                        action: super::super::message::action(Some(tool)),
                        input: args.clone(),
                        command: args
                            .get("command")
                            .or_else(|| args.get("cmd"))
                            .and_then(Json::as_str)
                            .map(str::to_owned),
                        path: args.get("path").and_then(Json::as_str).map(str::to_owned),
                    });
                } else {
                    limits.push("tool call requires textual IDs/name and object arguments; original call kept");
                    out.push(Block::Other {
                        type_name: "atif_tool_call".into(),
                        data: call.clone(),
                    });
                }
            }
        } else if !calls.is_null() {
            limits.push("tool_calls is not an array; original field kept");
        }
    }
    if let Some(observation) = step.get("observation").filter(|v| !v.is_null()) {
        if let Some(results) = observation.get("results").and_then(Json::as_array) {
            for result in results {
                let Some(fields) = result.as_object() else {
                    limits.push("observation result is not an object; original result kept");
                    out.push(Block::Other {
                        type_name: "atif_observation".into(),
                        data: result.clone(),
                    });
                    continue;
                };
                let content = fields.get("content");
                let (output, attachments) = result_content(content, &mut limits);
                let mut data = fields.clone();
                data.remove("source_call_id");
                if content.is_some_and(|v| v.is_string() || v.is_array()) {
                    // Ordered text is projected once through the shared block
                    // masker. Original part boundaries remain in the source.
                    data.remove("content");
                }
                // ATIF declares no success/failure field. A result's prose or
                // provider-specific extra data cannot establish an outcome.
                out.push(Block::ToolResult {
                    call_id: native(result, "source_call_id").map(str::to_owned),
                    outcome: ToolOutcome::Unknown {
                        reason: UnknownReason::StatusFieldUnreadable,
                    },
                    output,
                    data: (!data.is_empty()).then_some(Json::Object(data)),
                    attachments,
                });
            }
        } else {
            limits.push("observation requires a results array; original observation kept");
            out.push(Block::Other {
                type_name: "atif_observation".into(),
                data: observation.clone(),
            });
        }
    }
    (out, limits)
}

fn native<'a>(value: &'a Json, field: &str) -> Option<&'a str> {
    value
        .get(field)
        .and_then(Json::as_str)
        .filter(|s| !s.is_empty())
}

fn content(value: &Json, blocks: &mut Vec<Block>, limits: &mut Vec<&'static str>) {
    if let Some(text) = value.as_str() {
        blocks.push(Block::Text { text: text.into() });
    } else if let Some(parts) = value.as_array() {
        for part in parts {
            if part.get("type").and_then(Json::as_str) == Some("text")
                && let Some(text) = part.get("text").and_then(Json::as_str)
            {
                blocks.push(Block::Text { text: text.into() });
            } else if let Some(image) = image(part) {
                blocks.push(Block::Image(image));
            } else {
                limits.push("unrecognized ATIF content part; native reference kept without inferred model direction");
                blocks.push(Block::Other {
                    type_name: "atif_content_part".into(),
                    data: part.clone(),
                });
            }
        }
    } else {
        limits.push("message is neither text nor a content-part array; original value kept");
        blocks.push(Block::Other {
            type_name: "atif_message".into(),
            data: value.clone(),
        });
    }
}

fn image(value: &Json) -> Option<Attachment> {
    if value.get("type").and_then(Json::as_str) != Some("image") {
        return None;
    }
    let source = value.get("source")?;
    let path = native(source, "path")?;
    let media = native(source, "media_type")?;
    Some(Attachment {
        name: Some(path.into()),
        source_id: Some(path.into()),
        media_type: Some(media.into()),
        direction: AttachmentDirection::Unknown,
        ..Attachment::default()
    })
}

fn result_content(
    value: Option<&Json>,
    limits: &mut Vec<&'static str>,
) -> (Option<String>, Vec<Attachment>) {
    match value {
        Some(Json::String(text)) => (Some(text.clone()), vec![]),
        Some(Json::Array(parts)) => {
            let mut text = String::new();
            let mut images = Vec::new();
            for part in parts {
                if part.get("type").and_then(Json::as_str) == Some("text")
                    && let Some(value) = part.get("text").and_then(Json::as_str)
                {
                    text.push_str(value);
                } else if let Some(image) = image(part) {
                    images.push(image);
                } else {
                    limits.push("unsupported ATIF result content part; original array remains in the source");
                    return (None, vec![]);
                }
            }
            (Some(text), images)
        }
        Some(value) if !value.is_null() => {
            limits
                .push("result content is neither text nor content parts; original value kept");
            (None, vec![])
        }
        _ => (None, vec![]),
    }
}
