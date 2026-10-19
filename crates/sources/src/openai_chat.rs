//! Generic OpenAI chat transcripts: a JSON array of chat messages
//! (`[{"role", "content"}, …]`), or an object holding one (`{"messages":
//! […]}`, optionally with `id`, `model` and `title`), as many tools and
//! scripts save them. One file is one session.
//!
//! Messages carry no IDs, so each keeps its identity by its place
//! (architecture 5.5). The session is the object's `id` when it has one,
//! else the file name.

use crate::account::{Accounting, Disposition};
use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of};
use crate::fields::{Fields, Rule, unaccounted};
use crate::parse::{DocumentParser, MessageRow, Output, StepParser, Whole};
use polylogue_model::{
    Attachment, AttachmentDirection, Block, Json, Message, PromptKind, Role, Session, SessionKind,
    TitleSource, ToolOutcome, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "openai-chat";

/// An array whose first message opens with `role`, or an object whose first
/// key is `messages`.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    let json = path.extension().is_some_and(|e| e == "json");
    json && matches!(
        (first_byte(head), first_key(head)),
        (Some(b'['), Some("role")) | (Some(b'{'), Some("messages"))
    )
}

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    let name = path.file_name()?.to_str()?.rsplit("!/").next()?;
    let stem = name.strip_suffix(".json").unwrap_or(name).to_owned();
    Some(Box::new(Whole(OpenAiChat { file: stem })))
}

pub struct OpenAiChat {
    file: String,
}

const DOCUMENT_FIELDS: Fields = &[
    ("messages", Rule::Read),
    ("id", Rule::Read),
    ("model", Rule::Read),
    ("title", Rule::Read),
];

const MESSAGE_FIELDS: Fields = &[
    ("role", Rule::Read),
    ("content", Rule::Read),
    (
        "name",
        Rule::Kept("the speaker's name in a multi-party chat"),
    ),
    ("tool_calls", Rule::Read),
    ("tool_call_id", Rule::Read),
    ("reasoning", Rule::Read),
    ("reasoning_content", Rule::Read),
    ("refusal", Rule::Read),
    ("function_call", Rule::Read),
];

impl DocumentParser for OpenAiChat {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let doc = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) => v,
            Err(e) => {
                return sink.record(
                    "chat",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        let (messages, header) = match &doc {
            Json::Array(items) => (items.as_slice(), None),
            Json::Object(_) => match doc.get("messages").and_then(Json::as_array) {
                Some(items) => (items.as_slice(), Some(&doc)),
                None => return sink.record("chat", &Disposition::refused("no messages array")),
            },
            _ => return sink.record("chat", &Disposition::refused("not an array or object")),
        };
        if let Some(h) = header {
            for path in unaccounted(h, DOCUMENT_FIELDS) {
                sink.acc.unread("chat", &path);
            }
        }
        let native = header
            .and_then(|h| str_of(h, "id"))
            .unwrap_or(&self.file)
            .to_owned();
        let Some(sid) = session_id(ORIGIN, &native) else {
            return sink.record("chat", &Disposition::refused("unusable session name"));
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = header.and_then(|h| str_of(h, "title")) {
            session.offer_title(title, TitleSource::Generated);
        }
        sink.out.sessions.push(session);
        sink.record("chat", &Disposition::Materialized);
        let model = header.and_then(|h| string_of(h, "model"));

        let mut previous = None;
        for (i, m) in messages.iter().enumerate() {
            let kind = str_of(m, "role").unwrap_or("?").to_owned();
            for path in unaccounted(m, MESSAGE_FIELDS) {
                sink.acc.unread(&kind, &path);
            }
            let Some(id) = message_id(ORIGIN, &format!("{native}#{i}")) else {
                sink.record(&kind, &Disposition::refused("unusable message ID"));
                continue;
            };
            let role = match kind.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "system" | "developer" => Role::System,
                // A tool's answer reaches the model in the user's turn.
                "tool" | "function" => Role::User,
                other => {
                    sink.record(
                        &kind,
                        &Disposition::refused(format!("unknown role {other}")),
                    );
                    continue;
                }
            };
            let blocks = blocks(m, role);
            if blocks.is_empty() {
                sink.record(&kind, &Disposition::unknown("a message with no content"));
                continue;
            }
            let harness = matches!(kind.as_str(), "tool" | "function");
            let message = Message {
                id: id.clone(),
                session: sid.clone(),
                parent: previous.replace(id),
                role,
                at: None,
                model: (role == Role::Assistant).then(|| model.clone()).flatten(),
                prompt_kind: (role == Role::User).then_some(if harness {
                    PromptKind::Harness
                } else {
                    PromptKind::Prose
                }),
                blocks,
                usage: Usage::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                production: Default::default(),
            };
            let offset = sink.position();
            sink.out.messages.push(MessageRow {
                offset,
                end: offset + 1,
                message,
            });
            sink.record(&kind, &Disposition::Materialized);
        }
    }
}

fn blocks(m: &Json, role: Role) -> Vec<Block> {
    let mut blocks = Vec::new();
    for key in ["reasoning_content", "reasoning"] {
        if let Some(text) = str_of(m, key) {
            blocks.push(Block::Thinking {
                text: Some(text.to_owned()),
                signature: None,
                redacted: false,
            });
        }
    }
    let direction = if role == Role::User {
        AttachmentDirection::Input
    } else {
        AttachmentDirection::Output
    };
    let mut content = Vec::new();
    match m.get("content") {
        Some(Json::String(text)) if !text.is_empty() => {
            content.push(Block::Text { text: text.clone() })
        }
        Some(Json::Array(parts)) => {
            for p in parts {
                match str_of(p, "type") {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        if let Some(text) = str_of(p, "text") {
                            content.push(Block::Text {
                                text: text.to_owned(),
                            });
                        }
                    }
                    Some("image_url") => content.push(Block::Image(Attachment {
                        url: p
                            .pointer("/image_url/url")
                            .and_then(Json::as_str)
                            .filter(|u| !u.starts_with("data:"))
                            .map(str::to_owned),
                        direction,
                        ..Attachment::default()
                    })),
                    Some(other) => content.push(Block::Other {
                        type_name: other.to_owned(),
                        data: p.clone(),
                    }),
                    None => {}
                }
            }
        }
        _ => {}
    }
    if let Some(call_id) = str_of(m, "tool_call_id") {
        blocks.push(Block::ToolResult {
            call_id: Some(call_id.to_owned()),
            outcome: ToolOutcome::Ok,
            output: Some(
                content
                    .iter()
                    .filter_map(Block::text)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            data: None,
            attachments: Vec::new(),
        });
    } else {
        blocks.extend(content);
    }
    if let Some(refusal) = str_of(m, "refusal") {
        blocks.push(Block::Other {
            type_name: "refusal".into(),
            data: Json::String(refusal.to_owned()),
        });
    }
    let calls = m
        .get("tool_calls")
        .and_then(Json::as_array)
        .cloned()
        .unwrap_or_default();
    let legacy = m
        .get("function_call")
        .map(|f| serde_json::json!({"function": f}));
    for call in calls.iter().chain(legacy.iter()) {
        let function = call.get("function").cloned().unwrap_or(Json::Null);
        let input = function
            .get("arguments")
            .and_then(Json::as_str)
            .and_then(|a| serde_json::from_str(a).ok())
            .or_else(|| function.get("arguments").cloned())
            .unwrap_or(Json::Null);
        blocks.push(Block::ToolCall {
            call_id: string_of(call, "id"),
            tool: string_of(&function, "name"),
            input,
            command: None,
            path: None,
        });
    }
    blocks
}

#[cfg(test)]
mod tests;
