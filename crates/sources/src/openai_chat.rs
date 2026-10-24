//! Generic OpenAI chat transcripts: a JSON array of chat messages
//! (`[{"role", "content"}, …]`), or an object holding one (`{"messages":
//! […]}`, optionally with `id`, `model` and `title`), as many tools and
//! scripts save them. One file is one session.
//!
//! Messages carry no IDs, so each keeps its identity by its place
//! (architecture 5.5). The session is the object's `id` when it has one,
//! else the file name.

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse};
use crate::{Disposition, Records};
use polylogue_model::{
    Attachment, AttachmentDirection, Block, Json, Message, PromptKind, Role, Session, SessionKind,
    TitleSource, ToolAction, ToolOutcome, Usage,
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

pub fn open(path: &Path) -> Option<Parse> {
    let name = path.file_name()?.to_str()?.rsplit("!/").next()?;
    let stem = name.strip_suffix(".json").unwrap_or(name).to_owned();
    Some(Parse::Export(Box::new(OpenAiChat {
        file: stem,
        state: ChatState::default(),
    })))
}

pub struct OpenAiChat {
    file: String,
    state: ChatState,
}

#[derive(Default)]
struct ChatState {
    metadata: Option<Json>,
    session: Option<polylogue_model::SessionId>,
    model: Option<String>,
    previous: Option<polylogue_model::MessageId>,
    position: usize,
}

static DOCUMENT_FIELDS: Fields = Fields::new(&[
    ("messages", Rule::Read),
    ("id", Rule::Read),
    ("model", Rule::Read),
    ("title", Rule::Read),
]);

static MESSAGE_FIELDS: Fields = Fields::new(&[
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
]);

impl ExportParser for OpenAiChat {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.state = ChatState::default();
        let (metadata, stats) =
            crate::export::scan_array_or_nested(reader, &["messages"], |range, raw| {
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            })?;
        self.state.metadata = Some(metadata);
        Ok(stats)
    }
    fn begin(
        &mut self,
        length: u64,
        _first: Option<&[u8]>,
    ) -> Result<Option<crate::parse::Step>, String> {
        let doc = self
            .state
            .metadata
            .take()
            .ok_or("chat metadata not scanned")?;
        let mut output = Output::default();
        let mut records = Records::default();
        self.start_session(&doc, &mut Sink::new(&mut output, &mut records));
        Ok(Some(crate::parse::Step {
            output,
            accounting: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            snapshot: None,
        }))
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn ExportState,
    ) -> Result<(), String> {
        let m: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let m = &m;
        let Some(sid) = self.state.session.clone() else {
            return Ok(());
        };
        let native = sid.native().as_str();
        let i = self.state.position;
        self.state.position += 1;
        let kind = str_of(m, "role").unwrap_or("?").to_owned();
        sink.records.fields(&kind, m, &MESSAGE_FIELDS);
        let Some(id) = message_id(ORIGIN, &format!("{native}#{i}")) else {
            sink.records
                .record(&kind, &Disposition::refused("unusable message ID"));
            return Ok(());
        };
        let role = match kind.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" | "developer" => Role::System,
            // A tool's answer reaches the model in the user's turn.
            "tool" | "function" => Role::User,
            other => {
                sink.records.record(
                    &kind,
                    &Disposition::refused(format!("unknown role {other}")),
                );
                return Ok(());
            }
        };
        let blocks = blocks(m, role);
        if blocks.is_empty() {
            sink.records
                .record(&kind, &Disposition::unknown("a message with no content"));
            return Ok(());
        }
        let harness = matches!(kind.as_str(), "tool" | "function");
        let message = Message {
            id: id.clone(),
            session: sid.clone(),
            parent: self.state.previous.replace(id),
            role,
            at: None,
            model: (role == Role::Assistant)
                .then(|| self.state.model.clone())
                .flatten(),
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
        sink.records.record(&kind, &Disposition::Materialized);
        Ok(())
    }
}

impl OpenAiChat {
    fn start_session(&mut self, doc: &Json, sink: &mut Sink<'_>) {
        let header = match doc {
            Json::Array(_) => None,
            Json::Object(_) => match doc.get("messages").and_then(Json::as_array) {
                Some(_) => Some(doc),
                None => {
                    return sink
                        .records
                        .record("chat", &Disposition::refused("no messages array"));
                }
            },
            _ => {
                return sink
                    .records
                    .record("chat", &Disposition::refused("not an array or object"));
            }
        };
        if let Some(h) = header {
            sink.records.fields("chat", h, &DOCUMENT_FIELDS);
        }
        let native = header
            .and_then(|h| str_of(h, "id"))
            .unwrap_or(&self.file)
            .to_owned();
        let Some(sid) = session_id(ORIGIN, &native) else {
            return sink
                .records
                .record("chat", &Disposition::refused("unusable session name"));
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = header.and_then(|h| str_of(h, "title")) {
            session.offer_title(title, TitleSource::Generated);
        }
        sink.out.sessions.push(session);
        sink.records.record("chat", &Disposition::Materialized);
        self.state.session = Some(sid);
        self.state.model = header.and_then(|h| string_of(h, "model"));
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
            action: ToolAction::Other,
            input,
            command: None,
            path: None,
        });
    }
    blocks
}

#[cfg(test)]
mod tests;
