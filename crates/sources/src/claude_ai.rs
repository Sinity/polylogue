//! claude.ai conversations: the export's `conversations.json` (an array),
//! and the same conversation JSON a browser capture carries. Claude Design
//! chats (`design_chats/<uuid>.json` in the export) are a separate origin,
//! `claude-design`, read here too.
//!
//! A conversation is a tree by `parent_message_uuid`; every branch is
//! kept. Content blocks follow the Anthropic message format (text,
//! thinking, tool use and results); claude.ai runs some tools itself, so a
//! tool result can sit in the assistant message that made the call.

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::{Fields, Rule};
use crate::parse::{DocumentParser, InlineBlob, MessageRow, Output, Parse};
use crate::{Disposition, Records};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Message, PromptKind, Role,
    Session, SessionId, SessionKind, TitleSource, ToolAction, ToolOutcome,
};
use std::path::Path;

pub const ORIGIN: &str = "claude-ai";
pub const DESIGN_ORIGIN: &str = "claude-design";

/// The parent the export writes for a conversation's first message.
const ROOT_PARENT: &str = "00000000-0000-4000-8000-000000000000";

/// `conversations.json` whose conversations start with `uuid` (ChatGPT's
/// export has the same name and never does).
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    let named = path.file_name().and_then(|n| n.to_str()) == Some("conversations.json");
    named && first_byte(head) == Some(b'[') && first_key(head) == Some("uuid")
}

pub fn open(path: &Path) -> Option<Parse> {
    let named = path.file_name().and_then(|n| n.to_str()) == Some("conversations.json");
    named.then(|| Parse::Array(Box::new(Export)))
}

/// `design_chats/<uuid>.json`: one Claude Design chat.
pub fn recognize_design(path: &Path, head: &[u8]) -> bool {
    in_design_dir(path) && first_byte(head) == Some(b'{') && first_key(head) == Some("uuid")
}

pub fn open_design(path: &Path) -> Option<Parse> {
    in_design_dir(path).then(|| Parse::Document(Box::new(Design)))
}

fn in_design_dir(path: &Path) -> bool {
    let is_json = path.extension().is_some_and(|e| e == "json");
    let dir = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str());
    is_json && dir == Some("design_chats")
}

pub struct Export;

impl crate::array::ArrayParser for Export {
    fn order_key(&self, _: &serde_json::value::RawValue) -> Option<f64> {
        None
    }
    fn item(
        &self,
        value: &Json,
        sink: &mut Sink<'_>,
        _: &mut dyn crate::array::Owners,
    ) -> Result<(), String> {
        conversation(value, sink);
        Ok(())
    }
}

const MESSAGE_FIELDS: Fields = &[
    ("uuid", Rule::Read),
    ("text", Rule::Read),
    ("content", Rule::Read),
    ("sender", Rule::Read),
    ("index", Rule::Read),
    ("created_at", Rule::Read),
    (
        "updated_at",
        Rule::Excluded("edit time; the creation time orders messages"),
    ),
    (
        "edited_at",
        Rule::Excluded("edit time; the creation time orders messages"),
    ),
    // The model that wrote this message; it overrides the conversation's.
    ("model", Rule::Read),
    (
        "metadata",
        Rule::Kept("message settings such as the thinking configuration"),
    ),
    (
        "truncated",
        Rule::Kept("the export marks a message it shortened"),
    ),
    ("attachments", Rule::Read),
    ("files", Rule::Read),
    ("files_v2", Rule::Read),
    (
        "sync_sources",
        Rule::Kept("connected drive and repository sources"),
    ),
    ("parent_message_uuid", Rule::Read),
    ("stop_reason", Rule::Read),
    ("input_mode", Rule::Kept("how the prompt was entered")),
    ("compaction_summary", Rule::Read),
];

/// One claude.ai conversation object, shared with browser captures.
pub fn conversation(conv: &Json, sink: &mut Sink<'_>) {
    let Some(native) = str_of(conv, "uuid") else {
        sink.records
            .record("conversation", &Disposition::refused("no uuid"));
        return;
    };
    let Some(sid) = session_id(ORIGIN, native) else {
        sink.records
            .record("conversation", &Disposition::refused("unusable uuid"));
        return;
    };
    let Some(messages) = conv.get("chat_messages").and_then(Json::as_array) else {
        sink.records
            .record("conversation", &Disposition::refused("no chat_messages"));
        return;
    };
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    if let Some(name) = str_of(conv, "name") {
        session.offer_title(name, TitleSource::Generated);
    }
    if let Some(summary) = str_of(conv, "summary") {
        session.offer_title(summary, TitleSource::Summary);
    }
    if let Some(at) = conv.get("created_at").and_then(time_of) {
        session.offer_start(at);
    }
    sink.out.sessions.push(session);
    sink.records
        .record("conversation", &Disposition::Materialized);

    // Account, project, settings and model are kept whole as one event.
    if let Some(obj) = conv.as_object() {
        let settings: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, v)| k.as_str() != "chat_messages" && !v.is_null())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        sink.out.events.push(Event {
            session: sid.clone(),
            kind: "conversation_settings".into(),
            at: conv.get("updated_at").and_then(time_of),
            after: None,
            range: None,
            data: Json::Object(settings),
        });
    }

    let model = string_of(conv, "model");
    let mut rows: Vec<(Option<i64>, i64, usize, Message)> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let kind = str_of(m, "sender").unwrap_or("?").to_owned();
        sink.records.fields(&kind, m, MESSAGE_FIELDS);
        match message(&sid, native, i, m, model.as_deref()) {
            Ok(msg) => {
                // The summary claude.ai wrote when it compacted the
                // conversation: a marked discontinuity before this message.
                if let Some(summary) = m
                    .get("compaction_summary")
                    .filter(|v| v.as_array().is_some_and(|a| !a.is_empty()))
                {
                    sink.out.events.push(Event {
                        session: sid.clone(),
                        kind: "compaction".into(),
                        at: msg.at,
                        after: msg.parent.clone(),
                        range: None,
                        data: serde_json::json!({ "summary": summary }),
                    });
                }
                collect_inline(m, &mut sink.out.blobs);
                let at = msg.at.map(|t| t.unix_ms());
                let index = m.get("index").and_then(Json::as_i64).unwrap_or(i as i64);
                rows.push((at, index, i, msg));
                sink.records.record(&kind, &Disposition::Materialized);
            }
            Err(d) => sink.records.record(&kind, &d),
        }
    }
    rows.sort_by_key(|r| (r.0, r.1, r.2));
    for (_, _, _, message) in rows {
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
    }
    if let Some(leaf) = str_of(conv, "current_leaf_message_uuid")
        && let Some(id) = message_id(ORIGIN, leaf)
    {
        sink.out.events.push(Event {
            session: sid,
            kind: "active_leaf".into(),
            at: None,
            after: Some(id),
            range: None,
            data: Json::Null,
        });
    }
}

fn message(
    sid: &SessionId,
    conversation: &str,
    position: usize,
    m: &Json,
    model: Option<&str>,
) -> Result<Message, Disposition> {
    // A message the export wrote without a uuid keeps its identity by its
    // place in the conversation (architecture 5.5): the same place is the
    // same message, and changed text there is a new version of it.
    let native = match str_of(m, "uuid") {
        Some(u) => u.to_owned(),
        None => {
            let place = m
                .get("index")
                .and_then(Json::as_u64)
                .unwrap_or(position as u64);
            format!("{conversation}#{place}")
        }
    };
    let id = message_id(ORIGIN, &native).ok_or_else(|| Disposition::refused("unusable uuid"))?;
    let (role, prompt_kind, direction) = match str_of(m, "sender") {
        Some("human") => (
            Role::User,
            Some(PromptKind::Prose),
            AttachmentDirection::Input,
        ),
        Some("assistant") => (Role::Assistant, None, AttachmentDirection::Output),
        Some(other) => return Err(Disposition::unknown(format!("sender {other}"))),
        None => return Err(Disposition::refused("no sender")),
    };
    let mut blocks = Vec::new();
    let content = m.get("content").and_then(Json::as_array);
    for b in content.into_iter().flatten() {
        if let Some(block) = content_block(b, direction) {
            blocks.push(block);
        }
    }
    // Older exports carry only the flat text.
    if content.is_none_or(|c| c.is_empty())
        && let Some(text) = string_of(m, "text")
    {
        blocks.push(Block::Text { text });
    }
    blocks.extend(attachments(m, direction));
    let parent = str_of(m, "parent_message_uuid")
        .filter(|p| *p != ROOT_PARENT)
        .and_then(|p| message_id(ORIGIN, p));
    Ok(Message {
        id,
        session: sid.clone(),
        parent,
        role,
        at: m.get("created_at").and_then(time_of),
        model: (role == Role::Assistant)
            .then(|| string_of(m, "model").or_else(|| model.map(str::to_owned)))
            .flatten(),
        prompt_kind,
        blocks,
        usage: Default::default(),
        stop_reason: string_of(m, "stop_reason"),
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
    })
}

/// One content block. `token_budget` notes how much context remained; it
/// is not content and is left in the stored record.
fn content_block(b: &Json, direction: AttachmentDirection) -> Option<Block> {
    let kind = str_of(b, "type").unwrap_or("");
    Some(match kind {
        "text" => Block::Text {
            text: string_of(b, "text")?,
        },
        "voice_note" => Block::Text {
            text: string_of(b, "text")?,
        },
        "thinking" => {
            let text = string_of(b, "thinking").or_else(|| {
                // When the reasoning itself is withheld, its summaries remain.
                let summaries: Vec<&str> = b
                    .get("summaries")
                    .and_then(Json::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|s| str_of(s, "summary"))
                    .collect();
                (!summaries.is_empty()).then(|| summaries.join("\n"))
            });
            Block::Thinking {
                text,
                signature: string_of(b, "signature"),
                redacted: b.get("thinking_hidden").and_then(Json::as_bool) == Some(true),
            }
        }
        "tool_use" => Block::ToolCall {
            call_id: string_of(b, "id"),
            tool: string_of(b, "name"),
            action: action(str_of(b, "name").unwrap_or("")),
            input: b.get("input").cloned().unwrap_or(Json::Null),
            command: b
                .pointer("/input/command")
                .and_then(Json::as_str)
                .map(str::to_owned),
            path: b
                .pointer("/input/path")
                .or_else(|| b.pointer("/input/file_path"))
                .and_then(Json::as_str)
                .map(str::to_owned),
        },
        "image" => Block::Image(media_attachment(b, direction)),
        "document" => Block::File(media_attachment(b, direction)),
        "tool_result" => tool_result(b, direction),
        "token_budget" => return None,
        other => Block::Other {
            type_name: other.to_owned(),
            data: b.clone(),
        },
    })
}

fn inline_bytes(block: &Json) -> Option<Vec<u8>> {
    if !matches!(str_of(block, "type"), Some("image" | "document"))
        || block.pointer("/source/type").and_then(Json::as_str) != Some("base64")
    {
        return None;
    }
    base64::engine::general_purpose::STANDARD
        .decode(block.pointer("/source/data")?.as_str()?)
        .ok()
}

fn media_attachment(block: &Json, direction: AttachmentDirection) -> Attachment {
    let bytes = inline_bytes(block);
    Attachment {
        name: string_of(block, "name").or_else(|| string_of(block, "title")),
        media_type: block
            .pointer("/source/media_type")
            .and_then(Json::as_str)
            .map(str::to_owned),
        url: block
            .pointer("/source/url")
            .and_then(Json::as_str)
            .map(str::to_owned),
        blob: bytes.as_ref().map(|b| BlobHash::of(b)),
        size: bytes.as_ref().map(|b| b.len() as u64),
        direction,
        ..Attachment::default()
    }
}

fn collect_inline(value: &Json, blobs: &mut Vec<InlineBlob>) {
    if let Some(bytes) = inline_bytes(value) {
        let hash = BlobHash::of(&bytes);
        if !blobs.iter().any(|blob| blob.hash == hash) {
            blobs.push(InlineBlob {
                hash,
                size: bytes.len() as u64,
                bytes,
            });
        }
    }
    match value {
        Json::Array(items) => {
            for item in items {
                collect_inline(item, blobs);
            }
        }
        Json::Object(object) => {
            for item in object.values() {
                collect_inline(item, blobs);
            }
        }
        _ => {}
    }
}

/// What a claude.ai tool does, by its name.
fn action(tool: &str) -> ToolAction {
    match tool {
        "web_search" | "web_fetch" => ToolAction::Web,
        "repl" | "analysis" => ToolAction::Code,
        "google_drive_search" | "conversation_search" | "recent_chats" => ToolAction::Search,
        _ => ToolAction::Other,
    }
}

fn tool_result(b: &Json, direction: AttachmentDirection) -> Block {
    let mut texts = Vec::new();
    let mut attachments = Vec::new();
    match b.get("content") {
        Some(Json::String(s)) => texts.push(s.clone()),
        Some(Json::Array(items)) => {
            for item in items {
                match str_of(item, "type") {
                    Some("text") => {
                        if let Some(t) = string_of(item, "text") {
                            texts.push(t);
                        }
                    }
                    Some("image" | "document") => {
                        attachments.push(media_attachment(item, direction))
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let error = b.get("is_error").and_then(Json::as_bool) == Some(true);
    let structured = b
        .get("structured_content")
        .filter(|v| !v.is_null())
        .cloned();
    Block::ToolResult {
        call_id: string_of(b, "tool_use_id"),
        outcome: if error {
            ToolOutcome::Error { reason: None }
        } else {
            ToolOutcome::Ok
        },
        output: (!texts.is_empty()).then(|| texts.join("\n")),
        data: structured,
        attachments,
    }
}

/// Uploaded files: `attachments` (with the text claude.ai extracted from
/// them) and `files` (by file UUID).
fn attachments(m: &Json, direction: AttachmentDirection) -> Vec<Block> {
    let mut out = Vec::new();
    for a in m
        .get("attachments")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        out.push(Block::File(Attachment {
            name: string_of(a, "file_name"),
            media_type: string_of(a, "file_type"),
            size: a.get("file_size").and_then(Json::as_u64),
            direction,
            ..Attachment::default()
        }));
        if let Some(text) = string_of(a, "extracted_content") {
            // The text claude.ai read out of the file and gave the model.
            out.push(Block::Other {
                type_name: "attachment_text".into(),
                data: serde_json::json!({"file_name": a.get("file_name"), "text": text}),
            });
        }
    }
    for key in ["files", "files_v2"] {
        for f in m.get(key).and_then(Json::as_array).into_iter().flatten() {
            let image = str_of(f, "file_kind") == Some("image")
                || str_of(f, "file_type").is_some_and(|t| t.starts_with("image/"));
            let attachment = Attachment {
                name: string_of(f, "file_name"),
                media_type: string_of(f, "file_type"),
                source_id: string_of(f, "file_uuid"),
                direction,
                ..Attachment::default()
            };
            out.push(if image {
                Block::Image(attachment)
            } else {
                Block::File(attachment)
            });
        }
    }
    out
}

/// One Claude Design chat file.
pub struct Design;

impl DocumentParser for Design {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let mut sink = Sink::new(out, records);
        let chat = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                return sink
                    .records
                    .record("chat", &Disposition::refused("not an object"));
            }
            Err(e) => {
                return sink.records.record(
                    "chat",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        design_chat(&chat, &mut sink);
    }
}

fn design_chat(chat: &Json, sink: &mut Sink<'_>) {
    let Some(sid) = str_of(chat, "uuid").and_then(|u| session_id(DESIGN_ORIGIN, u)) else {
        sink.records
            .record("chat", &Disposition::refused("no uuid"));
        return;
    };
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    if let Some(title) = str_of(chat, "title") {
        session.offer_title(title, TitleSource::Generated);
    }
    if let Some(at) = chat.get("created_at").and_then(time_of) {
        session.offer_start(at);
    }
    sink.out.sessions.push(session);
    sink.records.record("chat", &Disposition::Materialized);
    if let Some(project) = chat.get("project").filter(|p| !p.is_null()) {
        sink.out.events.push(Event {
            session: sid.clone(),
            kind: "design_project".into(),
            at: None,
            after: None,
            range: None,
            data: project.clone(),
        });
    }
    let mut previous: Option<polylogue_model::MessageId> = None;
    for m in chat
        .get("messages")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let role_name = str_of(m, "role").unwrap_or("?").to_owned();
        let Some(id) = str_of(m, "uuid").and_then(|u| message_id(DESIGN_ORIGIN, u)) else {
            sink.records
                .record(&role_name, &Disposition::refused("no uuid"));
            continue;
        };
        let (role, prompt_kind, direction) = match role_name.as_str() {
            "user" => (
                Role::User,
                Some(PromptKind::Prose),
                AttachmentDirection::Input,
            ),
            "assistant" => (Role::Assistant, None, AttachmentDirection::Output),
            other => {
                sink.records
                    .record(other, &Disposition::unknown(format!("role {other}")));
                continue;
            }
        };
        let body = m.get("content").unwrap_or(&Json::Null);
        let mut blocks = Vec::new();
        match body.get("content") {
            Some(Json::String(text)) if !text.is_empty() => {
                blocks.push(Block::Text { text: text.clone() })
            }
            Some(Json::Array(items)) => {
                blocks.extend(items.iter().filter_map(|b| content_block(b, direction)))
            }
            _ => {}
        }
        for a in body
            .get("attachments")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            blocks.push(Block::File(Attachment {
                name: string_of(a, "name").or_else(|| string_of(a, "file_name")),
                media_type: string_of(a, "type").or_else(|| string_of(a, "mime_type")),
                direction,
                ..Attachment::default()
            }));
        }
        let message = Message {
            id: id.clone(),
            session: sid.clone(),
            parent: previous.replace(id),
            role,
            at: m
                .get("created_at")
                .and_then(time_of)
                .or_else(|| body.get("timestamp").and_then(time_of)),
            model: None,
            prompt_kind,
            blocks,
            usage: Default::default(),
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
        sink.records.record(&role_name, &Disposition::Materialized);
    }
}

#[cfg(test)]
mod tests;
