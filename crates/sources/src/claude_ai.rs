//! claude.ai conversations: the export's `conversations.json` (an array),
//! and the same conversation JSON a browser capture carries. Claude Design
//! chats (`design_chats/<uuid>.json` in the export) are a separate origin,
//! `claude-design`, read here too.
//!
//! A conversation is a tree by `parent_message_uuid`; every branch is
//! kept. Content blocks follow the Anthropic message format (text,
//! thinking, tool use and results); claude.ai runs some tools itself, so a
//! tool result can sit in the assistant message that made the call.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 5;

parser_modules! {
    mod fields;
    mod sidecars;
}

use crate::fields::ExpectedType as T;

use crate::document::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, time_of};
use crate::fields::{Fields, Rule};
use crate::parse::{DocumentParser, ExtractedBlob, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Message, MessageId, PromptKind,
    Role, Session, SessionId, SessionKind, TitleSource, ToolAction, ToolOutcome,
};
use std::path::Path;

pub const ORIGIN: &str = "claude-ai";
pub const DESIGN_ORIGIN: &str = "claude-design";

/// The parent the export writes for a conversation's first message.
const ROOT_PARENT: &str = "00000000-0000-4000-8000-000000000000";

/// `conversations.json` whose conversations start with `uuid` (ChatGPT's
/// export has the same name and never does).
///
/// One conversation alone (`uuid` and `chat_messages`) is recognized
/// whatever the file is named, once the whole document is read. A project
/// read whole names itself by its `docs` or `prompt_template`; its path alone
/// does not.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    if head.shape.is_some()
        && head.holds("uuid", Kind::String)
        && head.holds("chat_messages", Kind::Array)
    {
        return true;
    }
    if sidecars::recognize(path, head.bytes) {
        return head.shape.is_none()
            || !project_path(path)
            || (head.holds("uuid", Kind::String)
                && (head.has("docs") || head.has("prompt_template")));
    }
    let head = head.bytes;
    let named = path.extension().is_some_and(|ext| ext == "json");
    named && first_byte(head) == Some(b'[') && first_key(head) == Some("uuid")
}

/// A project of the export: `projects.json`, or a file under `projects/`.
/// Its path alone declares no export around it.
pub fn project_path(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension == "json")
        && (path.file_name().is_some_and(|name| name == "projects.json")
            || path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "projects"))
}

pub fn open(path: &Path) -> Option<Parse> {
    if let Some(parser) = sidecars::open(path) {
        return Some(parser);
    }
    let named = path.extension().is_some_and(|ext| ext == "json");
    named.then(|| Parse::Export(Box::new(Export)))
}

/// `design_chats/<uuid>.json`: one Claude Design chat.
pub fn recognize_design(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    in_design_dir(path) && head.top() == Some(Kind::Object) && head.holds("uuid", Kind::String)
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

impl crate::export::ExportParser for Export {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn crate::export::ItemSink,
    ) -> Result<crate::export::ScanStats, crate::export::ScanError> {
        crate::export::scan_array_or_object(reader, |range, raw| {
            item.push(range, Vec::new(), None, raw.get().as_bytes())
        })
    }

    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        _: &mut dyn crate::export::ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let value = &value;
        conversation(value, sink);
        Ok(())
    }
}

static MESSAGE_FIELDS: Fields = Fields::new(
    "claude_message",
    &[
        ("content[].remaining", Rule::Read(T::Number)),
        ("attachments[].file_name", Rule::Read(T::String)),
        ("attachments[].file_type", Rule::Read(T::String)),
        ("attachments[].file_size", Rule::Read(T::Number)),
        ("attachments[].extracted_content", Rule::Read(T::String)),
        ("attachments[].file_uuid", Rule::Read(T::String)),
        ("attachments[].file_kind", Rule::Read(T::String)),
        ("files[].file_name", Rule::Read(T::String)),
        ("files[].file_type", Rule::Read(T::String)),
        ("files[].file_size", Rule::Read(T::Number)),
        ("files[].extracted_content", Rule::Read(T::String)),
        ("files[].file_uuid", Rule::Read(T::String)),
        ("files[].file_kind", Rule::Read(T::String)),
        ("files_v2[].file_name", Rule::Read(T::String)),
        ("files_v2[].file_type", Rule::Read(T::String)),
        ("files_v2[].file_size", Rule::Read(T::Number)),
        ("files_v2[].extracted_content", Rule::Read(T::String)),
        ("files_v2[].file_uuid", Rule::Read(T::String)),
        ("files_v2[].file_kind", Rule::Read(T::String)),
        ("content[].type", Rule::Read(T::String)),
        ("content[].text", Rule::Read(T::String)),
        ("content[].thinking", Rule::Read(T::String)),
        ("content[].signature", Rule::Read(T::String)),
        ("content[].thinking_hidden", Rule::Read(T::Boolean)),
        (
            "content[].summaries",
            Rule::ReadWhole("summaries in indexed block attributes"),
        ),
        (
            "content[].citations",
            Rule::ReadWhole("citation blocks and indexed attributes"),
        ),
        (
            "content[].start_timestamp",
            Rule::ReadWhole("indexed block time"),
        ),
        (
            "content[].stop_timestamp",
            Rule::ReadWhole("indexed block time"),
        ),
        ("content[].id", Rule::Read(T::String)),
        ("content[].name", Rule::Read(T::String)),
        (
            "content[].input",
            Rule::ReadWhole("tool input in call block"),
        ),
        (
            "content[].content",
            Rule::ReadWhole("complete structured tool content"),
        ),
        ("content[].tool_use_id", Rule::Read(T::String)),
        ("content[].is_error", Rule::Read(T::Boolean)),
        (
            "content[].structured_content",
            Rule::ReadWhole("structured tool result"),
        ),
        ("content[].source.type", Rule::Read(T::String)),
        ("content[].source.media_type", Rule::Read(T::String)),
        ("content[].source.data", Rule::Read(T::String)),
        ("content[].source.url", Rule::Read(T::String)),
        ("uuid", Rule::Read(T::String)),
        ("text", Rule::Read(T::String)),
        ("content", Rule::Read(T::StringOrArray)),
        ("sender", Rule::Read(T::String)),
        ("index", Rule::Read(T::Number)),
        ("created_at", Rule::Read(T::StringOrNumber)),
        (
            "updated_at",
            Rule::Excluded("edit time; the creation time orders messages"),
        ),
        (
            "edited_at",
            Rule::Excluded("edit time; the creation time orders messages"),
        ),
        // The model that wrote this message; it overrides the conversation's.
        ("model", Rule::Read(T::String)),
        (
            "metadata",
            Rule::Kept("message settings such as the thinking configuration"),
        ),
        (
            "truncated",
            Rule::Kept("the export marks a message it shortened"),
        ),
        ("attachments", Rule::Read(T::Array)),
        ("files", Rule::Read(T::Array)),
        ("files_v2", Rule::Read(T::Array)),
        (
            "sync_sources",
            Rule::Kept("connected drive and repository sources"),
        ),
        ("parent_message_uuid", Rule::Read(T::String)),
        ("stop_reason", Rule::Read(T::String)),
        ("input_mode", Rule::Kept("how the prompt was entered")),
        ("compaction_summary", Rule::Read(T::Array)),
        ("compaction_summary[].type", Rule::Read(T::String)),
        ("compaction_summary[].text", Rule::Read(T::String)),
        (
            "compaction_summary[].citations",
            Rule::ReadWhole("summary block citation metadata"),
        ),
    ],
)
.shape(
    &[],
    &[
        crate::fields::KnownKinds {
            path: "sender",
            values: &["human", "assistant"],
        },
        crate::fields::KnownKinds {
            path: "content[].type",
            values: &[
                "text",
                "voice_note",
                "thinking",
                "tool_use",
                "tool_result",
                "image",
                "document",
                "token_budget",
            ],
        },
    ],
);

static CONVERSATION_FIELDS: Fields = Fields::new(
    "claude_conversation",
    &[
        ("", Rule::Kept("conversation settings event")),
        ("uuid", Rule::Read(T::String)),
        ("name", Rule::Read(T::String)),
        ("created_at", Rule::Read(T::StringOrNumber)),
        ("updated_at", Rule::Read(T::StringOrNumber)),
        ("chat_messages", Rule::Read(T::Array)),
    ],
);

/// One claude.ai conversation object, shared with browser captures.
pub fn conversation(conv: &Json, sink: &mut Sink<'_>) {
    let Some(native) = conv.str("uuid").filter(|s| !s.is_empty()) else {
        sink.records
            .record("conversation", &RecordOutcome::unreadable("no uuid"));
        return;
    };
    let Some(sid) = session_id(ORIGIN, native) else {
        sink.records
            .record("conversation", &RecordOutcome::unreadable("unusable uuid"));
        return;
    };
    let Some(messages) = conv.array("chat_messages") else {
        sink.records.record(
            "conversation",
            &RecordOutcome::unreadable("no chat_messages"),
        );
        return;
    };
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    if let Some(name) = conv.str("name").filter(|s| !s.is_empty()) {
        session.offer_title(name, TitleSource::Generated);
    }
    if let Some(summary) = conv.str("summary").filter(|s| !s.is_empty()) {
        session.offer_title(summary, TitleSource::Summary);
    }
    if let Some(at) = conv.get("created_at").and_then(time_of) {
        session
            .offer_start(at.timestamp(), at.accuracy())
            .expect("decoded source time is valid");
    }
    sink.out.sessions.push(session);
    sink.records.record("conversation", &RecordOutcome::Used);

    // Account, project, settings and model are kept whole as one event.
    if let Some(obj) = conv.as_object() {
        let settings: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| k.as_str() != "chat_messages")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        sink.out.events.push({
            let (at, time_accuracy) = crate::time::parts(conv.get("updated_at").and_then(time_of));
            Event {
                session: Some(sid.clone()),
                kind: "conversation_settings".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: None,
                range: None,
                data: Json::Object(settings),
            }
        });
    }

    sink.records.fields(
        "conversation",
        &crate::fields::header(conv, &["chat_messages"]),
        &CONVERSATION_FIELDS,
    );
    let model = conv
        .str("model")
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let mut rows = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        let kind = m
            .str("sender")
            .filter(|s| !s.is_empty())
            .unwrap_or("?")
            .to_owned();
        sink.records.fields(&kind, m, &MESSAGE_FIELDS);
        match message(&sid, native, i, m, model.as_deref()) {
            Ok(msg) => {
                // Summaries are readable messages; the boundary names that
                // message and preserves the provider's replacement context.
                if let Some(summary) = m
                    .array("compaction_summary")
                    .filter(|parts| !parts.is_empty())
                {
                    let mut blocks: Vec<_> = summary
                        .iter()
                        .filter_map(|part| {
                            part.str("text")
                                .filter(|s| !s.is_empty())
                                .map(|text| Block::Text {
                                    text: text.to_owned(),
                                })
                        })
                        .collect();
                    let summary_id = (!blocks.is_empty())
                        .then(|| {
                            crate::content_ids::named_blocks(
                                &sid,
                                Role::User,
                                &mut blocks,
                                i as u64,
                            )
                        })
                        .flatten();
                    if let Some(id) = summary_id {
                        let mut summary_message = context_summary(
                            sid.clone(),
                            id.clone(),
                            msg.parent.clone(),
                            msg.at,
                            msg.time_accuracy(),
                            blocks,
                        );
                        let details = content_details(summary);
                        if !details.is_empty() {
                            summary_message
                                .attrs
                                .insert("content".into(), Json::Array(details));
                        }
                        let index = m.i64("index").unwrap_or(i as i64);
                        rows.push((
                            msg.at.map(|at| at.unix_ms()),
                            index,
                            i,
                            0,
                            polylogue_model::IdKind::Content,
                            summary_message,
                        ));
                        let time_accuracy = msg.time_accuracy();
                        sink.out.events.push(Event {
                            session: Some(sid.clone()),
                            kind: "compaction".into(),
                            at: msg.at,
                            at_quality: time_accuracy.quality,
                            at_precision: time_accuracy.precision,
                            at_until: time_accuracy.until,
                            after: msg.parent.clone(),
                            range: None,
                            data: serde_json::to_value(polylogue_model::CompactionBoundary {
                                summary: Some(id),
                                ..Default::default()
                            })
                            .expect("boundary is JSON"),
                        });
                    } else {
                        let accuracy = msg.time_accuracy();
                        sink.out.events.push(Event {
                            session: Some(sid.clone()),
                            kind: "compaction".into(),
                            at: msg.at,
                            at_quality: accuracy.quality,
                            at_precision: accuracy.precision,
                            at_until: accuracy.until,
                            after: msg.parent.clone(),
                            range: None,
                            data: serde_json::to_value(
                                polylogue_model::CompactionBoundary::default(),
                            )
                            .expect("boundary is JSON"),
                        });
                    }
                }
                collect_inline(m, &mut sink.out.blobs);
                let at = msg.at.map(|t| t.unix_ms());
                let index = m.i64("index").unwrap_or(i as i64);
                rows.push((at, index, i, 1, polylogue_model::IdKind::Native, msg));
                sink.records.record(&kind, &RecordOutcome::Used);
            }
            Err(d) => sink.records.record(&kind, &d),
        }
    }
    // A message without a time stays right after the message before it
    // (docs/sessions.md, Order), so it takes that message's time as its key.
    let mut last = None;
    for row in &mut rows {
        if row.0.is_some() {
            last = row.0;
        } else {
            row.0 = last;
        }
    }
    rows.sort_by_key(|r| (r.0, r.1, r.2, r.3));
    for (_, _, _, _, id_kind, message) in rows {
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            response: None,
            page_text: false,
            id_kind,
            offset,
            end: offset + 1,
            message,
        });
    }
    if let Some(leaf) = conv
        .str("current_leaf_message_uuid")
        .filter(|s| !s.is_empty())
        && let Some(id) = message_id(ORIGIN, leaf)
    {
        sink.out.events.push({
            let (at, time_accuracy) = crate::time::parts(None);
            Event {
                session: Some(sid),
                kind: "active_leaf".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: Some(id),
                range: None,
                data: Json::Null,
            }
        });
    }
}

fn message(
    sid: &SessionId,
    conversation: &str,
    position: usize,
    m: &Json,
    model: Option<&str>,
) -> Result<Message, RecordOutcome> {
    // A message the export wrote without a uuid keeps its identity by its
    // place in the conversation: the same place is the
    // same message, and changed text there is a new version of it.
    let native = match m.str("uuid").filter(|s| !s.is_empty()) {
        Some(u) => u.to_owned(),
        None => {
            let place = m.u64("index").unwrap_or(position as u64);
            format!("{conversation}#{place}")
        }
    };
    let id =
        message_id(ORIGIN, &native).ok_or_else(|| RecordOutcome::unreadable("unusable uuid"))?;
    let (role, prompt_kind, direction) = match m.str("sender").filter(|s| !s.is_empty()) {
        Some("human") => (
            Role::User,
            Some(PromptKind::Prose),
            AttachmentDirection::Input,
        ),
        Some("assistant") => (Role::Assistant, None, AttachmentDirection::Output),
        Some(other) => return Err(RecordOutcome::skipped(format!("sender {other}"))),
        None => return Err(RecordOutcome::unreadable("no sender")),
    };
    let mut blocks = Vec::new();
    let content = m.array("content");
    for (position, b) in content.into_iter().flatten().enumerate() {
        if let Some(block) = content_block(b, direction) {
            blocks.push(block);
        }
        if let Some(citations) = b
            .get("citations")
            .filter(|value| !value.is_null() && !value.as_array().is_some_and(Vec::is_empty))
        {
            blocks.push(Block::Other {
                type_name: "citations".into(),
                data: serde_json::json!({"block":position,"citations":citations}),
            });
        }
    }
    // Older exports carry only the flat text.
    if content.is_none_or(|c| c.is_empty())
        && let Some(text) = m.str("text").filter(|s| !s.is_empty()).map(str::to_owned)
    {
        blocks.push(Block::Text { text });
    }
    blocks.extend(attachments(m, direction));
    let parent = m
        .str("parent_message_uuid")
        .filter(|s| !s.is_empty())
        .filter(|p| *p != ROOT_PARENT)
        .and_then(|p| message_id(ORIGIN, p));
    Ok({
        let (at, time_accuracy) = crate::time::parts(m.get("created_at").and_then(time_of));
        Message {
            id,
            session: sid.clone(),
            parent,
            role,
            at,
            at_quality: time_accuracy.quality,
            at_precision: time_accuracy.precision,
            at_until: time_accuracy.until,
            model: (role == Role::Assistant)
                .then(|| {
                    m.str("model")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .or_else(|| model.map(str::to_owned))
                })
                .flatten(),
            prompt_kind,
            blocks,
            usage: Default::default(),
            stop_reason: m
                .str("stop_reason")
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            reported_cost_usd: None,
            duration_ms: None,
            attrs: content_attrs(m),
            production: Default::default(),
        }
    })
}

fn content_attrs(message: &Json) -> serde_json::Map<String, Json> {
    let mut attrs = crate::fields::attributes(message, &MESSAGE_FIELDS);
    if let Some(value) = message.get("metadata") {
        attrs.insert("metadata".into(), value.clone());
    }
    let details = message
        .array("content")
        .map(|parts| content_details(parts))
        .unwrap_or_default();
    if !details.is_empty() {
        attrs.insert("content".into(), Json::Array(details));
    }
    attrs
}

fn content_details(parts: &[Json]) -> Vec<Json> {
    parts
        .iter()
        .enumerate()
        .filter_map(|(position, block)| {
            let fields: serde_json::Map<String, Json> = block
                .as_object()?
                .iter()
                .filter(|(key, _)| {
                    !matches!(
                        key.as_str(),
                        "type" | "text" | "thinking" | "signature" | "input" | "content" | "source"
                    )
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            (!fields.is_empty()).then(|| serde_json::json!({"position":position,"fields":fields}))
        })
        .collect()
}

/// One content block. Provider-only kinds remain structured index blocks.
fn content_block(b: &Json, direction: AttachmentDirection) -> Option<Block> {
    let kind = b.str("type").filter(|s| !s.is_empty()).unwrap_or("");
    Some(match kind {
        "text" => Block::Text {
            text: b.str("text").filter(|s| !s.is_empty()).map(str::to_owned)?,
        },
        "voice_note" => Block::Text {
            text: b.str("text").filter(|s| !s.is_empty()).map(str::to_owned)?,
        },
        "thinking" => {
            let text = b
                .str("thinking")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    // When the reasoning itself is withheld, its summaries remain.
                    let summaries: Vec<&str> = b
                        .array("summaries")
                        .into_iter()
                        .flatten()
                        .filter_map(|s| {
                            s.str("summary")
                                .filter(|s| !s.is_empty())
                                .or_else(|| s.str("text").filter(|s| !s.is_empty()))
                        })
                        .collect();
                    (!summaries.is_empty()).then(|| summaries.join("\n"))
                });
            Block::Thinking {
                text,
                signature: b
                    .str("signature")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                redacted: b.bool("thinking_hidden") == Some(true),
            }
        }
        "tool_use" => Block::ToolCall {
            written_text: None,
            run_by_script: false,
            call_id: b.str("id").filter(|s| !s.is_empty()).map(str::to_owned),
            tool: b.str("name").filter(|s| !s.is_empty()).map(str::to_owned),
            action: action(b.str("name").filter(|s| !s.is_empty()).unwrap_or("")),
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

        other => Block::Other {
            type_name: other.to_owned(),
            data: b.clone(),
        },
    })
}

fn inline_bytes(block: &Json) -> Option<Vec<u8>> {
    if !matches!(
        block.str("type").filter(|s| !s.is_empty()),
        Some("image" | "document")
    ) || block.pointer("/source/type").and_then(Json::as_str) != Some("base64")
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
        name: block
            .str("name")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                block
                    .str("title")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
            }),
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

fn collect_inline(value: &Json, blobs: &mut Vec<ExtractedBlob>) {
    if let Some(bytes) = inline_bytes(value) {
        let hash = BlobHash::of(&bytes);
        if !blobs.iter().any(|blob| blob.hash == hash) {
            blobs.push(ExtractedBlob {
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
                match item.str("type").filter(|s| !s.is_empty()) {
                    Some("text") => {
                        if let Some(t) = item
                            .str("text")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                        {
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
    let error = b.bool("is_error") == Some(true);
    let mut structured = b
        .get("structured_content")
        .filter(|v| !v.is_null())
        .cloned();
    if let Some(items) = b.array("content") {
        structured = Some(serde_json::json!({"structured_content":structured,"content":items}));
    }

    Block::ToolResult {
        reported_duration_ms: None,
        call_id: b
            .str("tool_use_id")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
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
    for a in m.array("attachments").into_iter().flatten() {
        out.push(Block::File(Attachment {
            name: a
                .str("file_name")
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            media_type: a
                .str("file_type")
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            size: a.u64("file_size"),
            direction,
            ..Attachment::default()
        }));
        if let Some(text) = a
            .str("extracted_content")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            // The text claude.ai read out of the file and gave the model.
            out.push(Block::Other {
                type_name: "attachment_text".into(),
                data: serde_json::json!({"file_name": a.get("file_name"), "text": text}),
            });
        }
    }
    for key in ["files", "files_v2"] {
        for f in m.array(key).into_iter().flatten() {
            let image = f.str("file_kind").filter(|s| !s.is_empty()) == Some("image")
                || f.str("file_type")
                    .filter(|s| !s.is_empty())
                    .is_some_and(|t| t.starts_with("image/"));
            let attachment = Attachment {
                name: f
                    .str("file_name")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                media_type: f
                    .str("file_type")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                source_id: f
                    .str("file_uuid")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
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
                    .record("chat", &RecordOutcome::unreadable("not an object"));
            }
            Err(e) => {
                return sink.records.record(
                    "chat",
                    &RecordOutcome::unreadable(format!("malformed JSON: {e}")),
                );
            }
        };
        design_chat(&chat, &mut sink);
    }
}

fn design_chat(chat: &Json, sink: &mut Sink<'_>) {
    let Some(sid) = chat
        .str("uuid")
        .filter(|s| !s.is_empty())
        .and_then(|u| session_id(DESIGN_ORIGIN, u))
    else {
        sink.records
            .record("chat", &RecordOutcome::unreadable("no uuid"));
        return;
    };
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    if let Some(title) = chat.str("title").filter(|s| !s.is_empty()) {
        session.offer_title(title, TitleSource::Generated);
    }
    if let Some(at) = chat.get("created_at").and_then(time_of) {
        session
            .offer_start(at.timestamp(), at.accuracy())
            .expect("decoded source time is valid");
    }
    sink.out.sessions.push(session);
    sink.records.record("chat", &RecordOutcome::Used);
    if let Some(project) = chat.get("project").filter(|p| !p.is_null()) {
        sink.out.events.push({
            let (at, time_accuracy) = crate::time::parts(None);
            Event {
                session: Some(sid.clone()),
                kind: "design_project".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: None,
                range: None,
                data: project.clone(),
            }
        });
    }
    let mut previous: Option<polylogue_model::MessageId> = None;
    for m in chat.array("messages").into_iter().flatten() {
        let role_name = m
            .str("role")
            .filter(|s| !s.is_empty())
            .unwrap_or("?")
            .to_owned();
        let Some(id) = m
            .str("uuid")
            .filter(|s| !s.is_empty())
            .and_then(|u| message_id(DESIGN_ORIGIN, u))
        else {
            sink.records
                .record(&role_name, &RecordOutcome::unreadable("no uuid"));
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
                    .record(other, &RecordOutcome::skipped(format!("role {other}")));
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
        collect_inline(m, &mut sink.out.blobs);
        for a in body.array("attachments").into_iter().flatten() {
            blocks.push(Block::File(Attachment {
                name: a
                    .str("name")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        a.str("file_name")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                    }),
                media_type: a
                    .str("type")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .or_else(|| {
                        a.str("mime_type")
                            .filter(|s| !s.is_empty())
                            .map(str::to_owned)
                    }),
                direction,
                ..Attachment::default()
            }));
        }
        let message = {
            let (at, time_accuracy) = crate::time::parts(
                m.get("created_at")
                    .and_then(time_of)
                    .or_else(|| body.get("timestamp").and_then(time_of)),
            );
            Message {
                id: id.clone(),
                session: sid.clone(),
                parent: previous.replace(id),
                role,
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                model: None,
                prompt_kind,
                blocks,
                usage: Default::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                attrs: Default::default(),
                production: Default::default(),
            }
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            response: None,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            offset,
            end: offset + 1,
            message,
        });
        sink.records.record(&role_name, &RecordOutcome::Used);
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod sidecars_tests;

/// A provider's context summary is harness material, not another model turn.
fn context_summary(
    session: SessionId,
    id: MessageId,
    parent: Option<MessageId>,
    at: Option<polylogue_model::Timestamp>,
    accuracy: polylogue_model::TimeAccuracy,
    blocks: Vec<polylogue_model::Block>,
) -> polylogue_model::Message {
    polylogue_model::Message {
        id,
        session,
        parent,
        role: polylogue_model::Role::User,
        at,
        at_quality: accuracy.quality,
        at_precision: accuracy.precision,
        at_until: accuracy.until,
        model: None,
        prompt_kind: Some(polylogue_model::PromptKind::CompactionSummary),
        blocks,
        usage: Default::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        attrs: Default::default(),
        production: Default::default(),
    }
}
