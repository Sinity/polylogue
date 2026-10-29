//! The browser extension's capture spool: one capture log per conversation
//! (`$XDG_STATE_HOME/polylogue/browser-capture/<provider>/*.json`). Each distinct
//! capture is parsed once. A file without the capture-log header is read as
//! one capture.
//!
//! A capture carries the provider's own conversation JSON
//! (`raw_provider_payload`) where the provider page exposes it. That JSON
//! is read by the provider's export parser, so a captured conversation and
//! the same conversation in an export get the same IDs and unite. Where
//! there is no provider JSON, the extension's own normalized turns are read
//! instead, under the provider's origin.
//! Attachment bytes the extension inlined are stored and linked to their
//! attachment; a file whose bytes an earlier capture already stored is named
//! by its blob hash instead.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{DocumentParser, ExtractedBlob, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Message, PromptKind, Role,
    Session, SessionKind, TitleSource,
};
use std::path::Path;

pub const ORIGIN: &str = "browser-capture";

pub const LOCATIONS: &[&str] = &["polylogue/browser-capture/*/*.json"];

const KIND_KEY: &str = "polylogue_capture_kind";

/// Recognizes capture logs and single captures, including ones that start
/// with `schema_version`, without treating an unrelated schema-versioned
/// JSON file as one.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension().is_some_and(|e| e == "json")
        && (head.starts_with(polylogue_model::capture_log::MAGIC)
            || (first_byte(head) == Some(b'{')
                && (matches!(first_key(head), Some(KIND_KEY | "capture_id"))
                    || (first_key(head) == Some("schema_version")
                        && head
                            .windows(b"\"polylogue_capture_kind\"".len())
                            .any(|part| part == b"\"polylogue_capture_kind\"")))))
}

pub fn open(path: &Path) -> Option<Parse> {
    path.extension()
        .is_some_and(|e| e == "json")
        .then(|| Parse::CaptureLog(Box::new(Capture)))
}

static CAPTURE_FIELDS: Fields = Fields::new(&[
    (KIND_KEY, Rule::Read(T::String)),
    ("schema_version", Rule::Read(T::Number)),
    (
        "capture_id",
        Rule::Excluded("the extension's own key; the provider IDs identify the conversation"),
    ),
    (
        "source",
        Rule::Excluded("names the capturing program, always the extension"),
    ),
    (
        "provenance",
        Rule::Kept("complete browser capture provenance event"),
    ),
    ("provenance.captured_at", Rule::Read(T::String)),
    (
        "provider_meta",
        Rule::Kept("complete browser capture provider metadata"),
    ),
    ("raw_provider_payload", Rule::Read(T::Object)),
    ("session", Rule::Read(T::Object)),
]);

static NATIVE_SESSION_FIELDS: Fields = Fields::new(&[
    ("provider", Rule::Read(T::String)),
    ("attachments", Rule::Read(T::Array)),
]);
static TURN_SESSION_FIELDS: Fields = Fields::new(&[
    ("provider", Rule::Read(T::String)),
    ("provider_session_id", Rule::Read(T::String)),
    ("title", Rule::Read(T::String)),
    ("turns", Rule::Read(T::Array)),
    ("attachments", Rule::Read(T::Array)),
]);
static ATTACHMENT_FIELDS: Fields = Fields::new(&[
    ("content_base64", Rule::Read(T::String)),
    ("blob", Rule::Read(T::String)),
    ("size_bytes", Rule::Read(T::Number)),
    ("provider_attachment_id", Rule::Read(T::String)),
    ("message_provider_id", Rule::Read(T::String)),
    ("name", Rule::Read(T::String)),
    ("mime_type", Rule::Read(T::String)),
    ("url", Rule::Read(T::String)),
    ("extracted_content", Rule::Read(T::String)),
    ("not_fetched", Rule::Read(T::Any)),
]);

/// Fields of the extension's normalized turns. Only read when the page
/// exposed no provider JSON.
static TURN_FIELDS: Fields = Fields::new(&[
    ("provider_turn_id", Rule::Read(T::String)),
    ("role", Rule::Read(T::String)),
    ("text", Rule::Read(T::String)),
    ("timestamp", Rule::Read(T::StringOrNumber)),
    ("ordinal", Rule::Read(T::Number)),
    ("parent_turn_id", Rule::Read(T::String)),
    ("model", Rule::Read(T::String)),
    ("blocks", Rule::Read(T::Array)),
    ("attachments", Rule::Read(T::Array)),
    (
        "provider_meta",
        Rule::Kept("the adapter's page observations for the turn; no reader yet"),
    ),
    (
        "identity_observation",
        Rule::Kept("how the adapter matched the turn to the page; no reader yet"),
    ),
]);

/// The origin a capture's `session.provider` names: the provider's export
/// origin, so a capture and an export of one conversation unite.
pub fn origin_of(provider: &str) -> &str {
    crate::origin::by_extension_name(provider).map_or(provider, |o| o.name)
}

/// The session a capture of `provider_session_id` from `provider` lands in.
pub fn session_for(
    provider: &str,
    provider_session_id: &str,
) -> Option<polylogue_model::SessionId> {
    session_id(origin_of(provider), provider_session_id)
}

mod files;
mod grok;
pub use files::files;

pub struct Capture;

impl DocumentParser for Capture {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records) {
        let mut sink = Sink::new(out, records);
        let capture = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                return sink
                    .records
                    .record("capture", &RecordOutcome::unreadable("not an object"));
            }
            Err(e) => {
                return sink.records.record(
                    "capture",
                    &RecordOutcome::unreadable(format!("malformed JSON: {e}")),
                );
            }
        };
        // `source` names the capturing program (`browser-extension`); the
        // provider whose conversation it is sits in the session.
        let Some(provider) = capture
            .pointer("/session/provider")
            .and_then(Json::as_str)
            .filter(|p| !p.is_empty())
        else {
            return sink
                .records
                .record("capture", &RecordOutcome::unreadable("no session.provider"));
        };
        let origin = origin_of(provider);
        if polylogue_model::Origin::new(origin).is_err() {
            return sink.records.record(
                "capture",
                &RecordOutcome::unreadable(format!("source {origin:?} is not an origin name")),
            );
        }
        sink.records.record("capture", &RecordOutcome::Used);

        let raw = capture
            .get("raw_provider_payload")
            .filter(|r| r.as_object().is_some_and(|o| !o.is_empty()));
        let native_dispatch = match (origin, raw) {
            ("chatgpt", Some(raw)) => raw.get("mapping").is_some(),
            ("claude-ai", Some(raw)) => raw.get("chat_messages").is_some(),
            ("grok", Some(raw)) => raw.get("responses").is_some(),
            ("codex-cloud", Some(raw)) => crate::codex_cloud::task_shape(raw),
            _ => false,
        };
        let header = crate::fields::header(
            &capture,
            if native_dispatch {
                &["session", "raw_provider_payload"]
            } else {
                &["session"]
            },
        );
        sink.records.fields("capture", &header, &CAPTURE_FIELDS);
        if let Some(session) = capture.get("session") {
            let header = crate::fields::header(
                session,
                if native_dispatch {
                    &["attachments"]
                } else {
                    &["turns", "attachments"]
                },
            );
            sink.records.fields(
                "session",
                &header,
                if native_dispatch {
                    &NATIVE_SESSION_FIELDS
                } else {
                    &TURN_SESSION_FIELDS
                },
            );
        }
        let first_message = sink.out.messages.len();
        match (origin, raw) {
            ("codex-cloud", Some(raw)) if crate::codex_cloud::task_shape(raw) => {
                let result = serde_json::to_vec(raw)
                    .map_err(|error| error.to_string())
                    .and_then(|bytes| {
                        crate::export::ExportParser::item(
                            &mut crate::codex_cloud::Cloud,
                            &bytes,
                            &mut sink,
                            &mut crate::export::MemoryState::default(),
                        )
                    });
                if let Err(error) = result {
                    sink.records
                        .record("task", &RecordOutcome::unreadable(error));
                }
            }
            ("chatgpt", Some(raw)) if raw.get("mapping").is_some() => {
                crate::chatgpt::conversation(raw, &mut sink, &mut Default::default());
            }
            ("claude-ai", Some(raw)) if raw.get("chat_messages").is_some() => {
                crate::claude_ai::conversation(raw, &mut sink);
            }
            ("grok", Some(raw)) if raw.get("responses").is_some() => {
                grok::conversation(
                    raw,
                    capture.get("session").unwrap_or(&Json::Null),
                    &mut sink,
                );
            }
            _ => match capture.get("session").filter(|s| s.is_object()) {
                Some(session) => turns(origin, session, &mut sink),
                None => sink.records.record(
                    "session",
                    &RecordOutcome::unreadable("neither provider JSON nor turns"),
                ),
            },
        }
        let session = sink.out.sessions.last().map(|s| s.id.clone()).or_else(|| {
            let native = capture.pointer("/session/provider_session_id")?.as_str()?;
            session_id(origin, native)
        });
        if let Some(session) = session {
            let provenance = capture.get("provenance").cloned().unwrap_or(Json::Null);
            sink.out.events.push(Event {
                session: Some(session),
                kind: "browser_capture".into(),
                at: provenance.get("captured_at").and_then(time_of),
                after: None,
                range: None,
                data: serde_json::json!({
                    "schema_version": capture.get("schema_version"),
                    "provenance": provenance,
                    "provider_meta": capture.get("provider_meta"),
                    "attachments": capture.pointer("/session/attachments").and_then(Json::as_array)
                        .into_iter().flatten().map(|a| serde_json::json!({
                            "id": a.get("provider_attachment_id"), "not_fetched": a.get("not_fetched")
                        })).collect::<Vec<_>>(),
                }),
            });
        }
        attach_files(&capture, origin, first_message, &mut sink);
    }
}

/// The extension's normalized turns, for a provider page that exposed no
/// conversation JSON.
fn turns(origin: &str, session: &Json, sink: &mut Sink<'_>) {
    let Some(native) = str_of(session, "provider_session_id") else {
        sink.records.record(
            "session",
            &RecordOutcome::unreadable("no provider_session_id"),
        );
        return;
    };
    let Some(sid) = session_id(origin, native) else {
        sink.records
            .record("session", &RecordOutcome::unreadable("unusable session ID"));
        return;
    };
    let mut facts = Session::new(sid.clone(), SessionKind::Main);
    if let Some(title) = str_of(session, "title") {
        facts.offer_title(title, TitleSource::Generated);
    }
    if let Some(at) = session.get("created_at").and_then(time_of) {
        facts.offer_start(at);
    }
    sink.out.sessions.push(facts);
    sink.records.record("session", &RecordOutcome::Used);
    let mut list: Vec<&Json> = session
        .get("turns")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .collect();
    list.sort_by_key(|t| t.get("ordinal").and_then(Json::as_i64).unwrap_or(i64::MAX));
    for (i, turn) in list.into_iter().enumerate() {
        let role_name = str_of(turn, "role").unwrap_or("?").to_owned();
        sink.records.fields("turn", turn, &TURN_FIELDS);
        let (role, prompt_kind, direction) = match role_name.as_str() {
            "user" | "human" => (
                Role::User,
                Some(PromptKind::Prose),
                AttachmentDirection::Input,
            ),
            "assistant" | "model" => (Role::Assistant, None, AttachmentDirection::Output),
            "system" => (Role::System, None, AttachmentDirection::Input),
            other => {
                sink.records
                    .record(other, &RecordOutcome::skipped(format!("role {other}")));
                continue;
            }
        };
        // A turn the page shows without an ID keeps its identity by its
        // place in the conversation.
        let turn_native = string_of(turn, "provider_turn_id").unwrap_or_else(|| {
            let place = turn
                .get("ordinal")
                .and_then(Json::as_u64)
                .unwrap_or(i as u64);
            format!("{native}#{place}")
        });
        let Some(id) = message_id(origin, &turn_native) else {
            sink.records
                .record(&role_name, &RecordOutcome::unreadable("unusable turn ID"));
            continue;
        };
        let mut blocks: Vec<Block> = turn
            .get("blocks")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
            .filter_map(|b| turn_block(b, direction))
            .collect();
        if blocks.is_empty()
            && let Some(text) = string_of(turn, "text")
        {
            blocks.push(Block::Text { text });
        }
        for a in turn
            .get("attachments")
            .and_then(Json::as_array)
            .into_iter()
            .flatten()
        {
            blocks.extend(attachment(a, direction, None));
        }
        let message = Message {
            id,
            session: sid.clone(),
            parent: str_of(turn, "parent_turn_id").and_then(|p| message_id(origin, p)),
            role,
            at: turn.get("timestamp").and_then(time_of),
            model: string_of(turn, "model"),
            prompt_kind,
            blocks,
            usage: Default::default(),
            stop_reason: None,
            reported_cost_usd: None,
            duration_ms: None,
            attrs: crate::fields::attributes(turn, &TURN_FIELDS),
            production: Default::default(),
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
        sink.records.record(&role_name, &RecordOutcome::Used);
    }
}

/// One block of a normalized turn.
fn turn_block(b: &Json, direction: AttachmentDirection) -> Option<Block> {
    let kind = str_of(b, "type").unwrap_or("");
    Some(match kind {
        "text" | "markdown" => Block::Text {
            text: string_of(b, "text")?,
        },
        "thinking" | "reasoning" => Block::Thinking {
            text: string_of(b, "text"),
            signature: None,
            redacted: false,
        },
        "image" => Block::Image(Attachment {
            name: string_of(b, "name"),
            media_type: string_of(b, "mime_type"),
            url: string_of(b, "url"),
            direction,
            ..Attachment::default()
        }),
        _ => Block::Other {
            type_name: kind.to_owned(),
            data: b.clone(),
        },
    })
}

/// An attachment the extension observed, as the block holding it, and the
/// text the page extracted from it when it shows one.
fn attachment(
    a: &Json,
    direction: AttachmentDirection,
    stored: Option<(BlobHash, Option<u64>)>,
) -> Vec<Block> {
    let media_type = string_of(a, "mime_type");
    let file = Attachment {
        name: string_of(a, "name"),
        size: stored
            .and_then(|(_, size)| size)
            .or_else(|| a.get("size_bytes").and_then(Json::as_u64)),
        blob: stored.map(|(hash, _)| hash),
        url: string_of(a, "url"),
        source_id: string_of(a, "provider_attachment_id"),
        direction,
        media_type,
    };
    let mut out = vec![if file
        .media_type
        .as_deref()
        .is_some_and(|m| m.starts_with("image/"))
    {
        Block::Image(file)
    } else {
        Block::File(file)
    }];
    if let Some(text) = string_of(a, "extracted_content") {
        out.push(Block::Other {
            type_name: "attachment_text".into(),
            data: serde_json::json!({ "file_name": str_of(a, "name"), "text": text }),
        });
    }
    out
}

/// Links the session's attachment list to its messages: stores the bytes
/// the extension inlined (`content_base64`), or takes the hash of bytes an
/// earlier capture stored (`blob`), and points the matching attachment at
/// them, the message by its provider ID, the attachment by its file ID or,
/// failing that, its name. An attachment no block of its message holds yet
/// is added to that message, with or without bytes.
fn attach_files(capture: &Json, origin: &str, first_message: usize, sink: &mut Sink<'_>) {
    for a in capture
        .pointer("/session/attachments")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        sink.records.fields("attachment", a, &ATTACHMENT_FIELDS);
        let stored = a
            .get("content_base64")
            .and_then(Json::as_str)
            .and_then(|data| base64::engine::general_purpose::STANDARD.decode(data).ok())
            .map(|bytes| {
                let hash = BlobHash::of(&bytes);
                let size = bytes.len() as u64;
                sink.out.blobs.push(ExtractedBlob { hash, size, bytes });
                (hash, Some(size))
            })
            .or_else(|| {
                let hash = BlobHash::parse(str_of(a, "blob")?).ok()?;
                Some((hash, a.get("size_bytes").and_then(Json::as_u64)))
            });
        let file_id = str_of(a, "provider_attachment_id");
        let name = str_of(a, "name");
        let Some(msg_id) = str_of(a, "message_provider_id").and_then(|m| message_id(origin, m))
        else {
            continue;
        };
        let Some(row) = sink.out.messages[first_message..]
            .iter_mut()
            .find(|r| r.message.id == msg_id)
        else {
            continue;
        };
        let mut matched = false;
        for block in &mut row.message.blocks {
            let attachments: Vec<&mut Attachment> = match block {
                Block::Image(x) | Block::File(x) => vec![x],
                Block::ToolResult { attachments, .. } => attachments.iter_mut().collect(),
                _ => continue,
            };
            for x in attachments {
                let same = match (x.source_id.as_deref(), file_id) {
                    (Some(s), Some(f)) => s == f || f.ends_with(s) || s.ends_with(f),
                    _ => name.is_some() && x.name.as_deref() == name,
                };
                if same {
                    matched = true;
                    if let Some((hash, size)) = stored
                        && x.blob.is_none()
                    {
                        x.blob = Some(hash);
                        x.size = size.or(x.size);
                    }
                }
            }
        }
        if !matched {
            let direction = if row.message.role == Role::User {
                AttachmentDirection::Input
            } else {
                AttachmentDirection::Output
            };
            row.message.blocks.extend(attachment(a, direction, stored));
        }
    }
}

#[cfg(test)]
mod tests;
