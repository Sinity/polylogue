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

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 6;

parser_modules! {
    mod files;
    mod grok;
}

use crate::document::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, time_of};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{DocumentParser, MessageRow, Output, Parse};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Attachment, AttachmentDirection, AttachmentFailure, BlobHash, Block, Event, IdKind, Json,
    Message, PromptKind, Role, Session, SessionKind, TitleSource,
};
use std::path::Path;

pub const ORIGIN: &str = "browser-capture";

pub const LOCATIONS: &[&str] = &["polylogue/browser-capture/*/*.json"];

const KIND_KEY: &str = "polylogue_capture_kind";

/// Recognizes capture logs and single captures, including ones that start
/// with `schema_version`, without treating an unrelated schema-versioned
/// JSON file as one.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    let head = head.bytes;
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

static CAPTURE_FIELDS: Fields = Fields::new(
    "capture",
    &[
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
    ],
);

static NATIVE_SESSION_FIELDS: Fields = Fields::new(
    "native_session",
    &[
        ("provider", Rule::Read(T::String)),
        ("session_kind", Rule::Read(T::String)),
        ("attachments", Rule::Read(T::Array)),
    ],
);
static TURN_SESSION_FIELDS: Fields = Fields::new(
    "turn_session",
    &[
        ("provider", Rule::Read(T::String)),
        ("session_kind", Rule::Read(T::String)),
        ("provider_session_id", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("turns", Rule::Read(T::Array)),
        ("attachments", Rule::Read(T::Array)),
    ],
);
static ATTACHMENT_FIELDS: Fields = Fields::new(
    "attachment",
    &[
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
    ],
);

/// Fields of the extension's normalized turns. Only read when the page
/// exposed no provider JSON.
static TURN_FIELDS: Fields = Fields::new(
    "turn",
    &[
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
    ],
);

static TURN_BLOCK_FIELDS: Fields = Fields::new(
    "block",
    &[
        ("type", Rule::Read(T::String)),
        ("text", Rule::Read(T::String)),
        ("name", Rule::Read(T::String)),
        ("mime_type", Rule::Read(T::String)),
        ("url", Rule::Read(T::String)),
    ],
);
static OTHER_BLOCK_FIELDS: Fields = Fields::new(
    "other_block",
    &[(
        "",
        Rule::Kept("the normalized block is kept whole as Other data"),
    )],
);

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

pub use files::files;

pub struct Capture;

/// The provider's own conversation JSON inside a capture, when the payload
/// has that provider's shape.
#[derive(Clone, Copy)]
enum NativeShape {
    CodexCloud,
    ChatGpt,
    ClaudeAi,
    Grok,
}

impl NativeShape {
    fn of(origin: &str, raw: &Json) -> Option<Self> {
        match origin {
            "chatgpt" => raw.get("mapping").is_some().then_some(Self::ChatGpt),
            "claude-ai" => raw.get("chat_messages").is_some().then_some(Self::ClaudeAi),
            "grok" => raw.get("responses").is_some().then_some(Self::Grok),
            "codex-cloud" => crate::codex_cloud::task_shape(raw).then_some(Self::CodexCloud),
            _ => None,
        }
    }
}

/// The session mode of a provider's temporary chat, which the provider does
/// not keep: the capture may be its only copy.
pub const TEMPORARY_MODE: &str = "temporary";

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
        let native = raw.and_then(|raw| NativeShape::of(origin, raw));
        let provider_format = native.is_some();
        let header = crate::fields::header(
            &capture,
            if provider_format {
                &["session", "raw_provider_payload"]
            } else {
                &["session"]
            },
        );
        sink.records.fields("capture", &header, &CAPTURE_FIELDS);
        if let Some(session) = capture.get("session") {
            let header = crate::fields::header(
                session,
                if provider_format {
                    &["attachments"]
                } else {
                    &["turns", "attachments"]
                },
            );
            sink.records.fields(
                "session",
                &header,
                if provider_format {
                    &NATIVE_SESSION_FIELDS
                } else {
                    &TURN_SESSION_FIELDS
                },
            );
        }
        let first_message = sink.out.messages.len();
        let first_session = sink.out.sessions.len();
        match (native, raw) {
            (Some(NativeShape::CodexCloud), Some(raw)) => {
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
            (Some(NativeShape::ChatGpt), Some(raw)) => {
                crate::chatgpt::conversation(raw, &mut sink, &mut Default::default());
            }
            (Some(NativeShape::ClaudeAi), Some(raw)) => {
                crate::claude_ai::conversation(raw, &mut sink);
            }
            (Some(NativeShape::Grok), Some(raw)) => {
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
        // A temporary chat is one the provider does not keep.
        if capture
            .pointer("/session/session_kind")
            .and_then(Json::as_str)
            == Some("temporary")
        {
            for row in &mut sink.out.sessions[first_session..] {
                row.mode = Some(TEMPORARY_MODE.to_owned());
            }
        }
        if let Some(session) = session {
            let provenance = capture.get("provenance").cloned().unwrap_or(Json::Null);
            sink.out.events.push({ let (at,time_accuracy)=crate::time::parts(provenance.get("captured_at").and_then(time_of).map(crate::time::SourceTime::captured)); Event {
                session: Some(session),
                kind: "browser_capture".into(),
 at, at_quality:time_accuracy.quality, at_precision:time_accuracy.precision, at_until:time_accuracy.until,
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
            } });
        }
        attach_files(&capture, origin, first_message, &mut sink);
    }
}

/// The extension's normalized turns, for a provider page that exposed no
/// conversation JSON.
fn turns(origin: &str, session: &Json, sink: &mut Sink<'_>) {
    let Some(native) = session.str("provider_session_id").filter(|s| !s.is_empty()) else {
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
    if let Some(title) = session.str("title").filter(|s| !s.is_empty()) {
        facts.offer_title(title, TitleSource::Generated);
    }
    if let Some(at) = session.get("created_at").and_then(time_of) {
        facts
            .offer_start(at.timestamp(), at.accuracy())
            .expect("decoded source time is valid");
    }
    sink.out.sessions.push(facts);
    sink.records.record("session", &RecordOutcome::Used);
    let mut list: Vec<&Json> = session.array("turns").into_iter().flatten().collect();
    list.sort_by_key(|t| t.i64("ordinal").unwrap_or(i64::MAX));
    for (position, turn) in list.into_iter().enumerate() {
        let role_name = turn
            .str("role")
            .filter(|s| !s.is_empty())
            .unwrap_or("?")
            .to_owned();
        let header = crate::fields::header(turn, &["attachments", "blocks"]);
        sink.records.fields("turn", &header, &TURN_FIELDS);
        for block in turn.array("blocks").into_iter().flatten() {
            sink.records.fields(
                "block",
                block,
                if matches!(
                    block.str("type").filter(|s| !s.is_empty()),
                    Some("text" | "markdown" | "thinking" | "reasoning" | "image")
                ) {
                    &TURN_BLOCK_FIELDS
                } else {
                    &OTHER_BLOCK_FIELDS
                },
            );
        }
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
        let mut blocks: Vec<Block> = turn
            .array("blocks")
            .into_iter()
            .flatten()
            .filter_map(|b| turn_block(b, direction))
            .collect();
        if blocks.is_empty()
            && let Some(text) = turn
                .str("text")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        {
            blocks.push(Block::Text { text });
        }
        for a in turn.array("attachments").into_iter().flatten() {
            sink.records.fields("attachment", a, &ATTACHMENT_FIELDS);
            let stored = attachment_bytes(a, sink);
            blocks.extend(attachment(a, direction, stored));
        }
        // A turn the page shows without an ID is named by its text, so a
        // capture holding only the later turns, or one taken after a turn
        // was edited, names every other turn the same.
        let (id, id_kind) = match turn.str("provider_turn_id").filter(|s| !s.is_empty()) {
            Some(native) => (message_id(origin, native), IdKind::Native),
            None => (
                crate::content_ids::named(
                    &sid,
                    role,
                    &crate::content_ids::text(&blocks),
                    position as u64,
                ),
                IdKind::Content,
            ),
        };
        let Some(id) = id else {
            sink.records
                .record(&role_name, &RecordOutcome::unreadable("unusable turn ID"));
            continue;
        };
        let message = {
            let (at, time_accuracy) = crate::time::parts(turn.get("timestamp").and_then(time_of));
            Message {
                id,
                session: sid.clone(),
                parent: turn
                    .str("parent_turn_id")
                    .filter(|s| !s.is_empty())
                    .and_then(|p| message_id(origin, p)),
                role,
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                model: turn
                    .str("model")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                prompt_kind,
                blocks,
                usage: Default::default(),
                stop_reason: None,
                reported_cost_usd: None,
                duration_ms: None,
                attrs: crate::fields::attributes(turn, &TURN_FIELDS),
                production: Default::default(),
            }
        };
        let offset = sink.position();
        // The extension's own turns are the page's text (docs/extension.md).
        sink.out.messages.push(MessageRow {
            response: None,
            page_text: true,
            id_kind,
            offset,
            end: offset + 1,
            message,
        });
        sink.records.record(&role_name, &RecordOutcome::Used);
    }
}

/// One block of a normalized turn.
fn turn_block(b: &Json, direction: AttachmentDirection) -> Option<Block> {
    let kind = b.str("type").filter(|s| !s.is_empty()).unwrap_or("");
    Some(match kind {
        "text" | "markdown" => Block::Text {
            text: b.str("text").filter(|s| !s.is_empty()).map(str::to_owned)?,
        },
        "thinking" | "reasoning" => Block::Thinking {
            text: b.str("text").filter(|s| !s.is_empty()).map(str::to_owned),
            signature: None,
            redacted: false,
        },
        "image" => Block::Image(Attachment {
            name: b.str("name").filter(|s| !s.is_empty()).map(str::to_owned),
            media_type: b
                .str("mime_type")
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            url: b.str("url").filter(|s| !s.is_empty()).map(str::to_owned),
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
fn attachment(a: &Json, direction: AttachmentDirection, bytes: AttachmentBytes) -> Vec<Block> {
    let (stored, failure) = match bytes {
        Ok(stored) => (stored, None),
        Err(failure) => (None, Some(failure)),
    };
    let media_type = a
        .str("mime_type")
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let file = Attachment {
        name: a.str("name").filter(|s| !s.is_empty()).map(str::to_owned),
        size: stored
            .and_then(|(_, size)| size)
            .or_else(|| a.u64("size_bytes")),
        blob: stored.map(|(hash, _)| hash),
        failure,
        url: a.str("url").filter(|s| !s.is_empty()).map(str::to_owned),
        source_id: a
            .str("provider_attachment_id")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
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
    if let Some(text) = a
        .str("extracted_content")
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
    {
        out.push(Block::Other {
            type_name: "attachment_text".into(),
            data: serde_json::json!({ "file_name": a.str("name").filter(|s| !s.is_empty()), "text": text }),
        });
    }
    out
}

type AttachmentBytes = Result<Option<(BlobHash, Option<u64>)>, AttachmentFailure>;

fn attachment_bytes(a: &Json, sink: &mut Sink<'_>) -> AttachmentBytes {
    let inline_failure = match a.get("content_base64").filter(|value| !value.is_null()) {
        Some(data) => match crate::inline_bytes::decode(data.as_str(), &mut sink.out.blobs) {
            Ok((hash, size)) => return Ok(Some((hash, Some(size)))),
            Err(failure) => Some(failure),
        },
        None => None,
    };
    if let Some(value) = a.get("blob").filter(|value| !value.is_null()) {
        if let Some(hash) = value.as_str().and_then(|hash| BlobHash::parse(hash).ok()) {
            return Ok(Some((hash, a.u64("size_bytes"))));
        }
        return Err(inline_failure.unwrap_or(AttachmentFailure::InvalidBlobHash));
    }
    inline_failure.map_or(Ok(None), Err)
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
        let stored = attachment_bytes(a, sink);
        let file_id = a.str("provider_attachment_id").filter(|s| !s.is_empty());
        let name = a.str("name").filter(|s| !s.is_empty());
        let Some(msg_id) = a
            .str("message_provider_id")
            .filter(|s| !s.is_empty())
            .and_then(|m| message_id(origin, m))
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
            for x in block.attachments_mut() {
                let same = match (x.source_id.as_deref(), file_id) {
                    (Some(s), Some(f)) => s == f || f.ends_with(s) || s.ends_with(f),
                    _ => name.is_some() && x.name.as_deref() == name,
                };
                if same {
                    matched = true;
                    if x.blob.is_none() {
                        match stored {
                            Ok(Some((hash, size))) => {
                                x.blob = Some(hash);
                                x.size = size.or(x.size);
                                x.failure = None;
                            }
                            Err(failure) => x.failure = Some(failure),
                            Ok(None) => {}
                        }
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
