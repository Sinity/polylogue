//! The browser extension's capture spool: one JSON file per captured
//! conversation, rewritten in place when the conversation is captured
//! again (`$XDG_STATE_HOME/polylogue/browser-capture/<provider>/*.json`).
//!
//! A capture carries the provider's own conversation JSON
//! (`raw_provider_payload`) where the provider page exposes it. That JSON
//! is read by the provider's export parser, so a captured conversation and
//! the same conversation in an export get the same IDs and unite
//! (architecture 5.5). Where there is no provider JSON, the extension's own
//! normalized turns are read instead, under the provider's origin.
//! Attachment bytes the extension inlined are stored and linked to their
//! attachment.

#[cfg(test)]
mod tests;

use crate::account::{Accounting, Disposition};
use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::fields::{Fields, Rule, unaccounted};
use crate::origin::Root;
use crate::parse::{DocumentParser, InlineBlob, MessageRow, Output, StepParser, Whole};
use base64::Engine as _;
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Message, PromptKind, Role,
    Session, SessionKind, TitleSource,
};
use std::path::Path;

pub const ROOT: Root = Root {
    env: Some("XDG_STATE_HOME"),
    default: ".local/state",
};

pub const LOCATIONS: &[&str] = &["polylogue/browser-capture/*/*.json"];

const KIND_KEY: &str = "polylogue_capture_kind";

/// A capture starts `{"polylogue_capture_kind"`, or `{"capture_id"` when
/// the extension wrote its keys sorted.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    path.extension().is_some_and(|e| e == "json")
        && first_byte(head) == Some(b'{')
        && matches!(first_key(head), Some(KIND_KEY | "capture_id"))
}

pub fn open(path: &Path) -> Option<Box<dyn StepParser>> {
    path.extension()
        .is_some_and(|e| e == "json")
        .then(|| Box::new(Whole(Capture)) as Box<dyn StepParser>)
}

const CAPTURE_FIELDS: Fields = &[
    (KIND_KEY, Rule::Read),
    ("schema_version", Rule::Read),
    (
        "capture_id",
        Rule::Excluded("the extension's own key; the provider IDs identify the conversation"),
    ),
    (
        "source",
        Rule::Excluded("names the capturing program, always the extension"),
    ),
    ("provenance", Rule::Read),
    ("provider_meta", Rule::Read),
    ("raw_provider_payload", Rule::Read),
    ("session", Rule::Read),
];

/// The origin a capture's `session.provider` names: the provider's export
/// origin, so a capture and an export of one conversation unite.
pub fn origin_of(provider: &str) -> &str {
    match provider {
        "chatgpt" => crate::chatgpt::ORIGIN,
        "claude-ai" | "claude" => crate::claude_ai::ORIGIN,
        other => other,
    }
}

/// The session a capture of `provider_session_id` from `provider` lands in.
pub fn session_for(
    provider: &str,
    provider_session_id: &str,
) -> Option<polylogue_model::SessionId> {
    session_id(origin_of(provider), provider_session_id)
}

pub struct Capture;

impl DocumentParser for Capture {
    fn document(&self, bytes: &[u8], out: &mut Output, acc: &mut Accounting) {
        let mut sink = Sink::new(out, acc);
        let capture = match serde_json::from_slice::<Json>(bytes) {
            Ok(v) if v.is_object() => v,
            Ok(_) => return sink.record("capture", &Disposition::refused("not an object")),
            Err(e) => {
                return sink.record(
                    "capture",
                    &Disposition::refused(format!("malformed JSON: {e}")),
                );
            }
        };
        for path in unaccounted(&capture, CAPTURE_FIELDS) {
            sink.acc.unread("capture", &path);
        }
        // `source` names the capturing program (`browser-extension`); the
        // provider whose conversation it is sits in the session.
        let Some(provider) = capture
            .pointer("/session/provider")
            .and_then(Json::as_str)
            .filter(|p| !p.is_empty())
        else {
            return sink.record("capture", &Disposition::refused("no session.provider"));
        };
        let origin = origin_of(provider);
        if polylogue_model::Origin::new(origin).is_err() {
            return sink.record(
                "capture",
                &Disposition::refused(format!("source {origin:?} is not an origin name")),
            );
        }
        sink.record("capture", &Disposition::Materialized);

        let raw = capture
            .get("raw_provider_payload")
            .filter(|r| r.as_object().is_some_and(|o| !o.is_empty()));
        let first_message = sink.out.messages.len();
        match (origin, raw) {
            ("chatgpt", Some(raw)) if raw.get("mapping").is_some() => {
                crate::chatgpt::conversation(raw, &mut sink, &mut Default::default());
            }
            ("claude-ai", Some(raw)) if raw.get("chat_messages").is_some() => {
                crate::claude_ai::conversation(raw, &mut sink);
            }
            _ => match capture.get("session").filter(|s| s.is_object()) {
                Some(session) => turns(origin, session, &mut sink),
                None => sink.record(
                    "session",
                    &Disposition::refused("neither provider JSON nor turns"),
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
                session,
                kind: "browser_capture".into(),
                at: provenance.get("captured_at").and_then(time_of),
                after: None,
                range: None,
                data: serde_json::json!({
                    "schema_version": capture.get("schema_version"),
                    "provenance": provenance,
                    "provider_meta": capture.get("provider_meta"),
                }),
            });
        }
        attach_bytes(&capture, origin, first_message, &mut sink);
    }
}

/// The extension's normalized turns, for a provider page that exposed no
/// conversation JSON.
fn turns(origin: &str, session: &Json, sink: &mut Sink<'_>) {
    let Some(native) = str_of(session, "provider_session_id") else {
        sink.record("session", &Disposition::refused("no provider_session_id"));
        return;
    };
    let Some(sid) = session_id(origin, native) else {
        sink.record("session", &Disposition::refused("unusable session ID"));
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
    sink.record("session", &Disposition::Materialized);
    let mut list: Vec<&Json> = session
        .get("turns")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .collect();
    list.sort_by_key(|t| t.get("ordinal").and_then(Json::as_i64).unwrap_or(i64::MAX));
    for (i, turn) in list.into_iter().enumerate() {
        let role_name = str_of(turn, "role").unwrap_or("?").to_owned();
        let (role, prompt_kind, direction) = match role_name.as_str() {
            "user" | "human" => (
                Role::User,
                Some(PromptKind::Prose),
                AttachmentDirection::Input,
            ),
            "assistant" | "model" => (Role::Assistant, None, AttachmentDirection::Output),
            "system" => (Role::System, None, AttachmentDirection::Input),
            other => {
                sink.record(other, &Disposition::unknown(format!("role {other}")));
                continue;
            }
        };
        // A turn the page shows without an ID keeps its identity by its
        // place in the conversation (architecture 5.5).
        let turn_native = string_of(turn, "provider_turn_id").unwrap_or_else(|| {
            let place = turn
                .get("ordinal")
                .and_then(Json::as_u64)
                .unwrap_or(i as u64);
            format!("{native}#{place}")
        });
        let Some(id) = message_id(origin, &turn_native) else {
            sink.record(&role_name, &Disposition::refused("unusable turn ID"));
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
            production: Default::default(),
        };
        let offset = sink.position();
        sink.out.messages.push(MessageRow {
            offset,
            end: offset + 1,
            message,
        });
        sink.record(&role_name, &Disposition::Materialized);
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

/// Stores the attachment bytes the extension inlined (`content_base64`)
/// and points the matching attachment at them: the message by its provider
/// ID, the attachment by its file ID or, failing that, its name. An
/// attachment with bytes and no match is added to its message.
fn attach_bytes(capture: &Json, origin: &str, first_message: usize, sink: &mut Sink<'_>) {
    for a in capture
        .pointer("/session/attachments")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let Some(data) = str_of(a, "content_base64") else {
            continue;
        };
        let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(data) else {
            continue;
        };
        let hash = BlobHash::of(&bytes);
        let size = bytes.len() as u64;
        sink.out.blobs.push(InlineBlob { hash, size, bytes });
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
                if same && x.blob.is_none() {
                    x.blob = Some(hash);
                    x.size.get_or_insert(size);
                    matched = true;
                }
            }
        }
        if !matched {
            let direction = if row.message.role == Role::User {
                AttachmentDirection::Input
            } else {
                AttachmentDirection::Output
            };
            row.message.blocks.push(Block::File(Attachment {
                name: name.map(str::to_owned),
                media_type: string_of(a, "mime_type"),
                size: Some(size),
                blob: Some(hash),
                url: string_of(a, "url"),
                source_id: file_id.map(str::to_owned),
                direction,
            }));
        }
    }
}
