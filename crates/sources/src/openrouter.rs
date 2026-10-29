//! OpenRouter chat-room exports (`OpenRouter Chat <date>.json`): the
//! models in the room (`characters`) and its messages keyed by ID, each
//! naming the character that wrote it and the message it answers.
//!
//! The export has no conversation ID. Its first message in declared order
//! names the session. Matching message IDs unite repeated room exports.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::chatgpt::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, str_of, string_of, time_of};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{ExtractedBlob, MessageRow, Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Attachment, AttachmentDirection, BlobHash, Block, Event, Json, Message, PromptKind, Role,
    Session, SessionKind, TitleSource, Usage,
};
use std::path::Path;

pub const ORIGIN: &str = "openrouter";

/// The export starts with `{"version": …` and its name says what it is.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    let named = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("OpenRouter Chat") && n.ends_with(".json"));
    named && first_byte(head) == Some(b'{') && first_key(head) == Some("version")
}

pub fn open(_path: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(OpenRouter::default())))
}

#[derive(Default)]
pub struct OpenRouter {
    metadata: Option<Json>,
    session: Option<polylogue_model::SessionId>,
    characters: Json,
}

static DOCUMENT_FIELDS: Fields = Fields::new(&[
    (
        "version",
        Rule::Excluded("the export format's version string"),
    ),
    ("title", Rule::Read(T::String)),
    ("characters", Rule::Kept("complete room_models event")),
    ("messages", Rule::Read(T::Array)),
    ("artifacts", Rule::Kept("artifact panes, kept as an event")),
    (
        "artifactFiles",
        Rule::Kept("artifact panes, kept as an event"),
    ),
    (
        "artifactVersions",
        Rule::Kept("artifact panes, kept as an event"),
    ),
    (
        "artifactFileContents",
        Rule::Kept("artifact panes, kept as an event"),
    ),
])
.shape(&["characters"], &[])
.for_kind("document");

static MESSAGE_FIELDS: Fields = Fields::new(&[
    ("attachments[].type", Rule::Read(T::String)),
    ("attachments[].content", Rule::Read(T::String)),
    ("id", Rule::Read(T::String)),
    ("type", Rule::Read(T::String)),
    ("content", Rule::Read(T::StringOrArray)),
    ("contentType", Rule::Read(T::String)),
    ("characterId", Rule::Read(T::String)),
    ("parentMessageId", Rule::Read(T::String)),
    ("createdAt", Rule::Read(T::String)),
    (
        "updatedAt",
        Rule::Excluded("edit time; the creation time orders messages"),
    ),
    ("attachments", Rule::Read(T::Array)),
    ("files", Rule::Read(T::Array)),
    ("citations", Rule::Kept("web search citations")),
    (
        "context",
        Rule::Excluded("always the main chat in a room export"),
    ),
    ("isGenerating", Rule::Excluded("the page's UI state")),
    ("isRetrying", Rule::Excluded("the page's UI state")),
    ("isCollapsed", Rule::Excluded("the page's UI state")),
    (
        "isEdited",
        Rule::Kept("the operator edited the message after sending"),
    ),
    ("metadata", Rule::Read(T::Object)),
    ("metadata.tokensCount", Rule::Read(T::Number)),
    ("metadata.generateDuration", Rule::Read(T::Number)),
    ("metadata.latencyDuration", Rule::Read(T::Number)),
    ("metadata.cost", Rule::Read(T::Number)),
    (
        "metadata.provider",
        Rule::Kept("the upstream host that served the reply"),
    ),
    ("metadata.variantSlug", Rule::Read(T::String)),
    (
        "metadata.tokensPerSecond",
        Rule::Excluded("tokens divided by duration, both read"),
    ),
    (
        "metadata.plugins",
        Rule::Kept("plugins enabled for the reply"),
    ),
    ("metadata.reasoning", Rule::Read(T::String)),
])
.for_kind("message");

impl ExportParser for OpenRouter {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        *self = Self::default();
        let (metadata, stats) = crate::export::scan_map(reader, "messages", |range, key, raw| {
            item.push(
                range,
                message_order(raw, key),
                Some(key),
                raw.get().as_bytes(),
            )
        })?;
        self.metadata = Some(metadata);
        Ok(stats)
    }

    fn begin(&mut self, length: u64, first: Option<&[u8]>) -> Result<Option<Step>, String> {
        let doc = self.metadata.take().ok_or("room metadata not scanned")?;
        let mut output = Output::default();
        let mut records = Records::default();
        let first = first
            .map(serde_json::from_slice::<Json>)
            .transpose()
            .map_err(|e| e.to_string())?;
        self.start(
            &doc,
            first.as_ref(),
            &mut Sink::new(&mut output, &mut records),
        );
        for event in &mut output.events {
            event.range = Some((0, length));
        }
        Ok(Some(Step {
            output,
            counts: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            repeats_whole_conversations: false,
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
        let Some(sid) = &self.session else {
            return Ok(());
        };
        let kind = str_of(m, "type").unwrap_or("?").to_owned();
        sink.records.fields(&kind, m, &MESSAGE_FIELDS);
        match message(sid, m, &self.characters, sink) {
            Ok(message) => {
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    offset,
                    end: offset + 1,
                    message,
                });
                sink.records.record(&kind, &RecordOutcome::Used);
            }
            Err(outcome) => sink.records.record(&kind, &outcome),
        }
        Ok(())
    }
}

impl OpenRouter {
    fn start(&mut self, doc: &Json, first: Option<&Json>, sink: &mut Sink<'_>) {
        sink.records.fields("room", doc, &DOCUMENT_FIELDS);
        let Some(first_id) = first.and_then(|m| str_of(m, "id")) else {
            return sink
                .records
                .record("room", &RecordOutcome::skipped("a room with no messages"));
        };
        let Some(sid) = session_id(ORIGIN, first_id) else {
            return sink
                .records
                .record("room", &RecordOutcome::unreadable("unusable message ID"));
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = str_of(doc, "title") {
            session.offer_title(title, TitleSource::Generated);
        }
        if let Some(at) = first.and_then(|m| m.get("createdAt")).and_then(time_of) {
            session
                .offer_start(at.timestamp(), at.evidence())
                .expect("decoded source time is valid");
        }
        self.session = Some(sid.clone());
        sink.out.sessions.push(session);
        sink.records.record("room", &RecordOutcome::Used);

        self.characters = doc.get("characters").cloned().unwrap_or(Json::Null);
        sink.out.events.push({
            let (at, time_evidence) = crate::time::parts(None);
            Event {
                session: Some(sid.clone()),
                kind: "room_models".into(),
                at,
                at_quality: time_evidence.quality,
                at_precision: time_evidence.precision,
                at_until: time_evidence.until,
                after: None,
                range: None,
                data: self.characters.clone(),
            }
        });
        let artifacts: serde_json::Map<String, Json> = [
            "artifacts",
            "artifactFiles",
            "artifactVersions",
            "artifactFileContents",
        ]
        .iter()
        .filter_map(|k| {
            let v = doc.get(*k)?;
            let filled = v.as_object().is_some_and(|o| !o.is_empty())
                || v.as_array().is_some_and(|a| !a.is_empty());
            filled.then(|| ((*k).to_owned(), v.clone()))
        })
        .collect();
        if !artifacts.is_empty() {
            sink.out.events.push({
                let (at, time_evidence) = crate::time::parts(None);
                Event {
                    session: Some(sid.clone()),
                    kind: "artifacts".into(),
                    at,
                    at_quality: time_evidence.quality,
                    at_precision: time_evidence.precision,
                    at_until: time_evidence.until,
                    after: None,
                    range: None,
                    data: Json::Object(artifacts),
                }
            });
        }
    }
}

// Match (optional integer milliseconds, optional message ID, object key).
// The final key resolves equal message headers independently of Cargo features.
fn message_order(raw: &serde_json::value::RawValue, key: &str) -> Vec<u8> {
    // Borrow unrelated fields, including the body. A map keeps last-value
    // semantics for duplicate header fields, like the full message decoder.
    let fields: std::collections::BTreeMap<String, &serde_json::value::RawValue> =
        serde_json::from_str(raw.get()).unwrap_or_default();
    let at = fields
        .get("createdAt")
        .and_then(|raw| serde_json::from_str::<Json>(raw.get()).ok())
        .as_ref()
        .and_then(time_of);
    let id = fields
        .get("id")
        .and_then(|raw| serde_json::from_str::<String>(raw.get()).ok())
        .filter(|id| !id.is_empty());
    room_order(
        at.map(crate::time::SourceTime::timestamp),
        id.as_deref(),
        key,
    )
}

fn room_order(at: Option<polylogue_model::Timestamp>, id: Option<&str>, key: &str) -> Vec<u8> {
    let mut order = Vec::new();
    if let Some(time) = at {
        order.push(1);
        order.extend(((time.unix_ms() as u64) ^ (1 << 63)).to_be_bytes());
    } else {
        order.push(0);
    }
    if let Some(id) = id {
        order.push(1);
        for byte in id.bytes() {
            if byte == 0 {
                order.extend([0, 255]);
            } else {
                order.push(byte);
            }
        }
        order.extend([0, 0]);
    } else {
        order.push(0);
    }
    order.extend(key.as_bytes());
    order
}

fn message(
    sid: &polylogue_model::SessionId,
    m: &Json,
    characters: &Json,
    sink: &mut Sink<'_>,
) -> Result<Message, RecordOutcome> {
    let role = match str_of(m, "type") {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("system") => Role::System,
        Some(other) => return Err(RecordOutcome::unreadable(format!("unknown type {other}"))),
        None => return Err(RecordOutcome::unreadable("no type")),
    };
    let id = str_of(m, "id")
        .and_then(|id| message_id(ORIGIN, id))
        .ok_or_else(|| RecordOutcome::unreadable("no usable id"))?;
    match str_of(m, "contentType") {
        None | Some("text") => {}
        Some(other) => {
            return Err(RecordOutcome::skipped(format!(
                "content type {other} is not read yet"
            )));
        }
    }
    let direction = if role == Role::User {
        AttachmentDirection::Input
    } else {
        AttachmentDirection::Output
    };
    let mut blocks = Vec::new();
    if let Some(text) = str_of(m, "content") {
        blocks.push(Block::Text {
            text: text.to_owned(),
        });
    }
    if let Some(reasoning) = m.pointer("/metadata/reasoning").and_then(Json::as_str) {
        blocks.insert(
            0,
            Block::Thinking {
                text: Some(reasoning.to_owned()),
                signature: None,
                redacted: false,
            },
        );
    }
    // Pasted text the room holds beside the prompt: stored as bytes.
    for a in m
        .get("attachments")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let mut attachment = Attachment {
            name: string_of(a, "name"),
            media_type: Some(match str_of(a, "type") {
                Some("text") | None => "text/plain".into(),
                Some(other) => other.to_owned(),
            }),
            direction,
            ..Attachment::default()
        };
        if let Some(content) = str_of(a, "content") {
            let bytes = content.as_bytes().to_vec();
            let hash = BlobHash::of(&bytes);
            attachment.size = Some(bytes.len() as u64);
            attachment.blob = Some(hash);
            sink.out.blobs.push(ExtractedBlob {
                hash,
                size: bytes.len() as u64,
                bytes,
            });
        }
        blocks.push(Block::File(attachment));
    }
    for f in m
        .get("files")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        blocks.push(Block::File(Attachment {
            name: string_of(f, "name").or_else(|| string_of(f, "filename")),
            media_type: string_of(f, "type").or_else(|| string_of(f, "mimeType")),
            url: string_of(f, "url"),
            source_id: string_of(f, "id"),
            direction,
            ..Attachment::default()
        }));
    }
    if blocks.is_empty() {
        return Err(RecordOutcome::skipped("a message with no content"));
    }

    let meta = m.get("metadata");
    let model = str_of(m, "characterId")
        .and_then(|c| characters.get(c))
        .and_then(|c| string_of(c, "model"));
    let mut production = polylogue_model::Production {
        first_token_ms: meta
            .and_then(|x| x.get("latencyDuration"))
            .and_then(Json::as_u64),
        ..Default::default()
    };
    if let Some(variant) = meta.and_then(|x| str_of(x, "variantSlug"))
        && model.as_deref() != Some(variant)
    {
        production.requested_model = Some(variant.to_owned());
    }
    let reported_cost_usd = meta.and_then(|x| x.get("cost")).and_then(|c| match c {
        Json::Number(n) => Some(n.clone()),
        Json::String(s) => serde_json::from_str::<serde_json::Number>(s).ok(),
        _ => None,
    });
    let usage = Usage {
        output: meta
            .and_then(|x| x.get("tokensCount"))
            .and_then(Json::as_u64),
        ..Usage::default()
    };
    Ok({
        let (at, time_evidence) = crate::time::parts(m.get("createdAt").and_then(time_of));
        Message {
            id,
            session: sid.clone(),
            parent: str_of(m, "parentMessageId").and_then(|p| message_id(ORIGIN, p)),
            role,
            at,
            at_quality: time_evidence.quality,
            at_precision: time_evidence.precision,
            at_until: time_evidence.until,
            model: if role == Role::Assistant { model } else { None },
            prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
            blocks,
            usage: if role == Role::Assistant {
                usage
            } else {
                Usage::default()
            },
            stop_reason: None,
            reported_cost_usd: if role == Role::Assistant {
                reported_cost_usd
            } else {
                None
            },
            duration_ms: meta
                .and_then(|x| x.get("generateDuration"))
                .and_then(Json::as_u64),
            attrs: crate::fields::attributes(m, &MESSAGE_FIELDS),
            production,
        }
    })
}

#[cfg(test)]
mod tests;
