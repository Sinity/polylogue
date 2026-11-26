//! OpenRouter chat-room exports (`OpenRouter Chat <date>.json`): the
//! models in the room (`characters`) and its messages keyed by ID, each
//! naming the character that wrote it and the message it answers.
//!
//! The export has no conversation ID. Its first message in declared order
//! names the session. Matching message IDs unite repeated room exports.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 2;

use crate::document::first_byte;
use crate::document::{Sink, first_key, message_id, session_id, text_of, time_of};
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
///
/// Read whole, the room's shape decides whatever the file is named: a
/// `version`, a `messages` object, and the room's `id` or `characters`.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    let named = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("OpenRouter Chat") && n.ends_with(".json"));
    if head.shape.is_some() {
        return head.top() == Some(Kind::Object)
            && head.has("version")
            && head.holds("messages", Kind::Object)
            && (named || head.has("id") || head.has("characters"));
    }
    let head = head.bytes;
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

static DOCUMENT_FIELDS: Fields = Fields::new(
    "document",
    &[
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
    ],
)
.shape(&["characters"], &[]);

static MESSAGE_FIELDS: Fields = Fields::new(
    "message",
    &[
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
    ],
);

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
        let kind = m
            .str("type")
            .filter(|s| !s.is_empty())
            .unwrap_or("?")
            .to_owned();
        sink.records.fields(&kind, m, &MESSAGE_FIELDS);
        match message(sid, m, &self.characters, sink) {
            Ok(message) => {
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    response: None,
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Native,
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
        let Some(first_id) = first.and_then(|m| m.str("id").filter(|s| !s.is_empty())) else {
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
        if let Some(title) = doc.str("title").filter(|s| !s.is_empty()) {
            session.offer_title(title, TitleSource::Generated);
        }
        if let Some(at) = first.and_then(|m| m.get("createdAt")).and_then(time_of) {
            session
                .offer_start(at.timestamp(), at.accuracy())
                .expect("decoded source time is valid");
        }
        self.session = Some(sid.clone());
        sink.out.sessions.push(session);
        sink.records.record("room", &RecordOutcome::Used);

        self.characters = doc.get("characters").cloned().unwrap_or(Json::Null);
        sink.out.events.push({
            let (at, time_accuracy) = crate::time::parts(None);
            Event {
                session: Some(sid.clone()),
                kind: "room_models".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
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
                let (at, time_accuracy) = crate::time::parts(None);
                Event {
                    session: Some(sid.clone()),
                    kind: "artifacts".into(),
                    at,
                    at_quality: time_accuracy.quality,
                    at_precision: time_accuracy.precision,
                    at_until: time_accuracy.until,
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
    let role = match m.str("type").filter(|s| !s.is_empty()) {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("system") => Role::System,
        Some(other) => return Err(RecordOutcome::unreadable(format!("unknown type {other}"))),
        None => return Err(RecordOutcome::unreadable("no type")),
    };
    let id = m
        .str("id")
        .filter(|s| !s.is_empty())
        .and_then(|id| message_id(ORIGIN, id))
        .ok_or_else(|| RecordOutcome::unreadable("no usable id"))?;
    match m.str("contentType").filter(|s| !s.is_empty()) {
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
    if let Some(text) = text_of(m.get("content")) {
        blocks.push(Block::Text { text });
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
    for a in m.array("attachments").into_iter().flatten() {
        let mut attachment = Attachment {
            name: a.str("name").filter(|s| !s.is_empty()).map(str::to_owned),
            media_type: Some(match a.str("type").filter(|s| !s.is_empty()) {
                Some("text") | None => "text/plain".into(),
                Some(other) => other.to_owned(),
            }),
            direction,
            ..Attachment::default()
        };
        if let Some(content) = a.str("content").filter(|s| !s.is_empty()) {
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
    for f in m.array("files").into_iter().flatten() {
        blocks.push(Block::File(Attachment {
            name: f
                .str("name")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    f.str("filename")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                }),
            media_type: f
                .str("type")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    f.str("mimeType")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                }),
            url: f.str("url").filter(|s| !s.is_empty()).map(str::to_owned),
            source_id: f.str("id").filter(|s| !s.is_empty()).map(str::to_owned),
            direction,
            ..Attachment::default()
        }));
    }
    if blocks.is_empty() {
        return Err(RecordOutcome::skipped("a message with no content"));
    }

    let meta = m.get("metadata");
    let model = m
        .str("characterId")
        .filter(|s| !s.is_empty())
        .and_then(|c| characters.get(c))
        .and_then(|c| c.str("model").filter(|s| !s.is_empty()).map(str::to_owned));
    let mut production = polylogue_model::Production {
        first_token_ms: meta.and_then(|x| x.u64("latencyDuration")),
        ..Default::default()
    };
    if let Some(variant) = meta.and_then(|x| x.str("variantSlug").filter(|s| !s.is_empty()))
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
        output: meta.and_then(|x| x.u64("tokensCount")),
        ..Usage::default()
    };
    Ok({
        let (at, time_accuracy) = crate::time::parts(m.get("createdAt").and_then(time_of));
        Message {
            id,
            session: sid.clone(),
            parent: m
                .str("parentMessageId")
                .filter(|s| !s.is_empty())
                .and_then(|p| message_id(ORIGIN, p)),
            role,
            at,
            at_quality: time_accuracy.quality,
            at_precision: time_accuracy.precision,
            at_until: time_accuracy.until,
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
            duration_ms: meta.and_then(|x| x.u64("generateDuration")),
            attrs: crate::fields::attributes(m, &MESSAGE_FIELDS),
            production,
        }
    })
}

#[cfg(test)]
mod tests;
