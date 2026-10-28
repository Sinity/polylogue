//! Grok account exports. One stored conversation is decoded at a time;
//! response order stays as supplied, independently of derived identity hashing.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

mod identity;
use crate::document::{Sink, first_key, message_id, session_id};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, Timestamp, TitleSource,
};
use std::collections::HashMap;
use std::path::Path;

pub const ORIGIN: &str = "grok";

/// Recognize the import framework's short prefix, or the account export's
/// standard filename. Metadata before the signature can require that filename.
pub fn recognize(path: &Path, head: &[u8]) -> bool {
    if path.extension().is_none_or(|extension| extension != "json") {
        return false;
    }
    first_key(head) == Some("conversations")
        || serde_json::from_slice::<Json>(head)
            .is_ok_and(|value| value.get("conversations").is_some())
        || (path
            .file_name()
            .is_some_and(|name| name == "prod-grok-backend.json")
            && crate::chatgpt::first_byte(head) == Some(b'{'))
}

pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Export::default())))
}

#[derive(Default)]
pub struct Export {
    metadata: Option<Json>,
    occurrences: HashMap<String, usize>,
}

static EXPORT_FIELDS: Fields = Fields::new(&[("conversations", Rule::Read)]);
static CONVERSATION_FIELDS: Fields = Fields::new(&[
    ("id", Rule::Read),
    ("conversationId", Rule::Read),
    ("title", Rule::Read),
    ("create_time", Rule::Read),
    ("createTime", Rule::Read),
]);
static RESPONSE_FIELDS: Fields = Fields::new(&[
    ("id", Rule::Read),
    ("responseId", Rule::Read),
    ("sender", Rule::Read),
    ("message", Rule::Read),
    ("create_time", Rule::Read),
    ("createTime", Rule::Read),
    ("model", Rule::Read),
    ("parentResponseId", Rule::Read),
]);
static WRAPPER_FIELDS: Fields =
    Fields::new(&[("conversation", Rule::Read), ("responses", Rule::Read)]);
static RESPONSE_WRAPPER: Fields = Fields::new(&[("response", Rule::Read)]);

impl ExportParser for Export {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.occurrences.clear();
        self.metadata = None;
        let (metadata, stats) =
            crate::export::scan_nested_unique(reader, &["conversations"], |range, raw| {
                item.push(range, Vec::new(), None, raw.get().as_bytes())
            })?;
        self.metadata = Some(metadata);
        Ok(stats)
    }

    fn begin(&mut self, length: u64, _: Option<&[u8]>) -> Result<Option<Step>, String> {
        let metadata = self
            .metadata
            .take()
            .ok_or("Grok export metadata was not scanned")?;
        let mut records = Records::default();
        records.fields("export", &metadata, &EXPORT_FIELDS);
        records.record(
            "export",
            &if metadata.get("conversations").is_some_and(Json::is_array) {
                RecordOutcome::Used
            } else {
                RecordOutcome::unreadable("Grok export has no conversations array")
            },
        );
        Ok(Some(Step {
            output: Output::default(),
            counts: records.finish().map_err(|error| error.to_string())?,
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
        let value: Json = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        self.conversation(&value, sink);
        Ok(())
    }
}

impl Export {
    fn conversation(&mut self, value: &Json, sink: &mut Sink<'_>) {
        let Some(conversation) = value.get("conversation").filter(|value| value.is_object()) else {
            sink.records.record(
                "conversation",
                &RecordOutcome::unreadable("no Grok conversation object"),
            );
            return;
        };
        let Some(responses) = value.get("responses").and_then(Json::as_array) else {
            sink.records.record(
                "conversation",
                &RecordOutcome::unreadable("no Grok responses array"),
            );
            return;
        };
        sink.records
            .fields("conversation wrapper", value, &WRAPPER_FIELDS);
        sink.records
            .fields("conversation", conversation, &CONVERSATION_FIELDS);
        let mut warnings = Vec::new();
        let native = match identity::native(conversation, "id", "conversationId") {
            Ok(Some(id)) => id.to_owned(),
            Ok(None) => {
                let hash = identity::conversation(conversation, responses, value);
                let n = self.occurrences.entry(hash.clone()).or_default();
                let id = format!("~{hash}-{n}");
                if *n > 0 {
                    warnings.push(
                        "equal ID-less conversations have indistinguishable physical occurrences"
                            .to_owned(),
                    );
                }
                *n += 1;
                warnings.push("content-derived conversation: filename and ordering are stable, but append/edit continuity is unproved".to_owned());
                id
            }
            Err(reason) => {
                sink.records
                    .record("conversation", &RecordOutcome::unreadable(reason));
                return;
            }
        };
        let Some(sid) = session_id(ORIGIN, &native) else {
            sink.records.record(
                "conversation",
                &RecordOutcome::unreadable("unusable Grok conversation identity"),
            );
            return;
        };
        let mut session = Session::new(sid.clone(), SessionKind::Main);
        if let Some(title) = conversation
            .get("title")
            .and_then(Json::as_str)
            .filter(|title| title.trim() != "New chat")
        {
            session.offer_title(title, TitleSource::Generated);
        }
        session.started = time(conversation, &mut warnings);
        sink.out.sessions.push(session);
        sink.records.record("conversation", &RecordOutcome::Used);
        let mut occurrences = HashMap::<String, usize>::new();
        for entry in responses {
            let fields = if let Some(response) = entry.get("response") {
                sink.records
                    .fields("response wrapper", entry, &RESPONSE_WRAPPER);
                response
            } else {
                entry
            };
            sink.records.fields("response", fields, &RESPONSE_FIELDS);
            let Some(fields) = fields.as_object().map(|_| fields) else {
                sink.records.record(
                    "response",
                    &RecordOutcome::unreadable("Grok response is not an object"),
                );
                warnings.push(
                    "a response is not an object; original bytes stay in the stored input".into(),
                );
                continue;
            };
            let role = match fields
                .get("sender")
                .and_then(Json::as_str)
                .map(|sender| sender.trim().to_ascii_lowercase())
                .as_deref()
            {
                Some("human" | "user") => Role::User,
                Some("assistant" | "grok") => Role::Assistant,
                _ => {
                    sink.records.record(
                        "response",
                        &RecordOutcome::skipped(
                            "unrecognized Grok sender; authorship was not assigned",
                        ),
                    );
                    warnings.push(
                        "unrecognized response sender; original bytes stay in the stored input"
                            .into(),
                    );
                    continue;
                }
            };
            let Some(text) = fields.get("message").and_then(Json::as_str) else {
                sink.records.record(
                    "response",
                    &RecordOutcome::unreadable("Grok message is not a supplied string"),
                );
                warnings.push("missing, null or structured message is not read as text; original bytes stay in the stored input".into());
                continue;
            };
            let native_message = match identity::native(fields, "id", "responseId") {
                Ok(Some(id)) => id.to_owned(),
                Ok(None) => {
                    let hash = identity::response(entry);
                    let n = occurrences.entry(hash.clone()).or_default();
                    let id = format!("{native}:~{}-{n}", &hash[..12]);
                    if *n > 0 {
                        warnings.push("equal ID-less responses keep distinct occurrence IDs; physical correspondence is ambiguous".into());
                    }
                    *n += 1;
                    id
                }
                Err(reason) => {
                    sink.records
                        .record("response", &RecordOutcome::unreadable(reason));
                    warnings.push(reason.into());
                    continue;
                }
            };
            let Some(id) = message_id(ORIGIN, &native_message) else {
                sink.records.record(
                    "response",
                    &RecordOutcome::unreadable("unusable Grok message identity"),
                );
                continue;
            };
            let at = time(fields, &mut warnings);
            let model = if role == Role::Assistant {
                match fields.get("model") {
                    Some(Json::String(model)) => Some(model.clone()),
                    Some(value) if !value.is_null() => {
                        warnings.push("non-string response model was not inferred".into());
                        None
                    }
                    _ => None,
                }
            } else {
                None
            };
            let offset = sink.position();
            sink.out.messages.push(MessageRow {
                offset,
                end: offset + 1,
                message: Message {
                    id,
                    session: sid.clone(),
                    parent: fields
                        .get("parentResponseId")
                        .and_then(Json::as_str)
                        .filter(|id| !id.is_empty())
                        .and_then(|id| message_id(ORIGIN, id)),
                    role,
                    at,
                    model,
                    prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
                    blocks: vec![Block::Text {
                        text: text.to_owned(),
                    }],
                    usage: Default::default(),
                    stop_reason: None,
                    reported_cost_usd: None,
                    duration_ms: None,
                    production: Default::default(),
                },
            });
            sink.records.record("response", &RecordOutcome::Used);
        }
        if !warnings.is_empty() {
            warnings.sort();
            warnings.dedup();
            sink.out.events.push(Event {
                session: Some(sid),
                kind: "grok_import_diagnostic".into(),
                at: None,
                after: None,
                range: None,
                data: serde_json::json!({"warnings": warnings}),
            });
        }
    }
}

fn time(fields: &Json, warnings: &mut Vec<String>) -> Option<Timestamp> {
    match identity::time(fields) {
        Ok(time) => time,
        Err(reason) => {
            warnings.push(reason.into());
            None
        }
    }
}

#[cfg(test)]
mod tests;
