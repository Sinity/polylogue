//! Grok account exports. One stored conversation is decoded at a time;
//! response order stays as supplied, independently of derived identity hashing.

use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 3;

use crate::fields::ExpectedType as T;
mod identity;
use crate::document::{Sink, message_id, session_id};
use crate::export::{ExportParser, ExportState, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::{
    Block, Event, Json, Message, PromptKind, Role, Session, SessionKind, TitleSource,
};
use std::collections::HashMap;
use std::path::Path;

pub const ORIGIN: &str = "grok";

/// Recognize the import framework's short prefix, or the account export's
/// standard filename. Metadata before the signature can require that filename.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    path.file_name().is_some_and(|name| name == EXPORT)
        && match head.shape {
            Some(_) => head.holds("conversations", Kind::Array),
            None => head.top() == Some(Kind::Object),
        }
}

/// The account export's one file.
pub const EXPORT: &str = "prod-grok-backend.json";

pub fn open(_: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Export::default())))
}

#[derive(Default)]
pub struct Export {
    metadata: Option<Json>,
    conversation_occurrences: HashMap<String, usize>,
}

static EXPORT_FIELDS: Fields = Fields::new("export", &[("conversations", Rule::Read(T::Array))]);
static CONVERSATION_FIELDS: Fields = Fields::new(
    "conversation",
    &[
        ("id", Rule::Read(T::String)),
        ("conversationId", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("create_time", Rule::Read(T::StringOrNumber)),
        ("createTime", Rule::Read(T::StringOrNumber)),
    ],
);
static RESPONSE_FIELDS: Fields = Fields::new(
    "response",
    &[
        ("id", Rule::Read(T::String)),
        ("responseId", Rule::Read(T::String)),
        ("sender", Rule::Read(T::String)),
        ("message", Rule::Read(T::String)),
        ("create_time", Rule::Read(T::StringOrNumber)),
        ("createTime", Rule::Read(T::StringOrNumber)),
        ("model", Rule::Read(T::String)),
        ("parentResponseId", Rule::Read(T::String)),
    ],
);
static WRAPPER_FIELDS: Fields = Fields::new(
    "wrapper",
    &[
        ("conversation", Rule::Read(T::Object)),
        ("responses", Rule::Read(T::Array)),
    ],
);
static RESPONSE_WRAPPER_FIELDS: Fields =
    Fields::new("response_wrapper", &[("response", Rule::Read(T::Object))]);

impl ExportParser for Export {
    fn scan(
        &mut self,
        _input: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.conversation_occurrences.clear();
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
        let Some(responses) = value.array("responses") else {
            sink.records.record(
                "conversation",
                &RecordOutcome::unreadable("no Grok responses array"),
            );
            return;
        };
        sink.records.fields(
            "conversation wrapper",
            &crate::fields::header(value, &["conversation", "responses"]),
            &WRAPPER_FIELDS,
        );
        sink.records
            .fields("conversation", conversation, &CONVERSATION_FIELDS);
        let mut warnings = Vec::new();
        let native = match identity::native_id(conversation, "id", "conversationId") {
            Ok(Some(id)) => id.to_owned(),
            Ok(None) => {
                let hash = identity::conversation(conversation, responses, value);
                let n = self
                    .conversation_occurrences
                    .entry(hash.clone())
                    .or_default();
                let id = format!("~{hash}-{n}");
                if *n > 0 {
                    warnings.push(
                        "equal conversations without an ID are indistinguishable in the file"
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
            .str("title")
            .filter(|title| title.trim() != "New chat")
        {
            session.offer_title(title, TitleSource::Generated);
        }
        let (at, time_accuracy) = crate::time::parts(time(conversation, &mut warnings));
        session
            .set_time(at, time_accuracy)
            .expect("decoded source time is valid");
        sink.out.sessions.push(session);
        sink.records.record("conversation", &RecordOutcome::Used);
        for entry in responses {
            let fields = if let Some(response) = entry.get("response") {
                sink.records.fields(
                    "response wrapper",
                    &crate::fields::header(entry, &["response"]),
                    &RESPONSE_WRAPPER_FIELDS,
                );
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
                .str("sender")
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
            let Some(text) = fields.str("message") else {
                sink.records.record(
                    "response",
                    &RecordOutcome::unreadable("Grok message is not a supplied string"),
                );
                warnings.push("missing, null or structured message is not read as text; original bytes stay in the stored input".into());
                continue;
            };
            let blocks = vec![Block::Text {
                text: text.to_owned(),
            }];
            let offset = sink.position();
            // A response without an ID is named by what it says
            // (`crate::content_ids`), so equal responses settle by order.
            let (id, id_kind) = match identity::native_id(fields, "id", "responseId") {
                Ok(Some(native)) => (message_id(ORIGIN, native), polylogue_model::IdKind::Native),
                Ok(None) => (
                    crate::content_ids::named(&sid, role, &blocks, offset),
                    polylogue_model::IdKind::Content,
                ),
                Err(reason) => {
                    sink.records
                        .record("response", &RecordOutcome::unreadable(reason));
                    warnings.push(reason.into());
                    continue;
                }
            };
            let Some(id) = id else {
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
            sink.out.messages.push(MessageRow {
                response: None,
                page_text: false,
                id_kind,
                offset,
                end: offset + 1,
                message: {
                    let (at, time_accuracy) = crate::time::parts(at);
                    Message {
                        id,
                        session: sid.clone(),
                        parent: fields
                            .str("parentResponseId")
                            .filter(|id| !id.is_empty())
                            .and_then(|id| message_id(ORIGIN, id)),
                        role,
                        at,
                        at_quality: time_accuracy.quality,
                        at_precision: time_accuracy.precision,
                        at_until: time_accuracy.until,
                        model,
                        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
                        blocks,
                        usage: Default::default(),
                        stop_reason: None,
                        reported_cost_usd: None,
                        duration_ms: None,
                        attrs: Default::default(),
                        production: Default::default(),
                    }
                },
            });
            sink.records.record("response", &RecordOutcome::Used);
        }
        if !warnings.is_empty() {
            warnings.sort();
            warnings.dedup();
            sink.out.events.push({
                let (at, time_accuracy) = crate::time::parts(None);
                Event {
                    session: Some(sid),
                    kind: "grok_import_diagnostic".into(),
                    at,
                    at_quality: time_accuracy.quality,
                    at_precision: time_accuracy.precision,
                    at_until: time_accuracy.until,
                    after: None,
                    range: None,
                    data: serde_json::json!({"warnings": warnings}),
                }
            });
        }
    }
}

fn time(fields: &Json, warnings: &mut Vec<String>) -> Option<crate::time::SourceTime> {
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
