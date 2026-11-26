//! Account/project text is staged as documents, never conversation turns.
/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use super::fields;
use crate::document::Sink;
use crate::export::{self, ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule};
use crate::parse::{Output, Parse, Step};
use crate::{RecordOutcome, Records};
use polylogue_model::JsonExt;
use polylogue_model::{
    Block, Document, DocumentId, DocumentPurpose, DocumentRow, Json, NativeId, Origin,
};
use serde_json::value::RawValue;
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Copy)]
enum Shape {
    Memories,
    ProjectList,
    ProjectFile,
}
fn shape(path: &Path) -> Option<Shape> {
    match path.file_name()?.to_str()? {
        "memories.json" => Some(Shape::Memories),
        "projects.json" => Some(Shape::ProjectList),
        _ if path.extension().is_some_and(|e| e == "json")
            && path.parent()?.file_name().is_some_and(|p| p == "projects") =>
        {
            Some(Shape::ProjectFile)
        }
        _ => None,
    }
}
pub(super) fn recognize(path: &Path, head: &[u8]) -> bool {
    shape(path).is_some_and(|shape| {
        matches!(
            (shape, crate::document::first_byte(head)),
            (Shape::Memories | Shape::ProjectList, Some(b'[')) | (Shape::ProjectFile, Some(b'{'))
        )
    })
}
pub(super) fn open(path: &Path) -> Option<Parse> {
    Some(Parse::Export(Box::new(Sidecars {
        shape: shape(path)?,
        items: Vec::new(),
        metadata: Vec::new(),
        at: 0,
    })))
}

struct Item {
    document: Option<Document>,
    parent: u64,
    suffix: String,
    missing_child: bool,
    kind: &'static str,
    keys: Json,
    /// Why the record yields no document; set only when `document` is `None`.
    skip_reason: Option<String>,
}
struct Metadata {
    kind: &'static str,
    keys: Json,
    diagnostic: Option<String>,
}
struct Sidecars {
    shape: Shape,
    items: Vec<Item>,
    metadata: Vec<Metadata>,
    at: usize,
}
static ACCOUNT: Fields = Fields::new(
    "account",
    &[
        ("account_uuid", Rule::Read(T::String)),
        ("conversations_memory", Rule::Read(T::String)),
        ("project_memories", Rule::Read(T::Object)),
        ("created_at", Rule::Read(T::StringOrNumber)),
        ("updated_at", Rule::Read(T::StringOrNumber)),
    ],
);
static PROJECT: Fields = Fields::new(
    "project",
    &[
        ("uuid", Rule::Read(T::String)),
        ("name", Rule::Read(T::String)),
        (
            "description",
            Rule::Kept("project descriptor, not supplied document text"),
        ),
        ("is_private", Rule::Kept("project visibility")),
        ("is_starter_project", Rule::Read(T::Boolean)),
        ("prompt_template", Rule::Read(T::String)),
        ("docs", Rule::Read(T::Array)),
        ("created_at", Rule::Read(T::StringOrNumber)),
        ("updated_at", Rule::Read(T::StringOrNumber)),
        (
            "creator",
            Rule::Kept("project creator is not text authorship"),
        ),
    ],
);
static DOC: Fields = Fields::new(
    "doc",
    &[
        ("uuid", Rule::Read(T::String)),
        ("filename", Rule::Read(T::String)),
        ("content", Rule::Read(T::StringOrArray)),
        ("content[].type", Rule::Read(T::String)),
        ("content[].text", Rule::Read(T::String)),
        ("created_at", Rule::Read(T::StringOrNumber)),
        ("updated_at", Rule::Read(T::StringOrNumber)),
        (
            "file_type",
            Rule::Kept("reference descriptor; original bytes are not supplied"),
        ),
        (
            "file_size",
            Rule::Kept("reference descriptor; original bytes are not supplied"),
        ),
        ("url", Rule::Kept("reference kept without fetching")),
    ],
);

fn field_names<'a>(names: impl Iterator<Item = &'a String>) -> Json {
    Json::Object(names.map(|name| (name.clone(), Json::Null)).collect())
}
fn native(fields: &BTreeMap<String, &RawValue>, name: &str) -> Option<String> {
    fields::text(fields, name).filter(|id| !id.is_empty())
}
fn component(id: &str) -> String {
    id.replace('%', "%25").replace(':', "%3A")
}
fn scope(input: &str, kind: &str, id: Option<&str>, at: u64) -> (String, bool) {
    match id {
        Some(id) => (format!("{kind}:{}", component(id)), false),
        None => (format!("~input:{}:at:{at}", component(input)), true),
    }
}
/// The declared time: `updated_at`, else `created_at`, skipping nulls. The
/// flag says the value is present but not a readable time.
fn declared_time(get: impl Fn(&str) -> Option<Json>) -> (Option<crate::time::SourceTime>, bool) {
    for key in ["updated_at", "created_at"] {
        if let Some(value) = get(key).filter(|value| !value.is_null()) {
            // These sidecar fields declare ISO times with an offset, not epochs.
            let time = crate::time::rfc3339(&value);
            return (time, time.is_none());
        }
    }
    (None, false)
}
fn document(
    native: String,
    purpose: DocumentPurpose,
    title: Option<String>,
    source_time: Option<crate::time::SourceTime>,
    source_kind: &str,
) -> Result<Document, String> {
    let id = DocumentId::new(
        Origin::new(super::ORIGIN).map_err(|e| e.to_string())?,
        NativeId::new(&native).map_err(|e| e.to_string())?,
    );
    let mut document = Document::new(id, purpose);
    document.title = title;
    let (at, accuracy) = crate::time::parts(source_time);
    document
        .set_time(at, accuracy)
        .expect("decoded source time is valid");
    document.metadata = serde_json::json!({"source_kind":source_kind});
    Ok(document)
}

impl Sidecars {
    fn stage(
        &mut self,
        sink: &mut dyn ItemSink,
        range: Range<u64>,
        raw: &RawValue,
        item: Item,
    ) -> Result<(), String> {
        let order = (self.items.len() as u64).to_be_bytes().to_vec();
        sink.push(range, order, None, raw.get().as_bytes())?;
        self.items.push(item);
        Ok(())
    }
    fn frame(
        &mut self,
        input: &str,
        parent: u64,
        field: &str,
        key: Option<&str>,
        record: (Range<u64>, &RawValue),
        sink: &mut dyn ItemSink,
    ) -> Result<(), String> {
        let (range, raw) = record;
        let (purpose, suffix, missing_child, kind, source_kind, title, at, keys) = match field {
            "conversations_memory" => (
                DocumentPurpose::Memory,
                "memory".into(),
                false,
                "memory_text",
                "account_memory",
                None,
                None,
                Json::Null,
            ),
            "project_memories" => {
                let native = key.filter(|key| !key.is_empty());
                let suffix = match native {
                    Some(key) => format!("project:{}:memory", component(key)),
                    None => format!("~input:{}:at:{}", component(input), range.start),
                };
                (
                    DocumentPurpose::Memory,
                    suffix,
                    native.is_none(),
                    "memory_text",
                    "project_memory",
                    None,
                    None,
                    Json::Null,
                )
            }
            "prompt_template" => (
                DocumentPurpose::Instruction,
                "instructions".into(),
                false,
                "instruction_text",
                "project_instructions",
                None,
                None,
                Json::Null,
            ),
            "docs" => {
                let fields = match fields::object(raw.get().as_bytes()) {
                    Ok(fields) => fields,
                    Err(reason) => {
                        return self.stage(
                            sink,
                            range,
                            raw,
                            Item {
                                document: None,
                                parent,
                                suffix: String::new(),
                                missing_child: true,
                                kind: "project_document",
                                keys: Json::Null,
                                skip_reason: Some(reason),
                            },
                        );
                    }
                };
                let id = native(&fields, "uuid");
                let (suffix, missing) = scope(input, "doc", id.as_deref(), range.start);
                // An unreadable time leaves the document's time unknown; the
                // document is still a row.
                let (at, _) = declared_time(|key| {
                    fields
                        .get(key)
                        .map(|raw| serde_json::from_str(raw.get()).unwrap_or(Json::Null))
                });
                (
                    DocumentPurpose::Context,
                    suffix,
                    missing,
                    "project_document",
                    "project_document",
                    fields::text(&fields, "filename"),
                    at,
                    field_names(fields.keys()),
                )
            }
            _ => return Err("unknown sidecar field".into()),
        };
        let mut doc = document(
            format!("~input:{}:at:{parent}:{suffix}", component(input)),
            purpose,
            title,
            at,
            source_kind,
        )?;
        if purpose == DocumentPurpose::Context {
            doc.metadata["text_kind"] = Json::String("extracted_text".into());
        }
        self.stage(
            sink,
            range,
            raw,
            Item {
                document: Some(doc),
                parent,
                suffix,
                missing_child,
                kind,
                keys,
                skip_reason: None,
            },
        )
    }
}

impl ExportParser for Sidecars {
    fn scan(
        &mut self,
        input: &str,
        reader: &mut dyn Input,
        sink: &mut dyn ItemSink,
    ) -> Result<ScanStats, ScanError> {
        self.items.clear();
        self.metadata.clear();
        self.at = 0;
        if input.is_empty() {
            return Err(ScanError::Callback(
                "sidecar parsing needs its input identity".into(),
            ));
        }
        use export::FieldShape::{Array, Map, Value};
        let account = matches!(self.shape, Shape::Memories);
        let frames: &[(&str, export::FieldShape)] = if account {
            &[("conversations_memory", Value), ("project_memories", Map)]
        } else {
            &[("prompt_template", Value), ("docs", Array)]
        };
        let keep = if account {
            &["account_uuid", "created_at", "updated_at"][..]
        } else {
            &[
                "uuid",
                "name",
                "created_at",
                "updated_at",
                "is_starter_project",
            ][..]
        };
        let (objects, stats) = export::scan_objects(
            reader,
            !matches!(self.shape, Shape::ProjectFile),
            frames,
            keep,
            |parent, field, key, range, raw| {
                self.frame(input, parent, field, key, (range, raw), sink)
            },
        )?;
        let parents: BTreeMap<_, _> = objects
            .into_iter()
            .map(|object| (object.start, object.fields))
            .collect();
        for fields in parents.values() {
            let keys = field_names(
                fields
                    .as_object()
                    .ok_or_else(|| ScanError::Callback("sidecar metadata is not an object".into()))?
                    .keys(),
            );
            let bad_time = declared_time(|key| fields.get(key).cloned()).1;
            self.metadata.push(Metadata {
                kind: if account { "account_memory" } else { "project" },
                keys,
                diagnostic: bad_time
                    .then(|| "unreadable declared source time; time remains unknown".into()),
            });
        }
        for item in &mut self.items {
            let Some(doc) = &mut item.document else {
                continue;
            };
            let fields = parents
                .get(&item.parent)
                .ok_or_else(|| ScanError::Callback("missing sidecar source identity".into()))?;
            let id = fields
                .str(if account { "account_uuid" } else { "uuid" })
                .filter(|id| !id.is_empty());
            let (scope, missing) = scope(
                input,
                if account { "account" } else { "project" },
                id,
                item.parent,
            );
            doc.id = DocumentId::new(
                Origin::new(super::ORIGIN).expect("static origin"),
                NativeId::new(&format!("{scope}:{}", item.suffix))
                    .map_err(|e| ScanError::Callback(e.to_string()))?,
            );
            let missing = missing || item.missing_child;
            doc.metadata["identity"] = if missing {
                serde_json::json!({"kind":"input_scoped","input":input})
            } else {
                serde_json::json!({"kind":"provider_scoped"})
            };
            if let Some(id) = id {
                doc.metadata[if account {
                    "account_uuid"
                } else {
                    "project_uuid"
                }] = Json::String(id.to_owned());
            }
            if doc.purpose != DocumentPurpose::Context {
                let (at, accuracy) =
                    crate::time::parts(declared_time(|key| fields.get(key).cloned()).0);
                doc.set_time(at, accuracy)
                    .expect("decoded source time is valid");
            }
            if doc.purpose == DocumentPurpose::Instruction {
                doc.title = fields.str("name").map(str::to_owned);
            }
        }
        Ok(ScanStats {
            elements: stats.elements + self.metadata.len() as u64,
            ..stats
        })
    }
    fn begin(&mut self, length: u64, _: Option<&[u8]>) -> Result<Option<Step>, String> {
        let mut records = Records::default();
        for metadata in std::mem::take(&mut self.metadata) {
            records.fields(
                metadata.kind,
                &metadata.keys,
                if metadata.kind == "project" {
                    &PROJECT
                } else {
                    &ACCOUNT
                },
            );
            records.record(
                metadata.kind,
                &metadata
                    .diagnostic
                    .map_or(RecordOutcome::Used, RecordOutcome::skipped),
            );
        }
        Ok(Some(Step {
            output: Output::default(),
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
        let item = self
            .items
            .get_mut(self.at)
            .ok_or("missing sidecar frame context")?;
        self.at += 1;
        if item.kind == "project_document" {
            sink.records.fields(item.kind, &item.keys, &DOC);
        }
        let Some(mut doc) = item.document.take() else {
            sink.records.record(
                item.kind,
                &RecordOutcome::skipped(
                    item.skip_reason
                        .as_deref()
                        .unwrap_or("unrecognized sidecar record"),
                ),
            );
            return Ok(());
        };
        let blocks = if item.kind == "project_document" {
            let fields = fields::object(bytes)?;
            fields
                .get("content")
                .and_then(|raw| serde_json::from_str::<Json>(raw.get()).ok())
                .and_then(|content| document_blocks(&content))
        } else {
            serde_json::from_slice::<String>(bytes)
                .ok()
                .map(|text| vec![Block::Text { text }])
        };
        let Some(blocks) = blocks else {
            sink.records.record(
                item.kind,
                &RecordOutcome::skipped(if item.kind == "project_document" {
                    "no supplied text; unresolved/binary reference remains in the input"
                } else {
                    "expected text; original value remains in the input"
                }),
            );
            return Ok(());
        };
        doc.blocks.extend(blocks);
        sink.out.documents.push(DocumentRow {
            offset: 0,
            end: 0,
            document: doc,
        });
        sink.records.record(item.kind, &RecordOutcome::Used);
        Ok(())
    }
}

/// A project document's `content`: its text, or its parts in order, a text
/// part as text and any other part kept whole. `None` when it holds nothing.
fn document_blocks(content: &Json) -> Option<Vec<Block>> {
    let blocks = match content {
        Json::String(text) => vec![Block::Text { text: text.clone() }],
        Json::Array(parts) => parts
            .iter()
            .map(|part| match part {
                Json::String(text) => Block::Text { text: text.clone() },
                part => match part.get("text") {
                    Some(Json::String(text)) => Block::Text { text: text.clone() },
                    _ => Block::Other {
                        type_name: part
                            .get("type")
                            .and_then(Json::as_str)
                            .unwrap_or("content")
                            .to_owned(),
                        data: part.clone(),
                    },
                },
            })
            .collect(),
        _ => return None,
    };
    (!blocks.is_empty()).then_some(blocks)
}
