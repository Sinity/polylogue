//! Account/project text is staged as documents, never conversation turns.
use super::fields::{self, Field};
use crate::document::Sink;
use crate::export::{self, ExportParser, ExportState, Input, ItemSink, ScanError, ScanStats};
use crate::fields::{Fields, Rule};
use crate::parse::{Output, Parse, Step};
use crate::{Disposition, Records};
use polylogue_model::{
    Block, Document, DocumentId, DocumentPurpose, DocumentRow, Json, NativeId, Origin, Timestamp,
};
use serde_json::value::RawValue;
use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;

#[derive(Clone, Copy)]
enum Shape {
    Memories,
    Projects,
    Project,
}
fn shape(path: &Path) -> Option<Shape> {
    match path.file_name()?.to_str()? {
        "memories.json" => Some(Shape::Memories),
        "projects.json" => Some(Shape::Projects),
        _ if path.extension().is_some_and(|e| e == "json")
            && path.parent()?.file_name().is_some_and(|p| p == "projects") =>
        {
            Some(Shape::Project)
        }
        _ => None,
    }
}
pub(super) fn recognize(path: &Path, head: &[u8]) -> bool {
    shape(path).is_some_and(|shape| {
        matches!(
            (shape, crate::chatgpt::first_byte(head)),
            (Shape::Memories | Shape::Projects, Some(b'[')) | (Shape::Project, Some(b'{'))
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
    diagnostic: Option<String>,
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
static ACCOUNT: Fields = Fields::new(&[
    ("account_uuid", Rule::Read),
    ("conversations_memory", Rule::Read),
    ("project_memories", Rule::Read),
    ("created_at", Rule::Read),
    ("updated_at", Rule::Read),
]);
static PROJECT: Fields = Fields::new(&[
    ("uuid", Rule::Read),
    ("name", Rule::Read),
    (
        "description",
        Rule::Kept("project descriptor, not supplied document text"),
    ),
    ("is_private", Rule::Kept("project visibility")),
    ("is_starter_project", Rule::Read),
    ("prompt_template", Rule::Read),
    ("docs", Rule::Read),
    ("created_at", Rule::Read),
    ("updated_at", Rule::Read),
    (
        "creator",
        Rule::Kept("project creator is not text authorship"),
    ),
]);
static DOC: Fields = Fields::new(&[
    ("uuid", Rule::Read),
    ("filename", Rule::Read),
    ("content", Rule::Read),
    ("created_at", Rule::Read),
    ("updated_at", Rule::Read),
    (
        "file_type",
        Rule::Kept("reference descriptor; original bytes are not supplied"),
    ),
    (
        "file_size",
        Rule::Kept("reference descriptor; original bytes are not supplied"),
    ),
    ("url", Rule::Kept("reference retained without fetching")),
]);

fn names(fields: &BTreeMap<String, Field<'_>>) -> Json {
    Json::Object(fields.keys().map(|key| (key.clone(), Json::Null)).collect())
}
fn native(fields: &BTreeMap<String, Field<'_>>, name: &str) -> Option<String> {
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
fn time(fields: &BTreeMap<String, Field<'_>>) -> (Option<Timestamp>, bool) {
    for field in ["updated_at", "created_at"] {
        if let Some(raw) = fields.get(field) {
            let value: Json = serde_json::from_str(raw.raw.get()).unwrap_or(Json::Null);
            if value.is_null() {
                continue;
            }
            // These sidecar fields declare offset-qualified ISO time, not epochs.
            let time = value.as_str().and_then(Timestamp::parse_rfc3339);
            return (time, time.is_none());
        }
    }
    (None, false)
}
fn source_time(fields: &Json) -> (Option<Timestamp>, bool) {
    for key in ["updated_at", "created_at"] {
        if let Some(value) = fields.get(key).filter(|value| !value.is_null()) {
            let time = value.as_str().and_then(Timestamp::parse_rfc3339);
            return (time, time.is_none());
        }
    }
    (None, false)
}
fn document(
    native: String,
    purpose: DocumentPurpose,
    title: Option<String>,
    source_time: Option<Timestamp>,
    source_kind: &str,
    input: &str,
    fallback: bool,
) -> Result<Document, String> {
    let id = DocumentId::new(
        Origin::new(super::ORIGIN).map_err(|e| e.to_string())?,
        NativeId::new(&native).map_err(|e| e.to_string())?,
    );
    let mut document = Document::new(id, purpose);
    document.title = title;
    document.source_time = source_time;
    document.metadata = serde_json::json!({"source_kind":source_kind,"identity":if fallback {serde_json::json!({"kind":"input_scoped","input":input})}else{serde_json::json!({"kind":"provider_scoped"})}});
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
        let (purpose, suffix, missing_child, kind, title, at, keys, diagnostic) = match field {
            "conversations_memory" => (
                DocumentPurpose::Memory,
                "memory".into(),
                false,
                "memory_text",
                None,
                None,
                Json::Null,
                None,
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
                    None,
                    None,
                    Json::Null,
                    None,
                )
            }
            "prompt_template" => (
                DocumentPurpose::Instruction,
                "instructions".into(),
                false,
                "instruction_text",
                None,
                None,
                Json::Null,
                None,
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
                                diagnostic: Some(reason),
                            },
                        );
                    }
                };
                let id = native(&fields, "uuid");
                let (suffix, missing) = scope(input, "doc", id.as_deref(), range.start);
                let (at, bad) = time(&fields);
                (
                    DocumentPurpose::Context,
                    suffix,
                    missing,
                    "project_document",
                    fields::text(&fields, "filename"),
                    at,
                    names(&fields),
                    bad.then(|| "unreadable document time; original value retained".into()),
                )
            }
            _ => return Err("unknown sidecar framing field".into()),
        };
        let mut doc = document(
            format!("~input:{}:at:{parent}:{suffix}", component(input)),
            purpose,
            title,
            at,
            match kind {
                "project_document" => "project_document",
                "instruction_text" => "project_instructions",
                _ => {
                    if field == "project_memories" {
                        "project_memory"
                    } else {
                        "account_memory"
                    }
                }
            },
            input,
            true,
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
                diagnostic,
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
                "sidecar parsing needs its retained input identity".into(),
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
            !matches!(self.shape, Shape::Project),
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
            let keys = Json::Object(
                fields
                    .as_object()
                    .ok_or_else(|| ScanError::Callback("sidecar metadata is not an object".into()))?
                    .keys()
                    .map(|key| (key.clone(), Json::Null))
                    .collect(),
            );
            let bad_time = source_time(fields).1;
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
                .get(if account { "account_uuid" } else { "uuid" })
                .and_then(Json::as_str)
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
                doc.source_time = source_time(fields).0;
            }
            if doc.purpose == DocumentPurpose::Instruction {
                doc.title = fields.get("name").and_then(Json::as_str).map(str::to_owned);
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
                    .map_or(Disposition::Materialized, Disposition::unknown),
            );
        }
        Ok(Some(Step {
            output: Output::default(),
            accounting: records.finish().map_err(|e| e.to_string())?,
            start: 0,
            consumed: length,
            snapshot: None,
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
                &Disposition::unknown(
                    item.diagnostic
                        .as_deref()
                        .unwrap_or("uninterpreted sidecar record"),
                ),
            );
            return Ok(());
        };
        let text = if item.kind == "project_document" {
            let fields = fields::object(bytes)?;
            fields::text(&fields, "content")
        } else {
            serde_json::from_slice::<String>(bytes).ok()
        };
        let Some(text) = text else {
            sink.records.record(
                item.kind,
                &Disposition::unknown(if item.kind == "project_document" {
                    "no supplied text; unresolved/binary reference remains in the input"
                } else {
                    "expected text; original value remains in the input"
                }),
            );
            return Ok(());
        };
        doc.blocks.push(Block::Text { text });
        sink.out.documents.push(DocumentRow {
            offset: 0,
            end: 0,
            document: doc,
        });
        sink.records.record(
            item.kind,
            &item
                .diagnostic
                .as_deref()
                .map_or(Disposition::Materialized, Disposition::unknown),
        );
        Ok(())
    }
}
