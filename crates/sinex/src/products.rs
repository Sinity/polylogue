//! Canonical, complete Polylogue facts built from parser output. The source
//! reading is part of every parsed fact; indexed summaries are never an input.
use crate::{Error, Result};
use polylogue_model::{Block, WrittenBy};
use polylogue_store::primary::products::Product;
use polylogue_store::user_writes::UserWrite;
use serde_json::{Value, json};

pub(crate) fn kind(product: &Product) -> &'static str {
    match product {
        Product::Input { .. } => "polylogue.input_observation",
        Product::Blob(_) => "polylogue.blob",
        Product::Session { .. } => "polylogue.session",
        Product::Message { .. } => "polylogue.message",
        Product::Document { .. } => "polylogue.document",
        Product::Event { .. } => "polylogue.event",
        Product::Link { .. } => "polylogue.link",
        Product::Background { .. } => "polylogue.background",
        Product::Repository { .. } => "polylogue.repository",
        Product::ParseProgress { .. } => "polylogue.parse_progress",
        Product::Request(request) => match &request.request.write {
            UserWrite::AddNote { .. }
            | UserWrite::SaveQuery { .. }
            | UserWrite::AnnotateQuery { .. }
            | UserWrite::Withdraw { .. } => "polylogue.note",
            UserWrite::SetSetting { .. } => "polylogue.setting",
            UserWrite::KeepResult { .. } => "polylogue.kept_result",
            UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) => "polylogue.delivery",
            UserWrite::DeliveryEvent { .. } => "polylogue.delivery_outcome",
            _ => "polylogue.request",
        },
    }
}

/// Source copies remain separate readings of one natural record. This retains
/// every source rank and provider attribute when another copy arrives.
pub(crate) fn key(product: &Product, occurrence: u64) -> Result<String> {
    let natural = match product {
        Product::Input { rows, .. } => json!([
            rows.input,
            rows.revision,
            rows.extent,
            rows.observation.as_ref().map(|observation| &observation.id)
        ]),
        Product::Blob(material) => json!(material.hash),
        Product::Session { facts, .. } => json!([facts.id]),
        Product::Message { facts, .. } => json!([facts.message.id, facts.offset]),
        Product::Document { facts, .. } => json!([facts.document.id, facts.offset]),
        Product::Event { id, .. } => json!([id]),
        Product::Link { facts, .. } => json!([facts.kind, facts.parent, facts.child]),
        Product::Background { facts, .. } => json!([facts.session, facts.task_id]),
        Product::Repository { facts, .. } => json!([facts.path]),
        Product::ParseProgress { reading, .. } => json!([reading.start]),
        Product::Request(request) => json!([request.request.id]),
        Product::Coverage(coverage) => json!([coverage.machine]),
        Product::WorkSnapshot { snapshot } => {
            json!([snapshot.session_id, snapshot.reducer_version])
        }
    };
    let source = product
        .reading()
        .map(|reading| json!([reading.input.id, reading.chain]));
    Ok(
        blake3::hash(&serde_json::to_vec(&(kind(product), source, natural))?)
            .to_hex()
            .to_string(),
    )
}

/// The full fact is authoritative; the common fields let Sinex join AI work
/// to its other domains without understanding Polylogue's internal model.
pub(crate) fn payload(product: &Product, occurrence: u64) -> Result<Value> {
    let mut payload = match product {
        Product::Session { facts, .. } => json!({
            "session": facts.id, "origin": facts.id.origin().as_str(), "kind": facts.kind,
            "title": facts.title.as_ref().map(|title| &title.text), "cwd": facts.cwd,
            "repo": facts.repo, "branch": facts.git_branch, "tool_version": facts.tool_version,
            "mode": facts.mode, "started": facts.started, "machine": Value::Null,
            "objectives": [], "git_identity": Value::Null, "terminal": Value::Null,
            "boot_id": Value::Null, "native_sources": [], "native_unavailable": [],
        }),
        Product::Message { facts, .. } => {
            let message = &facts.message;
            let text: Vec<_> = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            json!({
                "message": message.id, "session": message.session, "parent": message.parent,
                "role": message.role, "written_by": WrittenBy::of(message.role, message.prompt_kind),
                "prompt_kind": message.prompt_kind, "at": message.at, "model": message.model,
                "text": text.join("\n\n"), "usage": message.usage,
                "stop_reason": message.stop_reason, "duration_ms": message.duration_ms,
            })
        }
        Product::Document { facts, .. } => {
            json!({"document": facts.document.id, "at": facts.document.source_time})
        }
        Product::Event { facts, .. } => {
            json!({"session": facts.session, "event": facts.kind, "at": facts.at})
        }
        Product::Link { facts, .. } => json!({"session": facts.child, "parent": facts.parent}),
        Product::Background { facts, .. } => {
            json!({"session": facts.session, "task": facts.task_id})
        }
        Product::Repository { facts, .. } => json!({"path": facts.path}),
        Product::Request(prepared) => {
            let request = &prepared.request;
            let mut value =
                json!({"request": request.id, "at": request.at, "machine": request.machine});
            match &request.write {
                UserWrite::AddNote { note }
                | UserWrite::SaveQuery { note }
                | UserWrite::AnnotateQuery { note, .. } => {
                    value["note"] = request.id.clone().into();
                    value["kind"] = note.kind.as_str().into();
                    value["author"] = note.author.to_string().into();
                    value["value"] = serde_json::to_value(&note.value)?;
                    value["key"] = serde_json::to_value(&note.key)?;
                    value["targets"] = serde_json::to_value(&note.targets)?;
                    value["revises"] = serde_json::to_value(&note.revises)?;
                }
                UserWrite::Withdraw {
                    note,
                    author,
                    reason,
                } => {
                    value["note"] = request.id.clone().into();
                    value["kind"] = "withdrawal".into();
                    value["author"] = author.to_string().into();
                    value["withdraws"] = note.clone().into();
                    value["value"] = serde_json::to_value(reason)?;
                }
                UserWrite::SetSetting {
                    key,
                    value: setting,
                    author,
                    machine,
                } => {
                    value["setting"] = request.id.clone().into();
                    value["key"] = key.clone().into();
                    value["value"] = setting.clone().into();
                    value["author"] = author.to_string().into();
                    value["machine"] = machine.clone().unwrap_or_else(|| "*".into()).into();
                }
                UserWrite::Deliver(delivery) | UserWrite::DeliveredAdvice(delivery) => {
                    value["delivery"] = request.id.clone().into();
                    value["recipient_kind"] = delivery.recipient_kind.clone().into();
                    value["recipient_id"] = serde_json::to_value(&delivery.recipient_id)?;
                    value["sender"] = delivery.sender.clone().into();
                    value["event"] = delivery.event.clone().into();
                    value["cwd"] = serde_json::to_value(&delivery.cwd)?;
                    value["content_hash"] = blake3::hash(delivery.text.as_bytes())
                        .to_hex()
                        .to_string()
                        .into();
                    value["bytes"] = delivery.text.len().into();
                    value["refs"] = serde_json::to_value(&delivery.refs)?;
                }
                UserWrite::DeliveryEvent {
                    delivery,
                    what,
                    detail,
                } => {
                    value["delivery"] = delivery.clone().into();
                    value["what"] = what.as_str().into();
                    value["detail"] = serde_json::to_value(detail)?;
                }
                UserWrite::KeepResult {
                    name,
                    query,
                    rows_json,
                    read_at,
                } => {
                    value["result"] = request.id.clone().into();
                    value["name"] = serde_json::to_value(name)?;
                    value["query"] = query.clone().into();
                    value["rows"] = serde_json::from_str(rows_json)?;
                    value["read_at"] = serde_json::from_str(read_at)?;
                }
                _ => {}
            }
            value
        }
        Product::Coverage(_) => json!({}),
        Product::WorkSnapshot { snapshot } => json!({
            "session":snapshot.session_id,"reducer_version":snapshot.reducer_version,"source_position":snapshot.position,
            "complete":true,"repo":snapshot.session.as_ref().and_then(|session|session.repo.as_ref()),"branch":snapshot.session.as_ref().and_then(|session|session.git_branch.as_ref()),"kind":snapshot.session.as_ref().map(|session|session.kind),"mode":snapshot.session.as_ref().and_then(|session|session.mode.as_ref()),
            "machine":Value::Null,"objectives":[],"git_identity":Value::Null,"terminal":Value::Null,"boot_id":Value::Null,
            "native_sources":[],"native_unavailable":[],
        }),
        Product::Input { rows, .. } => json!({"input": rows.input}),
        Product::Blob(material) => json!({"blob": material.hash}),
        Product::ParseProgress { reading, .. } => json!({"input": reading.input.id}),
    };
    payload["record_key"] = key(product, occurrence)?.into();
    payload["record"] = serde_json::to_value(product)?;
    Ok(payload)
}

pub(crate) fn decode(event_type: &str, payload: Value) -> Result<Product> {
    let product: Product =
        serde_json::from_value(payload.get("record").cloned().ok_or_else(|| {
            Error::Invalid(format!("confirmed {event_type} has no complete facts"))
        })?)?;
    if let Product::WorkSnapshot { snapshot } = &product {
        snapshot
            .validate()
            .map_err(|error| Error::Invalid(error.into()))?;
        if payload.get("session") != Some(&serde_json::to_value(&snapshot.session_id)?)
            || payload.get("reducer_version").and_then(Value::as_u64)
                != Some(snapshot.reducer_version as u64)
            || payload.get("source_position").and_then(Value::as_i64) != Some(snapshot.position)
            || payload.get("complete").and_then(Value::as_bool) != Some(true)
        {
            return Err(Error::Invalid(
                "work snapshot wrapper differs from its complete facts".into(),
            ));
        }
    }
    if kind(&product) != event_type {
        return Err(Error::Invalid(format!(
            "confirmed {event_type} carries another product kind"
        )));
    }
    Ok(product)
}

#[derive(schemars::JsonSchema)]
#[allow(dead_code)]
struct WorkSnapshotMetadata {
    session: polylogue_model::SessionId,
    reducer_version: u32,
    source_position: i64,
    complete: bool,
    repo: Option<String>,
    branch: Option<String>,
    kind: Option<polylogue_model::SessionKind>,
    mode: Option<String>,
    machine: Option<String>,
    objectives: Vec<crate::kinds::Objective>,
    git_identity: Option<crate::kinds::GitIdentity>,
    terminal: Option<crate::kinds::TerminalIdentity>,
    boot_id: Option<String>,
    native_sources: Vec<crate::kinds::RecordRef>,
    native_unavailable: Vec<String>,
}

fn reachable_definitions(root: &Value, definitions: &Value) -> Value {
    fn references(value: &Value, found: &mut std::collections::BTreeSet<String>) {
        match value {
            Value::Object(object) => {
                if let Some(reference) = object
                    .get("$ref")
                    .and_then(Value::as_str)
                    .and_then(|reference| reference.strip_prefix("#/$defs/"))
                {
                    found.insert(reference.into());
                }
                for value in object.values() {
                    references(value, found);
                }
            }
            Value::Array(values) => {
                for value in values {
                    references(value, found);
                }
            }
            _ => {}
        }
    }
    let mut pending = std::collections::BTreeSet::new();
    references(root, &mut pending);
    let mut retained = serde_json::Map::new();
    while let Some(name) = pending.pop_first() {
        if retained.contains_key(&name) {
            continue;
        }
        let definition = definitions
            .get(&name)
            .expect("generated schema reference exists")
            .clone();
        references(&definition, &mut pending);
        retained.insert(name, definition);
    }
    Value::Object(retained)
}

/// Kind declarations validate the complete serde domain model. The outer
/// fields are the projection Sinex's cross-domain readers consume.
pub(crate) fn declarations(producer: &str) -> Vec<sinex_model::KindDeclaration> {
    use sinex_model::kinds::{FieldHint, KeyRule, Normalization, RemovedRule, Search, TimeRule};
    use sinex_model::{Class, KindDeclaration, KindRef, ReplaceRule, TimeQuality};
    const KINDS: &[(&str, &str)] = &[
        ("polylogue.input_observation", "input"),
        ("polylogue.blob", "blob"),
        ("polylogue.session", "session"),
        ("polylogue.message", "message"),
        ("polylogue.document", "document"),
        ("polylogue.event", "event"),
        ("polylogue.link", "link"),
        ("polylogue.background", "background"),
        ("polylogue.repository", "repository"),
        ("polylogue.parse_progress", "parse_progress"),
        ("polylogue.note", "request"),
        ("polylogue.setting", "request"),
        ("polylogue.kept_result", "request"),
        ("polylogue.delivery", "request"),
        ("polylogue.delivery_outcome", "request"),
        ("polylogue.request", "request"),
        ("polylogue.coverage", "coverage"),
        ("polylogue.work_snapshot", "work_snapshot"),
    ];
    let complete =
        serde_json::to_value(schemars::schema_for!(Product)).expect("product schema serializes");
    KINDS
        .iter()
        .map(|(name, product)| {
            let mut facts = complete.clone();
            let definitions = facts
                .as_object_mut()
                .expect("object schema")
                .remove("$defs")
                .unwrap_or_else(|| json!({}));
            let variants = facts["oneOf"]
                .as_array_mut()
                .expect("tagged product variants");
            variants.retain(|variant| variant["properties"]["product"]["const"] == *product);
            assert_eq!(variants.len(), 1, "one complete product schema per kind");
            let definitions = reachable_definitions(&facts, &definitions);
            let mut schema = json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object", "properties": {
                    "record_key": {"type":"string", "minLength":1}, "record":facts,
                },
                "required":["record_key","record"], "$defs":definitions,
            });
            if *product == "work_snapshot" {
                let metadata = serde_json::to_value(schemars::schema_for!(WorkSnapshotMetadata))
                    .expect("metadata schema serializes");
                let object = schema["properties"]
                    .as_object_mut()
                    .expect("schema properties");
                object.extend(
                    metadata["properties"]
                        .as_object()
                        .expect("metadata properties")
                        .clone(),
                );
                let required = schema["required"].as_array_mut().expect("required fields");
                required.extend(
                    metadata["required"]
                        .as_array()
                        .expect("metadata required fields")
                        .clone(),
                );
                if let Some(definitions) = metadata["$defs"].as_object() {
                    schema["$defs"]
                        .as_object_mut()
                        .expect("definitions")
                        .extend(definitions.clone());
                }
                schema["additionalProperties"] = false.into();
            }
            KindDeclaration {
                kind: KindRef {
                    name: (*name).into(),
                    version: 3,
                },
                declaring_producer: producer.into(),
                payload_schema: Some(schema),
                normalization: Normalization::Json,
                key_rule: Some(KeyRule::Fields {
                    fields: if *product == "work_snapshot" {
                        vec!["session".into(), "reducer_version".into()]
                    } else {
                        vec!["record_key".into()]
                    },
                }),
                replace_rule: Some(if *product == "request" || *product == "blob" {
                    ReplaceRule::SuppressRepeat
                } else {
                    ReplaceRule::ReplaceWhenNewer
                }),
                time_rule: Some(if *product == "work_snapshot" {
                    TimeRule::Unknown
                } else {
                    TimeRule::FirstPresent {
                        fields: vec!["at".into(), "started".into()],
                        quality: TimeQuality::Intrinsic,
                    }
                }),
                field_hints: [
                    "record", "text", "value", "targets", "query", "rows", "refs",
                ]
                .into_iter()
                .map(|field| FieldHint {
                    field: format!("/{field}"),
                    class: Class::Guarded,
                })
                .collect(),
                removed_rule: Some(RemovedRule::Never),
                keep_known: Vec::new(),
                owned_blobs: Vec::new(),
                search: Search::None,
            }
        })
        .collect()
}

/// Build each event from its complete fact and its confirmed material anchor.
pub(crate) fn events(
    writes: &[polylogue_store::primary::products::ProductWrite],
) -> Result<Vec<sinex_protocol::ProducerEvent>> {
    use polylogue_store::primary::products::ProductWrite;
    use sinex_model::{Class, MaterialId, RecordLocation};
    writes
        .iter()
        .map(|write| {
            let mut payload = payload(&write.product, write.output_index)?;
            let mut source = write.location.clone();
            let session = match &write.product {
                Product::Session { facts, .. } => Some(facts),
                Product::WorkSnapshot { snapshot } => snapshot.session.as_ref(),
                _ => None,
            };
            if let Some(facts) = session {
                let mut native = crate::native::Native::new(facts.repo.clone());
                if let Product::WorkSnapshot { snapshot } = &write.product {
                    for evidence in &snapshot.native {
                        native.observe(
                            &evidence.data,
                            crate::kinds::RecordRef {
                                material: MaterialId(evidence.material.parse().map_err(
                                    |error| Error::Invalid(format!("source material: {error}")),
                                )?),
                                anchor: evidence.anchor,
                            },
                        );
                    }
                }
                for ProductWrite {
                    product, location, ..
                } in writes
                {
                    if let Product::Event { facts: event, .. } = product
                        && event.session.as_ref() == Some(&facts.id)
                        && let Some(location) = location
                    {
                        native.observe(
                            &event.data,
                            crate::kinds::RecordRef {
                                material: MaterialId(location.material.parse().map_err(
                                    |error| Error::Invalid(format!("source material: {error}")),
                                )?),
                                anchor: location.anchor,
                            },
                        );
                    }
                }
                let native = native.finish();
                if let Some(first) = native.sources.first() {
                    source = Some(polylogue_store::primary::products::SourceLocation {
                        material: first.material.to_string(),
                        anchor: first.anchor,
                    });
                }
                payload["repo"] = serde_json::to_value(native.repo)?;
                payload["objectives"] = serde_json::to_value(native.objectives)?;
                payload["git_identity"] = serde_json::to_value(native.git_identity)?;
                payload["terminal"] = serde_json::to_value(native.terminal)?;
                payload["boot_id"] = serde_json::to_value(native.boot_id)?;
                payload["native_sources"] = serde_json::to_value(native.sources)?;
                payload["native_unavailable"] = serde_json::to_value(native.unavailable)?;
                payload["machine"] = serde_json::to_value(&write.machine)?;
            }
            let location = source
                .map(|source| -> Result<_> {
                    Ok(RecordLocation {
                        material: MaterialId(source.material.parse().map_err(|error| {
                            Error::Invalid(format!("source material: {error}"))
                        })?),
                        anchor: source.anchor,
                    })
                })
                .transpose()?;
            let output_index = write.output_index;
            Ok(sinex_protocol::ProducerEvent {
                kind: crate::kinds::kind_reference(kind(&write.product))?,
                payload,
                location,
                output_index,
                parents: write
                    .parents
                    .iter()
                    .map(|parent| {
                        parent
                            .parse()
                            .map(sinex_model::EventId)
                            .map_err(|error| Error::Invalid(format!("parent event: {error}")))
                    })
                    .collect::<Result<Vec<_>>>()?,
                reading_order: None,
                class: Class::Open,
            })
        })
        .collect()
}
