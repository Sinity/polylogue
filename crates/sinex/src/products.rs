//! Complete Polylogue facts built from parser output. The source
//! reading is part of every parsed fact; indexed summaries are never an input.
use crate::{Error, Result};
use polylogue_model::user::UserWrite;
use polylogue_model::{Block, WrittenBy};
use polylogue_store::primary::products::Product;
use serde_json::{Value, json};
mod source_formats;
pub(crate) use source_formats::decode_source;
#[cfg(test)]
pub(crate) use source_formats::historical_source_declarations;

pub(crate) fn require_source(product: &Product) -> Result<()> {
    if product
        .reading()
        .is_some_and(|reading| reading.delivery.is_some())
    {
        return Err(Error::Refused {
            name: kind(product).into(),
            reason: "export products require their bound immutable reading key".into(),
        });
    }
    Ok(())
}

pub(crate) fn require_export(
    product: &Product,
    key: &polylogue_store::primary::ExportReadingKey,
) -> Result<()> {
    key.validate()?;
    require_reading(product, key)?;
    if product
        .reading()
        .is_none_or(|reading| reading.delivery.is_none())
    {
        return Err(Error::Invalid(
            "export product has no delivery identity".into(),
        ));
    }
    Ok(())
}

pub(crate) fn require_reading(
    product: &Product,
    key: &polylogue_store::primary::ExportReadingKey,
) -> Result<()> {
    let reading = product.reading().ok_or_else(|| {
        Error::Invalid("export custody contains a product without a reading".into())
    })?;
    if reading.input.id != key.input.id
        || reading.input.format != key.input.format
        || reading.chain != key.chain
        || reading.reading_version != key.reading_version
        || reading
            .source_length
            .is_some_and(|length| length != key.length)
        || (reading.delivery.is_some() && reading.source_length != Some(key.length))
    {
        return Err(Error::Invalid(
            "export product differs from its immutable reading key".into(),
        ));
    }
    if let Some(delivery) = &reading.delivery {
        delivery
            .validate()
            .map_err(|error| Error::Invalid(error.to_string()))?;
    }
    Ok(())
}

pub(crate) fn kind(product: &Product) -> &'static str {
    match product {
        Product::AttachmentBinding { .. } => "polylogue.attachment_binding",
        Product::Input { .. } => crate::kinds::INPUT_KIND,
        Product::Blob(_) => "polylogue.blob",
        Product::Session { .. } => "polylogue.session",
        Product::Message { .. } => "polylogue.message",
        Product::Document { .. } => "polylogue.document",
        Product::Fragment { .. } => "polylogue.fragment",
        Product::Event { .. } => "polylogue.event",
        Product::Link { .. } => "polylogue.link",
        Product::Background { .. } => "polylogue.background",
        Product::Repository { .. } => "polylogue.repository",
        Product::ParseProgress { .. } => "polylogue.parse_progress",
        Product::RestoredRequest { .. } => "polylogue.request_restore",
        Product::RestoredAssertion { .. } => "polylogue.assertion_restore",
        Product::Query { .. } => "polylogue.query",
        Product::QueryRun { .. } => "polylogue.query_run",
        Product::QueryRunOutcome { .. } => "polylogue.query_run_outcome",
        Product::Coverage(_) => "polylogue.coverage",
        Product::Request(request) => match &request.request.write {
            UserWrite::Assert { .. }
            | UserWrite::AssertQuery { .. }
            | UserWrite::ReviseAssertions { .. }
            | UserWrite::WithdrawAssertions { .. }
            | UserWrite::SaveQuery { .. }
            | UserWrite::SetSetting { .. } => "polylogue.assertion",
            UserWrite::KeepResult { .. } => "polylogue.kept_result",
            UserWrite::Deliver(_) | UserWrite::DeliveredAdvice(_) => "polylogue.delivery",
            UserWrite::DeliveryEvent { .. } => "polylogue.delivery_outcome",
            _ => "polylogue.request",
        },
    }
}

/// Source copies remain separate readings of one natural record. This retains
/// every source rank and provider attribute when another copy arrives.
pub(crate) fn key(product: &Product, _occurrence: u64) -> Result<String> {
    let natural = match product {
        Product::AttachmentBinding { record } => json!([record.selection()]),
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
        Product::Fragment { facts, .. } => json!([facts.record.reference, facts.offset]),
        Product::Event { id, .. } => json!([id]),
        // Link assertions accumulate: a later source record may omit a branch
        // point which an earlier record supplied. Keep both contributions for
        // the canonical relation merger rather than replacing one with the other.
        Product::Link { facts, .. } => json!([facts]),
        Product::Background { facts, .. } => json!([facts.session, facts.task_id]),
        Product::Repository { facts, .. } => json!([facts.path]),
        Product::ParseProgress { reading, .. } => json!([reading.start]),
        Product::Request(request) => json!([request.request.id]),
        Product::RestoredRequest { prepared } => json!([prepared.request.id]),
        Product::RestoredAssertion { fact } => json!([fact.root, fact.id]),
        Product::Query { object } => json!([object.id]),
        Product::QueryRun { run } => json!([run.id]),
        Product::QueryRunOutcome { outcome } => json!([outcome.run]),
        Product::Coverage(coverage) => json!([coverage.machine]),
    };
    let source = product.reading().map(|reading| match &reading.delivery {
        Some(delivery) => json!([
            reading.input.id,
            reading.chain,
            [
                "export_delivery",
                delivery.version,
                delivery.ordinal,
                delivery.prefix_hash,
                reading.source_length
            ]
        ]),
        None => json!([reading.input.id, reading.chain]),
    });
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
        Product::Fragment { facts, .. } => json!({"fragment": facts.record.reference}),
        Product::Event { facts, .. } => {
            json!({"session": facts.session, "event": facts.kind, "at": facts.at})
        }
        Product::Link { facts, .. } => json!({"session": facts.child, "parent": facts.parent}),
        Product::Background { facts, .. } => {
            json!({"session": facts.session, "task": facts.task_id})
        }
        Product::Repository { facts, .. } => json!({"path": facts.path}),
        Product::RestoredRequest { prepared } => json!({
            "request": prepared.request.id,
            "at": prepared.request.at,
            "machine": prepared.request.machine
        }),
        Product::RestoredAssertion { fact } => json!({
            "assertion": format!("assertion:{}@{}", fact.root, fact.id),
            "at": fact.created_at,
            "machine": fact.machine
        }),
        Product::Request(prepared) => {
            let request = &prepared.request;
            let mut value =
                json!({"request": request.id, "at": request.at, "machine": request.machine});
            match &request.write {
                UserWrite::Assert { .. }
                | UserWrite::AssertQuery { .. }
                | UserWrite::ReviseAssertions { .. }
                | UserWrite::WithdrawAssertions { .. }
                | UserWrite::SaveQuery { .. }
                | UserWrite::SetSetting { .. } => {
                    value["assertions"] = serde_json::to_value(&prepared.effect.assertions)?;
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
        Product::Query { object } => json!({"question": object.id}),
        Product::QueryRun { run } => json!({
            "run": run.id, "question": run.question, "at": run.at,
            "machine": run.machine, "binding_hash": run.binding_hash()?, "run_hash": run.record_hash()?,
        }),
        Product::QueryRunOutcome { outcome } => json!({
            "run": outcome.run, "binding_hash": outcome.binding_hash, "run_hash": outcome.run_hash, "at": outcome.at,
        }),
        Product::AttachmentBinding { record } => {
            json!({"message":record.context().message,"binding":record.selection()})
        }
        Product::Coverage(_) => json!({}),
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
    if let Some(reading) = product.reading()
        && let Some(delivery) = &reading.delivery
    {
        delivery
            .validate()
            .map_err(|error| Error::Invalid(error.to_string()))?;
        if reading.source_length.is_none() {
            return Err(Error::Invalid(
                "export reading has no actual source length".into(),
            ));
        }
    }
    if matches!(&product, Product::ParseProgress { stopped_attempt:Some(_),reading,.. } if !matches!(reading.progress, polylogue_model::batch::StepProgress::Failed { .. } | polylogue_model::batch::StepProgress::Refused { .. }))
    {
        return Err(Error::Invalid(
            "only stopped progress has a stopped attempt".into(),
        ));
    }
    if kind(&product) != event_type {
        return Err(Error::Invalid(format!(
            "confirmed {event_type} carries another product kind"
        )));
    }
    Ok(product)
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
        ("polylogue.attachment_binding", "attachment_binding"),
        (crate::kinds::INPUT_KIND, "input"),
        ("polylogue.blob", "blob"),
        ("polylogue.session", "session"),
        ("polylogue.message", "message"),
        ("polylogue.document", "document"),
        ("polylogue.fragment", "fragment"),
        ("polylogue.event", "event"),
        ("polylogue.link", "link"),
        ("polylogue.background", "background"),
        ("polylogue.repository", "repository"),
        ("polylogue.parse_progress", "parse_progress"),
        ("polylogue.assertion", "request"),
        ("polylogue.kept_result", "request"),
        ("polylogue.delivery", "request"),
        ("polylogue.delivery_outcome", "request"),
        ("polylogue.request", "request"),
        ("polylogue.request_restore", "restored_request"),
        ("polylogue.assertion_restore", "restored_assertion"),
        ("polylogue.query", "query"),
        ("polylogue.query_run", "query_run"),
        ("polylogue.query_run_outcome", "query_run_outcome"),
        ("polylogue.coverage", "coverage"),
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
            let schema = json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object", "properties": {
                    "record_key": {"type":"string", "minLength":1}, "record":facts,
                    "text": {"type":"string"},
                    "query": {"type":"string"}, "rows": {},
                    "refs": {"type":"array", "items":{"type":"string"}},
                    "at": {"type":["string","null"]},
                    "started": {"type":["string","null"]},
                },
                "required":["record_key","record"], "$defs":definitions,
            });
            KindDeclaration {
                kind: KindRef {
                    name: (*name).into(),
                    version: if *name == "polylogue.assertion" {
                        2
                    } else {
                        match *product {
                            // A product holding a parser reading moved to one
                            // progress value in place of two flags.
                            "message" => 6,
                            "request" | "restored_request" => 6,
                            "session" | "document" | "event" | "link" | "background"
                            | "repository" => 5,
                            "parse_progress" => 6,
                            "input" => 4,
                            "fragment" => 3,
                            "query_run" => 4,
                            "restored_history" | "restored_assertion" | "attachment_binding"
                            | "query_run_outcome" => 1,
                            "query" => 2,
                            _ => 3,
                        }
                    },
                },
                declaring_producer: producer.into(),
                payload_schema: Some(schema),
                normalization: Normalization::Json,
                key_rule: Some(KeyRule::Fields {
                    fields: vec!["record_key".into()],
                }),
                replace_rule: Some(
                    if matches!(
                        *product,
                        "attachment_binding"
                            | "request"
                            | "restored_request"
                            | "restored_assertion"
                            | "blob"
                            | "query"
                            | "query_run"
                            | "query_run_outcome"
                    ) {
                        ReplaceRule::SuppressRepeat
                    } else {
                        ReplaceRule::ReplaceWhenNewer
                    },
                ),
                time_rule: Some(if *product == "attachment_binding" {
                    TimeRule::Unknown
                } else {
                    TimeRule::FirstPresent {
                        fields: vec!["at".into(), "started".into()],
                        quality: TimeQuality::Intrinsic,
                    }
                }),
                field_hints: ["record", "text", "query", "rows", "refs"]
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
                Product::Session { facts, .. } => Some((&facts.id, facts.repo.clone())),
                _ => None,
            };
            if let Some((session, repo)) = session {
                let mut native = crate::native::Native::new(repo);
                for ProductWrite {
                    product, location, ..
                } in writes
                {
                    if let Product::Event { facts: event, .. } = product
                        && event.session.as_ref() == Some(session)
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
