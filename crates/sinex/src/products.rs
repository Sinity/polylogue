//! Complete Polylogue facts built from parser output. The source
//! reading is part of every parsed fact; indexed summaries are never an input.
use crate::{Error, Result};
use polylogue_model::user::UserWrite;
use polylogue_store::primary::products::Product;
use serde_json::{Value, json};
mod ai;

/// Decode a confirmed fact whose kind definition is the one Polylogue declares.
pub(crate) fn decode_source(
    kind: &sinex_model::KindRef,
    definition_hash: &str,
    payload: Value,
) -> Result<Product> {
    match crate::kinds::declared_definition(kind) {
        Some(current) if current == definition_hash => decode(kind, payload),
        Some(_) => Err(Error::Invalid(
            "a confirmed event's kind definition differs from Polylogue's".into(),
        )),
        None => Err(Error::Invalid(format!(
            "unsupported Polylogue kind {}@{}",
            kind.name, kind.version
        ))),
    }
}

pub(crate) fn require_source(product: &Product) -> Result<()> {
    if matches!(product, Product::Interpreted { .. }) {
        return Err(Error::Invalid(
            "native eligibility metadata is not a domain fact".into(),
        ));
    }
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
        Product::Interpreted { .. } => "polylogue.interpreted",
        Product::AttachmentBinding { .. } => "polylogue.attachment_binding",
        Product::Input { .. } => crate::kinds::INPUT_KIND,
        Product::Blob(_) => "polylogue.blob",
        Product::Session { .. } => "polylogue.session",
        Product::Message { .. } => "polylogue.message",
        Product::ToolCall { .. } => "polylogue.tool_call",
        Product::Document { .. } => "polylogue.document",
        Product::Fragment { .. } => "polylogue.fragment",
        Product::Event { .. } => "polylogue.event",
        Product::Link { .. } => "polylogue.link",
        Product::PrefixCopy { .. } => "polylogue.prefix_copy",
        Product::PrefixMember { .. } => "polylogue.prefix_member",
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
        Product::Interpreted { record, .. } => json!([record.reference, record.value_hash]),
        Product::AttachmentBinding { record } => json!([record.selection()]),
        Product::Input { rows, .. } => json!([
            rows.input,
            rows.revision,
            rows.extents,
            rows.observation.as_ref().map(|observation| &observation.id)
        ]),
        Product::Blob(material) => json!(material.hash),
        Product::Session { facts, .. } => json!([facts.id]),
        Product::Message { facts, .. } => json!([facts.message.id, facts.offset]),
        Product::ToolCall { facts, .. } => json!([facts.message, facts.block]),
        Product::Document { facts, .. } => json!([facts.document.id, facts.offset]),
        Product::Fragment { facts, .. } => json!([facts.record.reference, facts.offset]),
        Product::Event { id, .. } => json!([id]),
        // Link assertions accumulate: a later source record may omit a branch
        // point which an earlier record supplied. Keep both contributions for
        // the canonical relation merger rather than replacing one with the other.
        Product::Link { facts, .. } => json!([facts]),
        Product::PrefixCopy { facts, .. } => json!([facts]),
        Product::PrefixMember { facts, .. } => json!([facts]),
        Product::Background { facts, .. } => json!([facts.session, facts.task_id]),
        Product::Repository { facts, .. } => json!([facts.path]),
        Product::ParseProgress { reading, .. } => json!([reading.start]),
        Product::Request(request) => json!([request.request.id]),
        Product::RestoredRequest { prepared } => json!([prepared.request.id]),
        Product::RestoredAssertion { fact, import } => match import {
            None => json!([fact.root, fact.id]),
            Some(i) => json!([fact.root, fact.id, i.input, i.chain, i.machine, i.actor]),
        },
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
    if let Product::ToolCall { reading, facts } = product {
        return Ok(serde_json::to_value(ai::ToolFields {
            record_key: key(product, occurrence)?,
            source: reading,
            call: facts,
            context: Default::default(),
        })?);
    }
    let mut payload = match product {
        Product::Interpreted { .. } => unreachable!("eligibility is local metadata"),
        Product::ToolCall { .. } => unreachable!("handled as a native fact"),
        Product::Session { facts, .. } => serde_json::to_value(ai::SessionFields::from(facts))?,
        Product::Message { facts, .. } => {
            serde_json::to_value(ai::MessageFields::from(&facts.message))?
        }
        Product::Document { facts, .. } => {
            json!({"document": facts.document.id, "at": facts.document.source_time})
        }
        Product::Fragment { facts, .. } => json!({"fragment": facts.record.reference}),
        Product::Event { facts, .. } => {
            json!({"session": facts.session, "event": facts.kind, "at": facts.at})
        }
        Product::Link { facts, .. } => json!({"session": facts.child, "parent": facts.parent}),
        Product::PrefixCopy { facts, .. } => {
            json!({"session": facts.child, "parent": facts.parent})
        }
        Product::PrefixMember { facts, .. } => {
            json!({"session":facts.child,"parent":facts.parent,"message":facts.message})
        }
        Product::Background { facts, .. } => {
            json!({"session": facts.session, "task": facts.task_id})
        }
        Product::Repository { facts, .. } => json!({"path": facts.path}),
        Product::RestoredRequest { prepared } => json!({
            "request": prepared.request.id,
            "at": prepared.request.at,
            "machine": prepared.request.machine
        }),
        Product::RestoredAssertion { fact, .. } => json!({
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

/// Reads a product payload of `kind`; a kind version Polylogue does not
/// declare is refused, never read through the current shape.
pub(crate) fn decode(event_kind: &sinex_model::KindRef, payload: Value) -> Result<Product> {
    if crate::kinds::kind_reference(&event_kind.name)?.version != event_kind.version {
        return Err(Error::Invalid(format!(
            "unsupported Polylogue kind {}@{}",
            event_kind.name, event_kind.version
        )));
    }
    let event_type = event_kind.name.as_str();
    if event_type == "polylogue.tool_call" {
        let reading = serde_json::from_value(payload["source"].clone())?;
        let facts = Box::new(serde_json::from_value(payload)?);
        return Ok(Product::ToolCall { reading, facts });
    }
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
        ("polylogue.tool_call", "tool_call"),
        ("polylogue.document", "document"),
        ("polylogue.fragment", "fragment"),
        ("polylogue.event", "event"),
        ("polylogue.link", "link"),
        ("polylogue.prefix_copy", "prefix_copy"),
        ("polylogue.prefix_member", "prefix_member"),
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
            let mut schema = if *product == "tool_call" {
                ai::schema(product).expect("tool fact schema")
            } else {
                json!({
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
                })
            };
            if *product != "tool_call"
                && let Some(fields) = ai::schema(product)
            {
                schema["properties"]
                    .as_object_mut()
                    .expect("object properties")
                    .extend(
                        fields["properties"]
                            .as_object()
                            .expect("AI properties")
                            .clone(),
                    );
                if let Some(definitions) = fields.get("$defs").and_then(Value::as_object) {
                    let retained = schema["$defs"].as_object_mut().expect("object definitions");
                    for (name, definition) in definitions {
                        if let Some(existing) = retained.get(name) {
                            assert_eq!(existing, definition, "one definition for {name}");
                        } else {
                            retained.insert(name.clone(), definition.clone());
                        }
                    }
                }
                schema["required"]
                    .as_array_mut()
                    .expect("required properties")
                    .extend(
                        fields["required"]
                            .as_array()
                            .expect("required AI properties")
                            .clone(),
                    );
            }
            let field_hints = [
                "record",
                "text",
                "query",
                "rows",
                "refs",
                "blocks",
                "production",
                "attrs",
                "instructions",
                "source",
                "input",
                "written_text",
                "command",
                "path",
                "context",
            ]
            .into_iter()
            .filter(|field| schema["properties"].get(*field).is_some())
            .map(|field| FieldHint {
                field: format!("/{field}"),
                class: Class::Guarded,
            })
            .collect();
            KindDeclaration {
                kind: KindRef {
                    name: (*name).into(),
                    version: if *name == "polylogue.assertion" {
                        3
                    } else {
                        match *product {
                            // A product holding a parser reading moved to one
                            // progress value in place of two flags.
                            "message" => 7,
                            "request" | "restored_request" => 7,
                            "session" => 6,
                            "document" | "event" | "link" | "background" | "repository" => 5,
                            "parse_progress" => 6,
                            // An input step's rows refuse fields they do not
                            // name (version 5 moved `extent` to `extents`).
                            "input" => 6,
                            "fragment" => 3,
                            "query_run" => 4,
                            "restored_assertion" => 2,
                            "tool_call" | "restored_history" | "attachment_binding"
                            | "query_run_outcome" | "prefix_copy" | "prefix_member" => 1,
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
                time_rule: Some(if matches!(*product, "message" | "session" | "tool_call") {
                    TimeRule::InstantEvidence {
                        field: if *product != "session" {
                            "at"
                        } else {
                            "started"
                        }
                        .into(),
                        evidence_field: if *product != "session" {
                            "at_evidence"
                        } else {
                            "started_evidence"
                        }
                        .into(),
                    }
                } else if *product == "attachment_binding" {
                    TimeRule::Unknown
                } else {
                    TimeRule::FirstPresent {
                        fields: vec!["at".into(), "started".into()],
                        quality: TimeQuality::Intrinsic,
                    }
                }),
                field_hints,
                removed_rule: Some(RemovedRule::Never),
                keep_known: Vec::new(),
                owned_blobs: Vec::new(),
                search: if matches!(*product, "message" | "session" | "tool_call") {
                    Search::Words
                } else {
                    Search::None
                },
            }
        })
        .collect()
}

/// Build each event from its complete fact and its confirmed material anchor.
pub(crate) fn events(
    writes: &[polylogue_store::primary::products::ProductWrite],
) -> Result<Vec<sinex_protocol::ProducerEvent>> {
    use sinex_model::{Class, MaterialId, RecordLocation};
    let mut observations = std::collections::BTreeMap::<String, crate::native::Native>::new();
    for write in writes.iter().filter(|write| write.publish) {
        match &write.product {
            Product::Session { facts, .. } => {
                let context = observations
                    .entry(facts.id.to_string())
                    .or_insert_with(|| crate::native::Native::new(facts.repo.clone()));
                if let Some(repository) = &facts.repo {
                    context.repository(repository);
                }
            }
            Product::Event { facts, .. } => {
                if let (Some(session), Some(source)) = (&facts.session, &write.location) {
                    observations
                        .entry(session.to_string())
                        .or_default()
                        .observe(
                            &facts.data,
                            crate::kinds::RecordRef {
                                material: MaterialId(source.material.parse().map_err(|error| {
                                    Error::Invalid(format!("source material: {error}"))
                                })?),
                                anchor: source.anchor,
                            },
                        );
                }
            }
            _ => {}
        }
    }
    let contexts = observations
        .into_iter()
        .map(|(session, context)| (session, context.finish()))
        .collect::<std::collections::BTreeMap<_, _>>();
    writes
        .iter()
        .filter(|write| write.publish)
        .map(|write| {
            let mut payload = payload(&write.product, write.output_index)?;
            let mut source = write.location.clone();
            let session = match &write.product {
                Product::Session { facts, .. } => Some((&facts.id, true)),
                Product::ToolCall { facts, .. } => Some((&facts.session, false)),
                _ => None,
            };
            if let Some((session, session_record)) = session {
                let native = contexts
                    .get(&session.to_string())
                    .cloned()
                    .unwrap_or_default();
                if session_record && let Some(first) = native.sources.first() {
                    source = Some(polylogue_store::primary::products::SourceLocation {
                        material: first.material.to_string(),
                        anchor: first.anchor,
                    });
                }
                if session_record {
                    payload["repo"] = serde_json::to_value(native.repo)?;
                    payload["objectives"] = serde_json::to_value(native.objectives)?;
                    payload["git_identity"] = serde_json::to_value(native.git_identity)?;
                    payload["terminal"] = serde_json::to_value(native.terminal)?;
                    payload["boot_id"] = serde_json::to_value(native.boot_id)?;
                    payload["native_sources"] = serde_json::to_value(native.sources)?;
                    payload["native_unavailable"] = serde_json::to_value(native.unavailable)?;
                    payload["machine"] = serde_json::to_value(&write.machine)?;
                } else {
                    payload["context"] = serde_json::to_value(native)?;
                }
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

#[cfg(test)]
mod tests;
