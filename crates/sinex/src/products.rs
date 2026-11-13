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

/// Declared native facts retain domain values independently of the local index.
pub(crate) fn payload(product: &Product, occurrence: u64) -> Result<Value> {
    let mut payload = match product {
        Product::Interpreted { .. } => unreachable!("eligibility is local metadata"),
        Product::ToolCall { facts, .. } => serde_json::to_value(ai::ToolFields {
            call: facts,
            context: Default::default(),
        })?,
        Product::Session { facts, .. } => serde_json::to_value(ai::SessionFields::from(facts))?,
        Product::Message {
            facts,
            order,
            projection,
            native_fields,
            declared_injections,
            declared_attachment_bindings,
            ..
        } => serde_json::to_value(ai::MessageFields::new(
            facts,
            *order,
            projection,
            native_fields,
            declared_injections,
            declared_attachment_bindings,
        ))?,
        Product::Document { facts, .. } => {
            let mut value = serde_json::to_value(&facts.document)?;
            rename_field(&mut value, "id", "document");
            value["source_start"] = facts.offset.into();
            value["source_end"] = facts.end.into();
            value["at"] = serde_json::to_value(facts.document.source_time)?;
            value["at_evidence"] = serde_json::to_value(facts.document.time_evidence())?;
            value
        }
        Product::Fragment { facts, .. } => {
            let mut value = serde_json::to_value(&facts.record)?;
            value["source_start"] = facts.offset.into();
            value["source_end"] = facts.end.into();
            value
        }
        Product::Event {
            id,
            facts,
            imported,
            ..
        } => {
            let mut value = serde_json::to_value(facts)?;
            rename_field(&mut value, "kind", "event");
            value["at"] = serde_json::to_value(facts.at)?;
            value["at_evidence"] = serde_json::to_value(polylogue_model::TimeEvidence {
                quality: facts.at_quality,
                precision: facts.at_precision,
                until: facts.at_until,
            })?;
            value["event_id"] = id.clone().into();
            value["imported"] = serde_json::to_value(imported)?;
            value
        }
        Product::Link { facts, .. } => {
            let mut value = serde_json::to_value(facts)?;
            rename_field(&mut value, "child", "session");
            rename_field(&mut value, "at", "branch_point");
            value
        }
        Product::PrefixCopy { facts, .. } => {
            let mut value = serde_json::to_value(facts)?;
            rename_field(&mut value, "child", "session");
            value
        }
        Product::PrefixMember { facts, .. } => {
            let mut value = serde_json::to_value(facts)?;
            rename_field(&mut value, "child", "session");
            value
        }
        Product::Background { facts, .. } => {
            let mut value = serde_json::to_value(facts)?;
            rename_field(&mut value, "task_id", "task");
            value
        }
        Product::Repository { facts, .. } => serde_json::to_value(facts)?,
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
        Product::ParseProgress {
            reading,
            counts,
            stopped_attempt,
        } => json!({
            "input": reading.input.id, "counts": counts, "stopped_attempt": stopped_attempt,
        }),
    };
    payload["record_key"] = key(product, occurrence)?.into();
    if let Some(reading) = product.reading() {
        payload["source"] = serde_json::to_value(reading)?;
        payload["interpretation"] = serde_json::to_value(
            polylogue_store::primary::SourceRecord::of(product)?.expect("source fact identity"),
        )?;
    } else {
        payload["record"] = serde_json::to_value(product)?;
    }
    Ok(payload)
}

fn rename_field(value: &mut Value, from: &str, to: &str) {
    let fields = value.as_object_mut().expect("domain fact object");
    if let Some(field) = fields.remove(from) {
        fields.insert(to.into(), field);
    }
}

fn source_kind(kind: &str) -> bool {
    matches!(
        kind,
        "polylogue.session"
            | "polylogue.message"
            | "polylogue.tool_call"
            | "polylogue.document"
            | "polylogue.fragment"
            | "polylogue.event"
            | "polylogue.link"
            | "polylogue.prefix_copy"
            | "polylogue.prefix_member"
            | "polylogue.background"
            | "polylogue.repository"
            | "polylogue.parse_progress"
    )
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
    if source_kind(event_type) {
        let reading: polylogue_store::primary::Reading =
            serde_json::from_value(payload["source"].clone())?;
        let record: polylogue_store::primary::SourceRecord =
            serde_json::from_value(payload["interpretation"].clone())?;
        if reading.chain < 1
            || reading.input.id.is_empty()
            || reading.input.format.is_empty()
            || reading.reading_version.is_empty()
            || reading.start > reading.parsed_upto
            || reading
                .source_length
                .is_some_and(|length| reading.parsed_upto > length)
            || record.family != event_type.trim_start_matches("polylogue.")
            || record.reference.is_empty()
            || record.start > record.end
            || record.end > reading.parsed_upto
        {
            return Err(Error::Invalid(
                "native interpretation differs from its original source".into(),
            ));
        }
        if let Some(delivery) = &reading.delivery {
            delivery
                .validate()
                .map_err(|error| Error::Invalid(error.to_string()))?;
            if reading.source_length.is_none() {
                return Err(Error::Invalid(
                    "export reading has no actual source length".into(),
                ));
            }
        }
        if event_type == "polylogue.parse_progress"
            && !payload["stopped_attempt"].is_null()
            && !matches!(
                reading.progress,
                polylogue_model::batch::StepProgress::Failed { .. }
                    | polylogue_model::batch::StepProgress::Refused { .. }
            )
        {
            return Err(Error::Invalid(
                "only stopped progress has a stopped attempt".into(),
            ));
        }
        return Ok(Product::Interpreted { reading, record });
    }
    let product: Product =
        serde_json::from_value(payload.get("record").cloned().ok_or_else(|| {
            Error::Invalid(format!("confirmed {event_type} has no complete facts"))
        })?)?;
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

/// Native source kinds declare their domain fields and source evidence.
/// Durable authored and input records retain their exact operation decoder.
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
            let source = source_kind(name);
            let mut schema = if source {
                ai::schema(product).unwrap_or_else(|| {
                    assert_eq!(*product, "parse_progress");
                    let fields = &facts["oneOf"][0]["properties"];
                    json!({
                        "type":"object", "properties": {
                            "input":{"type":"string"}, "counts":fields["counts"],
                            "stopped_attempt":fields["stopped_attempt"],
                        }, "required":["input","counts","stopped_attempt"],
                    })
                })
            } else {
                // This published record-schema annotation is immutable wire
                // metadata, independent of internal Product documentation.
                facts["description"] = json!(
                    "Complete parser and authored facts; rebuilding the cache requires no parsing."
                );
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
                    "required":["record_key","record"],
                })
            };
            let mut definitions = definitions.as_object().expect("schema definitions").clone();
            if let Some(native) = schema
                .as_object_mut()
                .expect("native schema object")
                .remove("$defs")
            {
                for (name, definition) in native.as_object().expect("native definitions") {
                    if let Some(existing) = definitions.get(name) {
                        assert_eq!(existing, definition, "one definition for {name}");
                    } else {
                        definitions.insert(name.clone(), definition.clone());
                    }
                }
            }
            if source {
                let interpreted = complete["oneOf"]
                    .as_array()
                    .expect("record variants")
                    .iter()
                    .find(|variant| variant["properties"]["product"]["const"] == "interpreted")
                    .expect("source eligibility schema");
                let fields = schema["properties"].as_object_mut().expect("domain fields");
                fields.insert("record_key".into(), json!({"type":"string","minLength":1}));
                fields.insert(
                    "source".into(),
                    interpreted["properties"]["reading"].clone(),
                );
                fields.insert(
                    "interpretation".into(),
                    interpreted["properties"]["record"].clone(),
                );
                if matches!(*product, "document" | "fragment") {
                    fields.insert("source_start".into(), json!({"type":"integer","minimum":0}));
                    fields.insert("source_end".into(), json!({"type":"integer","minimum":0}));
                }
                if *product == "document" {
                    fields.insert("at".into(), json!({"type":["string","null"]}));
                }
                if *product == "event" {
                    let variant = &facts["oneOf"][0]["properties"];
                    fields.insert("event_id".into(), variant["id"].clone());
                    fields.insert("imported".into(), variant["imported"].clone());
                }
                let required = schema["required"]
                    .as_array_mut()
                    .expect("required domain fields");
                required.extend([
                    json!("record_key"),
                    json!("source"),
                    json!("interpretation"),
                ]);
            }
            schema["$defs"] = reachable_definitions(&schema, &Value::Object(definitions));
            let mut guarded = [
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
                "interpretation",
                "field_claims",
                "injections",
                "attachment_bindings",
                "projection",
                "order",
                "id_kind",
                "page_text",
                "source_start",
                "source_end",
                "aliases",
                "title_source",
                "data",
                "imported",
                "counts",
                "stopped_attempt",
                "content",
                "provenance",
                "coverage",
                "source_version",
                "root_commits",
                "remote",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
            if source && !matches!(*product, "session" | "message" | "tool_call") {
                for field in schema["properties"]
                    .as_object()
                    .expect("domain fields")
                    .keys()
                {
                    if !matches!(
                        field.as_str(),
                        "record_key"
                            | "document"
                            | "at"
                            | "session"
                            | "event"
                            | "parent"
                            | "message"
                            | "task"
                            | "path"
                            | "input"
                    ) && !guarded.contains(field)
                    {
                        guarded.push(field.clone());
                    }
                }
            }
            let field_hints = guarded
                .into_iter()
                .filter(|field| schema["properties"].get(field).is_some())
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
                            "message" => 8,
                            "request" | "restored_request" => 7,
                            "session" => 7,
                            "document" | "event" | "link" | "background" | "repository" => 6,
                            "parse_progress" => 7,
                            // Input rows refuse fields outside their declared source shape.
                            "input" => 6,
                            "fragment" => 4,
                            "query_run" => 4,
                            "restored_assertion" => 2,
                            "tool_call" | "prefix_copy" | "prefix_member" => 2,
                            "restored_history" | "attachment_binding" | "query_run_outcome" => 1,
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
                time_rule: Some(
                    if matches!(
                        *product,
                        "message" | "session" | "tool_call" | "document" | "event"
                    ) {
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
                    } else if source || *product == "attachment_binding" {
                        TimeRule::Unknown
                    } else {
                        TimeRule::FirstPresent {
                            fields: vec!["at".into(), "started".into()],
                            quality: TimeQuality::Intrinsic,
                        }
                    },
                ),
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
