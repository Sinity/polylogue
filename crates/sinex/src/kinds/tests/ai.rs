use super::*;

fn declaration(name: &str) -> sinex_model::KindDeclaration {
    declarations("polylogue")
        .into_iter()
        .find(|declaration| declaration.kind.name == name)
        .unwrap()
}

#[test]
fn native_ai_fields_bind_without_reading_the_internal_product() {
    let declaration = declaration("polylogue.message");
    let catalog = sinex_model::query::catalog::QueryCatalog::new([declaration.clone()]);
    for path in [
        vec!["message"],
        vec!["session"],
        vec!["role"],
        vec!["model"],
        vec!["blocks"],
        vec!["usage", "input"],
        vec!["production", "phase"],
        vec!["at_evidence", "quality"],
        vec!["at_evidence", "precision"],
    ] {
        let path = path.into_iter().map(str::to_string).collect::<Vec<_>>();
        assert!(catalog.field(&path).is_some(), "{path:?}");
    }
    let payload = products::payload(&message("input-a", 0), 0).unwrap();
    for field in payload.as_object().unwrap().keys() {
        assert!(
            declaration.payload_schema.as_ref().unwrap()["properties"]
                .get(field)
                .is_some(),
            "undeclared native message field {field}"
        );
    }
    assert_eq!(payload["message"], "codex:m");
    assert_eq!(payload["session"], "codex:s");
    assert_eq!(payload["role"], "assistant");
    assert_eq!(payload["production"]["phase"], "final");
    assert_eq!(payload["blocks"][0]["call_id"], "native");
    assert_eq!(payload["blocks"][0]["output"], "done");
    assert_eq!(payload["blocks"][0]["reported_duration_ms"], 17);
    let validator =
        jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
    validator.validate(&payload).unwrap();
    for (field, invalid) in [
        ("role", json!(17)),
        ("blocks", json!(false)),
        ("usage", json!("bad")),
    ] {
        let mut changed = payload.clone();
        changed[field] = invalid;
        assert!(validator.validate(&changed).is_err(), "{field}");
    }
}

#[test]
fn source_time_quality_and_precision_are_not_promoted_to_intrinsic_instants() {
    let mut product = message("input-a", 0);
    let at = polylogue_model::Timestamp::parse_rfc3339("2026-01-02T03:04:05Z").unwrap();
    for (timestamp, evidence, expected) in [
        (
            Some(at),
            polylogue_model::TimeEvidence::intrinsic(polylogue_model::Precision::Instant),
            sinex_model::TsOrig::known(
                serde_json::from_value(json!(at)).unwrap(),
                sinex_model::TimeQuality::Intrinsic,
            ),
        ),
        (
            Some(at),
            polylogue_model::TimeEvidence {
                quality: polylogue_model::TimeQuality::FileMtime,
                precision: Some(polylogue_model::Precision::Minute),
                until: None,
            },
            sinex_model::TsOrig::unknown(),
        ),
        (
            None,
            polylogue_model::TimeEvidence::UNKNOWN,
            sinex_model::TsOrig::unknown(),
        ),
    ] {
        let Product::Message { facts, .. } = &mut product else {
            unreachable!()
        };
        facts.message.set_time(timestamp, evidence).unwrap();
        let mut document = polylogue_model::Document::new(
            polylogue_model::DocumentId::parse("codex:document").unwrap(),
            polylogue_model::DocumentPurpose::Context,
        );
        document.set_time(timestamp, evidence).unwrap();
        let event = polylogue_model::Event {
            session: None,
            kind: "captured".into(),
            at: timestamp,
            at_quality: evidence.quality,
            at_precision: evidence.precision,
            at_until: evidence.until,
            after: None,
            range: Some((0, 1)),
            data: json!({}),
        };
        for product in [
            product.clone(),
            Product::Document {
                reading: reading("input-a", 0),
                facts: polylogue_model::DocumentRow {
                    offset: 0,
                    end: 1,
                    document,
                },
            },
            Product::Event {
                reading: reading("input-a", 0),
                id: "captured-event".into(),
                facts: event,
                imported: None,
            },
        ] {
            let payload = products::payload(&product, 0).unwrap();
            assert_eq!(payload["at"], json!(timestamp));
            assert_eq!(payload["at_evidence"], json!(evidence));
            assert_eq!(
                declaration(products::kind(&product))
                    .evaluate(&payload, sinex_model::Class::Open)
                    .unwrap()
                    .ts_orig,
                expected
            );
        }
    }
}

#[test]
fn session_identity_and_clock_evidence_are_declared_native_fields() {
    let mut session = polylogue_model::Session::new(
        polylogue_model::SessionId::parse("codex:s").unwrap(),
        polylogue_model::SessionKind::Main,
    );
    session.cwd = vec!["/synthetic/project".into()];
    session.repo = Some("https://example.test/project".into());
    session
        .set_time(
            polylogue_model::Timestamp::parse_rfc3339("2026-01-02T03:00:00Z"),
            polylogue_model::TimeEvidence {
                quality: polylogue_model::TimeQuality::FileMtime,
                precision: Some(polylogue_model::Precision::Day),
                until: None,
            },
        )
        .unwrap();
    let product = Product::Session {
        reading: reading("input-a", 0),
        facts: session,
    };
    let payload = products::payload(&product, 0).unwrap();
    let declaration = declaration("polylogue.session");
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&payload)
        .unwrap();
    for field in payload.as_object().unwrap().keys() {
        assert!(
            declaration.payload_schema.as_ref().unwrap()["properties"]
                .get(field)
                .is_some(),
            "{field}"
        );
    }
    assert_eq!(payload["session"], "codex:s");
    assert_eq!(payload["repo"], "https://example.test/project");
    assert_eq!(payload["cwd"], json!(["/synthetic/project"]));
    assert_eq!(payload["started_evidence"]["precision"], "day");
    assert_eq!(
        declaration
            .evaluate(&payload, sinex_model::Class::Open)
            .unwrap()
            .ts_orig,
        sinex_model::TsOrig::unknown()
    );
}

#[test]
fn tool_facts_keep_original_coordinates_and_only_their_sessions_observed_context() {
    use polylogue_store::primary::{ProductWrite, SourceLocation};
    let mut output = polylogue_model::Output::default();
    output.messages.push(serde_json::from_value(json!({
        "offset":50,"end":150,
        "message": {
            "id":"codex:action","session":"codex:s","role":"assistant",
            "at":"2026-01-02T03:04:05Z","at_quality":"intrinsic","at_precision":"instant",
            "blocks":[
                {"type":"tool_call","call_id":"call-1","tool":"write_file","action":"file_write","path":"/synthetic/project/src/example.rs","input":{"path":"/synthetic/project/src/example.rs","content":"one exact change"},"written_text":"one exact change"},
                {"type":"tool_result","call_id":"call-1","outcome":{"kind":"ok"},"output":"written"}
            ]
        }
    })).unwrap());
    let mut step = polylogue_store::ParseStep::new(output, Default::default(), 0..150, true);
    step.reading_version = "reader-3".into();
    step.prepare();
    let input = reading("input-a", 0).input;
    let mut products =
        polylogue_store::primary::products::parse(&input, 1, Some(150), &step).unwrap();
    let context = |session, process, window| {
        Product::Event {
        id: format!("context-{process}"),
        reading: reading("input-a", 0),
        imported: None,
        facts: serde_json::from_value(json!({
            "session":session,"kind":"native_context","range":[0,40],
            "data":{
                "repository":"https://example.test/project",
                "terminal_identity":{"host":"fixture-host","boot_id":"boot-a","adapter":"kitty","process_id":process,"window_id":window,"parent_process_id":null}
            }
        })).unwrap(),
    }
    };
    products.extend([
        context("codex:s", 100, "related-window"),
        context("codex:other", 200, "unrelated-window"),
    ]);
    let writes = products
        .into_iter()
        .enumerate()
        .map(|(index, product)| {
            let anchor = match &product {
                Product::ToolCall { facts, .. } => facts.source_start,
                Product::Message { facts, .. } => facts.offset,
                _ => 0,
            };
            ProductWrite {
                product,
                location: Some(SourceLocation {
                    material: "00000000-0000-0000-0000-000000000001".into(),
                    anchor,
                }),
                parents: vec![],
                machine: Some("fixture-host".into()),
                output_index: index as u64,
                publish: true,
            }
        })
        .collect::<Vec<_>>();
    let events = products::events(&writes).unwrap();
    let calls = events
        .iter()
        .filter(|event| event.kind.name == "polylogue.tool_call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    let call = calls[0];
    assert_eq!(call.location.as_ref().unwrap().anchor, 50);
    assert_eq!(call.payload["message"], "codex:action");
    assert_eq!(call.payload["call_id"], "call-1");
    assert_eq!(call.payload["action"], "file_write");
    assert_eq!(call.payload["source_start"], 50);
    assert_eq!(call.payload["source_end"], 150);
    assert_eq!(call.payload["source"]["input"]["id"], "input-a");
    assert_eq!(call.payload["input"]["content"], "one exact change");
    assert_eq!(
        call.payload["context"]["terminal"]["window_id"],
        "related-window"
    );
    assert_eq!(call.payload["context"]["terminal"]["process_id"], 100);
    assert_eq!(
        call.payload["context"]["sources"],
        json!([{"material":"00000000-0000-0000-0000-000000000001","anchor":0}])
    );
    let declaration = declaration("polylogue.tool_call");
    jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap())
        .unwrap()
        .validate(&call.payload)
        .unwrap();
    let catalog = sinex_model::query::catalog::QueryCatalog::new([declaration.clone()]);
    for path in [
        vec!["action"],
        vec!["path"],
        vec!["context", "terminal", "process_id"],
        vec!["context", "terminal", "boot_id"],
    ] {
        assert!(
            catalog
                .field(&path.into_iter().map(str::to_string).collect::<Vec<_>>())
                .is_some()
        );
    }
    assert_eq!(
        declaration
            .evaluate(&call.payload, sinex_model::Class::Open)
            .unwrap()
            .ts_orig
            .quality,
        sinex_model::TimeQuality::Intrinsic
    );
    let decoded = products::decode(&call.kind, call.payload.clone()).unwrap();
    assert_eq!(
        decoded,
        writes
            .iter()
            .find(|write| matches!(write.product, Product::ToolCall { .. }))
            .map(|write| eligibility(&write.product))
            .unwrap()
    );
}
