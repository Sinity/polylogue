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
    let declaration = declaration("polylogue.message");
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
        let payload = products::payload(&product, 0).unwrap();
        assert_eq!(payload["at"], json!(timestamp));
        assert_eq!(payload["at_evidence"], json!(evidence));
        assert_eq!(
            declaration
                .evaluate(&payload, sinex_model::Class::Open)
                .unwrap()
                .ts_orig,
            expected
        );
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
