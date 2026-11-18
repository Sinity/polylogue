use super::*;
#[test]
fn restored_assertion_keys_keep_each_import_source() {
    let fact = serde_json::from_value(json!({
        "id":"synthetic","root":"synthetic","kind":"note","key":null,
        "value":"Synthetic text", "author":"unattributed","via":"import","machine":"host",
        "created_at":null,"valid_until":null,"links":[]
    }))
    .unwrap();
    let ingress = |input: &str, at: i64| polylogue_model::authored::ImportAttribution {
        input: input.into(),
        chain: 1,
        machine: "receiving-host".into(),
        actor: polylogue_model::Author::Unattributed,
        at: polylogue_model::Timestamp::from_unix_ms(at).unwrap(),
    };
    let first = Product::RestoredAssertion {
        fact,
        import: Some(ingress("one", 1)),
    };
    let mut second = first.clone();
    if let Product::RestoredAssertion { import, .. } = &mut second {
        *import = Some(ingress("two", 2));
    }
    assert_ne!(key(&first, 0).unwrap(), key(&second, 0).unwrap());
    if let Product::RestoredAssertion { import, .. } = &mut second {
        *import = Some(ingress("one", 3));
    }
    assert_eq!(key(&first, 0).unwrap(), key(&second, 0).unwrap());
}

#[test]
fn the_save_transition_versions_changed_authored_kinds_and_keeps_the_old_manifest() {
    use polylogue_model::authored::{AuthoredEffect, PreparedRequest};
    use polylogue_model::user::UserOutcome;
    let prepared = |write| PreparedRequest {
        request: polylogue_store::fixture::authored::operator_request(
            "declared-line".into(),
            write,
        ),
        outcome: UserOutcome {
            id: "declared-line".into(),
            repeat: false,
            pending: false,
        },
        effect: AuthoredEffect::default(),
    };
    let previous: sinex_model::SourceDeclaration =
        serde_json::from_str(include_str!("../../tests/rebuild/source7.json")).unwrap();
    let current = declarations("polylogue");
    assert!(
        previous
            .kinds
            .iter()
            .any(|kind| kind.kind.name == "polylogue.kept_result")
    );
    assert!(
        !current
            .iter()
            .any(|kind| kind.kind.name == "polylogue.kept_result")
    );
    for name in [
        "polylogue.request",
        "polylogue.assertion",
        "polylogue.delivery",
        "polylogue.delivery_outcome",
        "polylogue.request_restore",
    ] {
        let old = previous
            .kinds
            .iter()
            .find(|kind| kind.kind.name == name)
            .unwrap();
        let now = current.iter().find(|kind| kind.kind.name == name).unwrap();
        assert_eq!(now.kind.name, old.kind.name);
        assert_eq!(now.kind.version, old.kind.version + 1, "{name}");
        assert_ne!(now.definition_hash(), old.definition_hash(), "{name}");
    }
    let journal = prepared(UserWrite::ResolveLine {
        request: json!({"selection":"sessions", "steps":[{"verb":"read"}]}),
        resolved: Some(vec![
            polylogue_model::Reference::parse("session:codex:one").unwrap(),
        ]),
        retained: None,
    });
    let mut saved = prepared(UserWrite::Save(Box::new(polylogue_model::user::Save {
        name: None,
        query: "sessions".into(),
        denominator: None,
        rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
        read_at: r#"{"now":1}"#.into(),
        watch: None,
        outcomes: Some(polylogue_model::user::SaveReport {
            saved: None,
            request: "declared-line".into(),
            repeat: false,
            pending: false,
            kept: polylogue_model::user::KeptPart {
                result: "result:declared-line".into(),
                rows: 0,
                at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            },
            watch: None,
        }),
    })));
    saved
        .effect
        .kept_results
        .push(polylogue_model::authored::KeptResult {
            id: "declared-line".into(),
            name: None,
            query: "sessions".into(),
            ran_at: saved.request.at,
            read_at: r#"{"now":1}"#.into(),
            rows_json: r#"{"unit":"sessions","columns":[],"rows":[]}"#.into(),
        });
    let authored = prepared(UserWrite::Assert { assertions: vec![] });
    for (product, old_product, name, old_name) in [
        (
            Product::Request(journal.clone()),
            Product::Request(authored.clone()),
            "polylogue.line_resolution",
            "polylogue.assertion",
        ),
        (
            Product::RestoredRequest { prepared: journal },
            Product::RestoredRequest {
                prepared: authored.clone(),
            },
            "polylogue.line_resolution_restore",
            "polylogue.request_restore",
        ),
        (
            Product::Request(saved.clone()),
            Product::Request(authored.clone()),
            "polylogue.line_save",
            "polylogue.assertion",
        ),
        (
            Product::RestoredRequest { prepared: saved },
            Product::RestoredRequest { prepared: authored },
            "polylogue.line_save_restore",
            "polylogue.request_restore",
        ),
    ] {
        assert_eq!(kind(&product), name);
        let declaration = current.iter().find(|kind| kind.kind.name == name).unwrap();
        let old = previous
            .kinds
            .iter()
            .find(|kind| kind.kind.name == old_name)
            .unwrap();
        let payload = payload(&product, 0).unwrap();
        let validator =
            jsonschema::validator_for(declaration.payload_schema.as_ref().unwrap()).unwrap();
        validator.validate(&payload).unwrap();
        assert!(
            validator
                .validate(&super::payload(&old_product, 0).unwrap())
                .is_err()
        );
        assert!(
            jsonschema::validator_for(old.payload_schema.as_ref().unwrap())
                .unwrap()
                .validate(&payload)
                .is_err()
        );
        assert_eq!(
            decode_source(&declaration.kind, &declaration.definition_hash(), payload).unwrap(),
            product
        );
        assert!(
            decode_source(
                &old.kind,
                &old.definition_hash(),
                super::payload(&old_product, 0).unwrap()
            )
            .is_err(),
            "an old declaration is not the new source format"
        );
        if matches!(&product, Product::Request(prepared) | Product::RestoredRequest { prepared }
            if matches!(prepared.request.write, UserWrite::Save(_)))
        {
            let accepted = current
                .iter()
                .filter(|kind| {
                    jsonschema::validator_for(kind.payload_schema.as_ref().unwrap())
                        .unwrap()
                        .is_valid(&super::payload(&product, 0).unwrap())
                })
                .map(|kind| kind.kind.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                accepted,
                vec![name],
                "one living declaration owns a Save effect"
            );
        }
    }
}
