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
fn line_resolution_has_its_own_kinds_and_existing_authored_definitions_stay_exact() {
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
    for name in [
        "polylogue.request",
        "polylogue.assertion",
        "polylogue.kept_result",
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
        assert_eq!(now.kind, old.kind);
        assert_eq!(now.definition_hash(), old.definition_hash(), "{name}");
    }
    let journal = prepared(UserWrite::ResolveLine {
        request: json!({"selection":"sessions", "steps":[{"verb":"read"}]}),
        resolved: Some(vec![
            polylogue_model::Reference::parse("session:codex:one").unwrap(),
        ]),
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
            Product::RestoredRequest { prepared: authored },
            "polylogue.line_resolution_restore",
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
        assert_eq!(
            decode_source(
                &old.kind,
                &old.definition_hash(),
                super::payload(&old_product, 0).unwrap()
            )
            .unwrap(),
            old_product
        );
    }
}
