use super::*;
use query_language::identity;

fn bindings() -> Bindings {
    let mut bindings = Bindings::default();
    for (name, text, revision, root) in [
        (
            "project",
            "sessions where repo:morrow",
            "revision-a",
            "root-a",
        ),
        (
            "alias",
            "sessions where repo:morrow",
            "revision-b",
            "root-b",
        ),
        (
            "nested",
            "query:project | group by origin",
            "revision-c",
            "root-c",
        ),
    ] {
        let plan = prepare(text, &bindings, |_| Ok(())).unwrap();
        bindings.add_saved(name, &plan, revision, root).unwrap();
    }
    bindings
}
#[test]
fn origin_prefixed_session_reference_replays_as_a_canonical_reference() {
    let options = Bindings {
        origins: vec!["codex".into()],
        ..Bindings::default()
    };
    let compact = prepare("codex:019d0f2a", &options, |_| Ok(())).unwrap();
    let typed = prepare("session:codex:019d0f2a", &options, |_| Ok(())).unwrap();
    assert_eq!(compact.question_id(), typed.question_id());

    let stored = compact.query_object().unwrap();
    let replayed = replay(&stored, catalog_fingerprint()).unwrap();
    assert_eq!(replayed.question_id(), typed.question_id());
}

#[test]
fn saved_object_replays_names_and_shapes_without_a_clock() {
    let plan = prepare("query:nested | where count > 1", &bindings(), |_| Ok(())).unwrap();
    let object = plan.query_object().unwrap();
    assert_eq!(object.definitions.len(), 2);
    assert_eq!(
        object.source_outputs[0]
            .schema
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["count", "origin"]
    );
    assert_eq!(
        plan.output_schema()
            .unwrap()
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        ["origin", "count"]
    );
    let restored = replay(&object, catalog_fingerprint()).unwrap();
    assert_eq!(restored.question_id(), plan.question_id());
    assert_eq!(
        identity::normal_form(restored.expanded_query().query(), restored.catalog()).unwrap(),
        identity::normal_form(plan.expanded_query().query(), plan.catalog()).unwrap()
    );
    assert!(matches!(
        replay(&object, "retired-registry"),
        Err(QueryError::NotBuilt { .. })
    ));
    let bytes = serde_json::to_vec(&object).unwrap();
    let decoded: polylogue_model::query_objects::QueryObject =
        serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, object);
    validate_object(&decoded).unwrap();
}
#[test]
fn saved_object_ignores_alias_labels_and_registry_help() {
    let a = prepare("query:project", &bindings(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    let b = prepare("query:alias", &bindings(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    assert_eq!(a, b);
    assert!(!serde_json::to_string(&a).unwrap().contains("revision-a"));
    assert!(!serde_json::to_string(&a).unwrap().contains("root-a"));
}
#[test]
fn object_validation_refuses_forged_or_missing_dependencies() {
    let object = prepare("query:nested | where count > 1", &bindings(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    let mut changed = object.clone();
    changed.query.transforms.push(Transform {
        span: Span::default(),
        kind: TransformKind::Limit { count: 1 },
    });
    assert_eq!(
        refusal(&changed),
        "question identity, normal form and typed syntax disagree"
    );
    let mut changed = object.clone();
    changed.definitions.clear();
    assert!(refusal(&changed).starts_with("missing saved question "));
    let mut changed = object.clone();
    changed.source_outputs[0]
        .schema
        .fields
        .push(query_language::catalog::Field {
            name: "invented".into(),
            ..Default::default()
        });
    assert_eq!(
        refusal(&changed),
        "saved output schema differs from its saved definition"
    );
    let valid = prepare("sessions | sort by id", &Bindings::default(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    validate_object(&valid).unwrap();
    let mut changed = valid.clone();
    let TransformKind::Sort { keys } = &mut changed.query.transforms[0].kind else {
        panic!("expected a sort");
    };
    assert_eq!(
        keys.as_slice(),
        &[SortKey {
            expression: Expression::field("id"),
            descending: false,
        }]
    );
    keys[0].expression = Expression::field("invented");
    let catalog = crate::catalog::catalog(&[]);
    changed.normal_form = identity::normal_form(&changed.query, &catalog).unwrap();
    changed.id =
        identity::question_id_from_normal_form(&changed.normal_form, &changed.parameters).unwrap();
    assert_eq!(
        validate_object(&changed).unwrap_err().kind(),
        polylogue_model::Code::UnknownField
    );
}
fn refusal(object: &polylogue_model::query_objects::QueryObject) -> String {
    match validate_object(object) {
        Err(QueryError::Invalid { message, .. }) => message,
        other => panic!("expected an Invalid refusal, got {other:?}"),
    }
}
#[test]
fn object_validation_refuses_duplicate_and_unrelated_definitions() {
    let object = prepare("query:nested | where count > 1", &bindings(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    let mut duplicated = object.clone();
    duplicated.definitions.push(object.definitions[0].clone());
    assert_eq!(refusal(&duplicated), "duplicate saved question body");

    let other = prepare(
        "sessions where id in (sessions where repo:other)",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap()
    .query_object()
    .unwrap();
    let mut unrelated = object.clone();
    unrelated.definitions.push(other.definitions[0].clone());
    assert_eq!(
        refusal(&unrelated),
        "question contains an unrelated saved definition"
    );
}
#[test]
fn inline_composition_has_complete_saved_bodies() {
    for text in [
        "sessions | union (sessions where repo:morrow)",
        "sessions where id in (sessions where repo:morrow)",
    ] {
        let plan = prepare(text, &Bindings::default(), |_| Ok(())).unwrap();
        let object = plan.query_object().unwrap();
        assert_eq!(object.definitions.len(), 1, "{text}");
        replay(&object, catalog_fingerprint()).unwrap();
    }
}
#[test]
fn saved_outputs_expose_only_defined_columns() {
    let schema = bindings().output_schema("query:project").unwrap();
    assert_eq!(
        schema
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        crate::units::spec(crate::units::Unit::Sessions)
            .columns
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
    );
}

#[test]
fn kept_replay_keeps_its_typed_columns() {
    let mut bindings = Bindings::default();
    bindings.kept.insert(
        "kept-one".into(),
        crate::Kept {
            output_metadata: Default::default(),
            shape: None,
            row_identity: false,
            unit: "sessions".into(),
            columns: vec![
                crate::Column {
                    name: "id".into(),
                    kind: crate::ColKind::Text,
                    reference: Some(crate::ColumnReference::Session),
                },
                crate::Column {
                    name: "when".into(),
                    kind: crate::ColKind::Time,
                    reference: None,
                },
                crate::Column {
                    name: "ready".into(),
                    kind: crate::ColKind::Bool,
                    reference: None,
                },
            ],
            match_columns: vec![],
        },
    );
    let plan = prepare("result:kept-one | where ready:true", &bindings, |_| Ok(())).unwrap();
    let object = plan.query_object().unwrap();
    let restored = replay(&object, catalog_fingerprint()).unwrap();
    // The saved object stores the schema the rows were read with; a kept
    // result without a declaration reads with its physical column types, so
    // replay restores that schema as the declaration.
    let mut expected = bindings.kept.clone();
    let kept = expected.get_mut("kept-one").unwrap();
    kept.output_metadata.rows = Some(kept.output_schema(ResultPart::Rows));
    assert_eq!(restored.bindings().kept, expected);
    assert_eq!(restored.question_id(), plan.question_id());
}

#[test]
fn typed_addresses_bind_in_their_row_domains() {
    for (text, expected) in [
        ("events where event:native-event", "events"),
        ("documents where document:test:document", "documents"),
        ("blocks where block:test:message#0", "blocks"),
        ("deliveries where delivery:receipt", "deliveries"),
    ] {
        let plan = prepare(text, &Bindings::default(), |_| Ok(())).unwrap();
        let Source::Unit { unit, .. } = &plan.bound_query().query().source else {
            panic!("unit source")
        };
        assert_eq!(unit, expected);
        let object = plan.query_object().unwrap();
        assert_eq!(
            replay(&object, catalog_fingerprint())
                .unwrap()
                .question_id(),
            plan.question_id()
        );
    }
    let plan = prepare(
        "deliveries where event:session_start",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    let Source::Unit {
        unit,
        condition: Some(condition),
        ..
    } = &plan.bound_query().query().source
    else {
        panic!("delivery unit")
    };
    assert_eq!(unit, "deliveries");
    assert!(matches!(&condition.kind,ConditionKind::Match{field,..} if field=="event"));
}

#[test]
fn saved_names_keep_their_dependencies_after_renames() {
    let mut original = bindings();
    let before = prepare("query:nested | where count > 1", &original, |_| Ok(())).unwrap();
    let changed = prepare(
        "sessions where repo:other",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    original.saved.remove("project");
    original
        .add_saved("project", &changed, "new-revision", "root-a")
        .unwrap();
    let after = prepare("query:nested | where count > 1", &original, |_| Ok(())).unwrap();
    assert_eq!(before.question_id(), after.question_id());
    assert_eq!(
        before.query_object().unwrap(),
        after.query_object().unwrap()
    );
    let mut forged = original;
    forged.saved.get_mut("nested").unwrap()[0].hash = changed.question_id();
    assert!(
        prepare("query:nested", &forged, |_| Ok(()))
            .unwrap_err()
            .to_string()
            .contains("declared hash")
    );
}

#[test]
fn saved_forks_do_not_poison_unit_queries() {
    let mut snapshot = bindings();
    snapshot.saved.get_mut("project").unwrap()[0].body = crate::SavedQueryBody::Unavailable {
        reason: "stored body is absent".into(),
    };
    assert_eq!(
        prepare("query:project", &snapshot, |_| Ok(()))
            .unwrap_err()
            .kind(),
        polylogue_model::Code::NotYetAvailable
    );
    prepare("sessions", &snapshot, |_| Ok(())).unwrap();
    let fork = snapshot.saved.get_mut("alias").unwrap();
    fork[0].conflicts = vec![
        polylogue_model::Reference::parse("annotation:root-b@left").unwrap(),
        polylogue_model::Reference::parse("annotation:root-b@right").unwrap(),
    ];
    let error = prepare("query:alias", &snapshot, |_| Ok(())).unwrap_err();
    assert_eq!(error.kind(), polylogue_model::Code::Ambiguous);
    let QueryError::Ambiguous { candidates, .. } = error else {
        panic!("expected an ambiguous refusal, got {error:?}");
    };
    assert_eq!(candidates.len(), 2);
    assert!(candidates[0].contains("root-b@left") && candidates[1].contains("root-b@right"));
}

#[test]
fn denominator_classification_and_reference_use_the_pure_validated_plan() {
    for (text, required) in [
        ("sessions where repo:morrow", false),
        ("sessions | count", true),
        ("sessions | agg count", true),
        ("sessions | measure sessions", true),
    ] {
        assert_eq!(
            prepare(text, &Bindings::default(), |_| Ok(()))
                .unwrap()
                .requires_denominator(),
            required,
            "{text}"
        );
    }
    let plan = prepare("sessions", &Bindings::default(), |_| Ok(())).unwrap();
    let reference = format!("query:{}", plan.question_id());
    assert_eq!(
        question_id_from_reference(&reference).unwrap(),
        plan.question_id()
    );
    for malformed in [
        plan.question_id().to_string(),
        "query:abc".into(),
        format!("QUERY:{}", plan.question_id()),
        format!("query:{}", plan.question_id().to_string().to_uppercase()),
    ] {
        assert_eq!(
            question_id_from_reference(&malformed).unwrap_err(),
            QueryError::Invalid {
                message: "expected an exact lowercase query:<hash> reference".into(),
                position: 0,
            },
            "{malformed}"
        );
    }
}

#[test]
fn object_requirements_follow_typed_sources_not_scalar_spelling() {
    let id = prepare("sessions", &Bindings::default(), |_| Ok(()))
        .unwrap()
        .question_id();
    for text in [
        format!("query:{id} | where repo:morrow"),
        format!("sessions where query:{id}"),
        format!("sessions | union(query:{id})"),
    ] {
        let parsed = crate::parse::with_catalog(&text, &Bindings::default().catalog()).unwrap();
        assert_eq!(referenced_question_ids(&parsed), vec![id], "{text}");
    }
    for text in [
        format!("sessions where repo:\"query:{id}\""),
        format!("\"query:{id}\""),
        "query:saved-name".into(),
    ] {
        let parsed = crate::parse::with_catalog(&text, &Bindings::default().catalog()).unwrap();
        assert!(referenced_question_ids(&parsed).is_empty(), "{text}");
    }
}

#[test]
fn saved_name_requirements_follow_typed_nodes_without_rebinding_payloads() {
    let catalog = Bindings::default().catalog();
    let parsed = crate::parse::with_catalog(
        "query:Historical(text=\"query:Payload\") | union (sessions where query:Filter AND title:\"query:Scalar\") | intersect (query:Nested)",
        &catalog,
    ).unwrap();
    assert_eq!(
        referenced_saved_names(&parsed),
        ["Filter", "Historical", "Nested"]
    );
    assert!(referenced_question_ids(&parsed).is_empty());
    for text in [
        "sessions where query:Historical AND query:Historical",
        "sessions where id in (query:Historical)",
    ] {
        let parsed = crate::parse::with_catalog(text, &catalog).unwrap();
        assert_eq!(referenced_saved_names(&parsed), ["Historical"], "{text}");
    }
    for text in [
        "\"query:Historical\"",
        "sessions where title:\"query:Historical\"",
    ] {
        let parsed = crate::parse::with_catalog(text, &catalog).unwrap();
        assert!(referenced_saved_names(&parsed).is_empty(), "{text}");
    }
}

mod metadata;

#[test]
fn recorded_definitions_do_not_reinterpret_live_search_words() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("tests/fixtures/recorded-unit-objects.json")).unwrap();
    let object: polylogue_model::query_objects::QueryObject =
        serde_json::from_str(fixture["objects"][0]["original_json"].as_str().unwrap()).unwrap();
    let old = replay(&object, catalog_fingerprint()).unwrap();
    let mut loaded = Bindings::default();
    loaded
        .add_saved("recorded", &old, "revision", "root")
        .unwrap();
    for words in ["\"annotations\"", "links", "changes"] {
        let before = prepare(words, &Bindings::default(), |_| Ok(())).unwrap();
        let after = prepare(words, &loaded, |_| Ok(())).unwrap();
        assert_eq!(before.question_id(), after.question_id(), "{words}");
        assert_eq!(
            identity::normal_form(before.expanded_query().query(), before.catalog()).unwrap(),
            identity::normal_form(after.expanded_query().query(), after.catalog()).unwrap(),
            "{words}"
        );
        assert!(
            matches!(after.expanded_query().query().source, Source::Unit{ref unit,..} if unit=="sessions")
        );
    }
}

#[test]
fn an_original_authored_binding_keeps_identity_across_catalog_changes() {
    use polylogue_model::{query_objects::ExecutionBinding, user::UserRequest};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("tests/pre-sinex-binding.json")).unwrap();
    let original: serde_json::Value =
        serde_json::from_str(fixture["original_request"].as_str().unwrap()).unwrap();
    let request: UserRequest = serde_json::from_value(original.clone()).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), original);
    let intent = canonical_json::canonical_json(&(
        &request.machine,
        &request.authorship,
        request.write.intent(),
    ))
    .unwrap();
    assert_eq!(
        blake3::hash(&intent).to_hex().as_str(),
        fixture["request_hash"].as_str().unwrap(),
    );
    let execution: ExecutionBinding =
        serde_json::from_value(original["write"]["retained"]["binding"].clone()).unwrap();
    assert_ne!(
        catalog_fingerprint(),
        execution.replay.context.versions["catalog"],
        "F4 changes live unit and field declarations, not the original binding",
    );
    let object: polylogue_model::query_objects::QueryObject =
        serde_json::from_str(fixture["original_object"].as_str().unwrap()).unwrap();
    assert_eq!(
        object.id.to_string(),
        fixture["question_hash"].as_str().unwrap()
    );
    let stale = replay_binding(&execution.replay).unwrap_err();
    assert_eq!(
        stale,
        QueryError::NotBuilt {
            what: "the captured query catalog version".into()
        }
    );
    let restored = replay(&object, catalog_fingerprint()).unwrap();
    assert_eq!(restored.query_object().unwrap(), object);
    assert_eq!(
        serde_json::to_value(&execution).unwrap(),
        original["write"]["retained"]["binding"],
    );
}

#[test]
fn human_reference_spellings_bind_the_existing_stored_question_identity() {
    let catalog = crate::catalog::catalog(&[]);
    for (surface, stored) in [
        (
            "annotation:root@revision",
            "annotations where annotation:root@revision",
        ),
        (
            "annotations where about:annotation:root@revision",
            "annotations where about:annotation:root@revision",
        ),
        (
            "annotations where links_to:alternate:root@revision#2",
            "annotations where links_to:alternate:root@revision#2",
        ),
        (
            "annotations where reference:annotation:root@revision",
            "annotations where reference:annotation:root@revision",
        ),
        (
            "alternates where source:annotation:root@revision",
            "alternates where source:annotation:root@revision",
        ),
        (
            "deliveries where refs:annotation:root@revision",
            "deliveries where refs:annotation:root@revision",
        ),
        (
            "deliveries where left_out:annotation:root@revision",
            "deliveries where left_out:annotation:root@revision",
        ),
        (
            "feed where ref:annotation:root@revision",
            "feed where ref:annotation:root@revision",
        ),
    ] {
        let plan = prepare(surface, &Bindings::default(), |_| Ok(())).unwrap();
        // Explicit canonical input bypasses the surface normalization being tested.
        let old = query_language::parse::parse(stored, &catalog).unwrap();
        let expected =
            query_language::identity::question_id(&old, &catalog, &BTreeMap::new()).unwrap();
        assert_eq!(plan.question_id(), expected, "{surface}");
        let object = plan.query_object().unwrap();
        let restored = replay(&object, catalog_fingerprint()).unwrap();
        assert_eq!(restored.query_object().unwrap(), object);
    }
    let plan = prepare(
        "annotations where root:alternate:opaque",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    // Opaque root IDs keep their bytes.
    let Source::Unit {
        condition:
            Some(Condition {
                kind:
                    ConditionKind::Match {
                        field,
                        value: Value::Scalar { text, .. },
                    },
                ..
            }),
        ..
    } = &plan.identity_query().source
    else {
        panic!("{:?}", plan.identity_query().source);
    };
    assert_eq!(
        (field.as_str(), text.as_str()),
        ("root", "alternate:opaque")
    );
}

#[test]
fn comparison_values_use_the_normalized_conditions_binding() {
    prepare(
        "tool_calls where action:* NOT (exit_code != 0) AND at < 2026-10-01",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    let error = prepare(
        "tool_calls where exit_code != invalid",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap_err();
    let QueryError::Language { error, .. } = error else {
        panic!("expected the shared parser's numeric refusal, got {error:?}");
    };
    assert_eq!(error.kind, query_language::ErrorKind::BadValue);
    assert_eq!(
        error.message,
        "`invalid` is not a Number value for `exit_code`"
    );
}

#[test]
fn a_malformed_session_reference_is_refused() {
    let text = "annotations where   about:\"session:\"";
    let error = prepare(text, &Bindings::default(), |_| Ok(())).unwrap_err();
    let QueryError::BadValue {
        field,
        value,
        position,
        ..
    } = error
    else {
        panic!("expected a refused reference value, got {error:?}");
    };
    assert_eq!((field.as_str(), value.as_str()), ("about", "session:"));
    // The refusal points at the condition holding the value.
    assert_eq!(&text[position..], "about:\"session:\"");
    // Text with no kind, or an unknown kind, is left for the field to compare.
    for value in ["plain", "unknown-kind:value"] {
        prepare(
            &format!("annotations where about:\"{value}\""),
            &Bindings::default(),
            |_| Ok(()),
        )
        .unwrap();
    }
}

#[test]
fn annotation_history_roots_bind_natural_references_to_the_stored_family_identity() {
    let bare = prepare(
        "annotations where root:first withdrawn:*",
        &Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    for text in [
        "annotations where root:annotation:first withdrawn:*",
        "annotations where root:annotation:first@revision withdrawn:*",
    ] {
        let reference = prepare(text, &Bindings::default(), |_| Ok(())).unwrap();
        assert_eq!(reference.question_id(), bare.question_id(), "{text}");
    }
    for root in ["session:claude-code:first", "alternate:opaque"] {
        let opaque = prepare(
            &format!("annotations where root:{root}"),
            &Bindings::default(),
            |_| Ok(()),
        )
        .unwrap();
        assert!(
            serde_json::to_string(opaque.identity_query())
                .unwrap()
                .contains(root),
            "opaque roots keep their bytes: {root}"
        );
    }
}
