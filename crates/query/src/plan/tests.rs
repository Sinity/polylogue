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
fn canonical_object_replays_frozen_names_and_shapes_without_a_clock() {
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
    let decoded = serde_json::from_slice(&bytes).unwrap();
    validate_object(&decoded).unwrap();
}
#[test]
fn canonical_object_ignores_alias_labels_and_registry_help() {
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
fn object_validation_refuses_forged_ast_schema_and_missing_closure() {
    let object = prepare("query:nested | where count > 1", &bindings(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    let mut changed = object.clone();
    changed
        .query
        .transforms
        .push(Transform {
            span: Span::default(),
            kind: TransformKind::Limit { count: 1 },
        });
    assert!(validate_object(&changed).is_err());
    let mut changed = object.clone();
    changed.definitions.clear();
    assert!(validate_object(&changed).is_err());
    let mut changed = object.clone();
    changed.source_outputs[0]
        .schema
        .fields
        .push(query_language::catalog::Field {
            name: "invented".into(),
            ..Default::default()
        });
    assert!(validate_object(&changed).is_err());
    let mut changed = prepare("sessions", &Bindings::default(), |_| Ok(()))
        .unwrap()
        .query_object()
        .unwrap();
    changed
        .query
        .transforms
        .push(Transform {
            span: Span::default(),
            kind: TransformKind::Sort {
                keys: vec![SortKey {
                    expression: Expression::field("invented"),
                    descending: false,
                }],
            },
        });
    let native = crate::catalog::catalog(&[]);
    changed.normal_form =
        identity::normal_form(&changed.query, &native).unwrap();
    changed.id =
        identity::question_id_from_normal_form(&changed.normal_form, &changed.parameters).unwrap();
    assert!(validate_object(&changed).is_err());
}
#[test]
fn inline_composition_has_complete_frozen_bodies() {
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
fn native_saved_outputs_expose_only_columns_the_definition_produces() {
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
fn canonical_kept_replay_retains_its_immutable_typed_columns() {
    let mut bindings = Bindings::default();
    bindings.kept.insert(
        "kept-one".into(),
        crate::Kept {
            output_metadata: Default::default(),
            shape: None,
            native_identity: false,
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
    assert_eq!(restored.bindings().kept, bindings.kept);
    assert_eq!(restored.question_id(), plan.question_id());
}

#[test]
fn native_addresses_bind_in_explicit_row_domains_without_stealing_scoped_fields() {
    for (text, expected) in [
        ("events where event:native-event", "events"),
        ("documents where document:test:document", "documents"),
        ("blocks where block:test:message#0", "blocks"),
        ("deliveries where delivery:receipt", "deliveries"),
    ] {
        let plan = prepare(text, &Bindings::default(), |_| Ok(())).unwrap();
        let Source::Unit { unit, .. } = &plan.bound_query().query().source else {
            panic!("native unit")
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
fn saved_names_retain_frozen_dependencies_after_current_name_changes() {
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
            .contains("declared semantic hash")
    );
}

#[test]
fn saved_custody_and_cross_name_forks_are_explicit_without_poisoning_native_queries() {
    let mut snapshot = bindings();
    snapshot.saved.get_mut("project").unwrap()[0].body = crate::SavedQueryBody::Unavailable {
        reason: "retained body is absent".into(),
    };
    assert_eq!(
        prepare("query:project", &snapshot, |_| Ok(()))
            .unwrap_err()
            .kind(),
        "not_yet_available"
    );
    prepare("sessions", &snapshot, |_| Ok(())).unwrap();
    let fork = snapshot.saved.get_mut("alias").unwrap();
    fork[0].conflicts = vec![
        polylogue_model::Reference::parse("assertion:root-b@left").unwrap(),
        polylogue_model::Reference::parse("assertion:root-b@right").unwrap(),
    ];
    let error = prepare("query:alias", &snapshot, |_| Ok(()))
        .unwrap_err()
        .to_string();
    assert!(error.contains("root-b@left") && error.contains("root-b@right"));
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
        reference.to_uppercase(),
    ] {
        assert!(question_id_from_reference(&malformed).is_err());
    }
}

#[test]
fn exact_object_requirements_follow_typed_sources_not_scalar_spelling() {
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
fn the_catalog_fingerprint_is_computed_once_per_process() {
    let first = catalog_fingerprint();
    // Every caller borrows the same hashed value instead of rebuilding the catalog.
    assert!(std::ptr::eq(first, catalog_fingerprint()));
    let bytes = serde_json::to_vec(&crate::catalog::catalog(&[])).unwrap();
    assert_eq!(first, blake3::hash(&bytes).to_hex().as_str());
}
