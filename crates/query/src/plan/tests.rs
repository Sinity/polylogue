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
        .body
        .query_mut()
        .unwrap()
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
        .body
        .query_mut()
        .unwrap()
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
        identity::normal_form(changed.executable_ast().unwrap(), &native).unwrap();
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

#[test]
fn captured_dependency_reuses_proven_original_identity_after_representation_change() {
    use polylogue_model::query_objects::QueryObject;
    use serde_json::json;
    let span = json!({"start":0,"end":1});
    let ast = json!({"source":{"kind":"unit","unit":"sessions","implicit":false,"condition":{"span":span,"kind":{"kind":"exists","unit":"messages","condition":null}}},"attach":[],"stages":[{"span":span,"kind":{"kind":"matches"}}]});
    let native = crate::catalog::catalog(&[]);
    let form =
        polylogue_model::query_objects::evolution::legacy_normal_form(&ast, &native).unwrap();
    let id = identity::question_id_from_normal_form(&form, &BTreeMap::new()).unwrap();
    let migrated = QueryObject::from_structured_v1(json!({"id":id,"normal_form":form,"parameters":{},"identity_ast":ast,"source_outputs":[],"definitions":[],"dependencies":[],"edges":[]})).unwrap();
    assert_ne!(
        identity::normal_form(migrated.executable_ast().unwrap(), &native).unwrap(),
        migrated.normal_form
    );
    let captured = replay(&migrated, catalog_fingerprint()).unwrap();
    let mut bindings = Bindings::default();
    bindings
        .add_saved(
            "retained",
            &captured,
            "synthetic-revision",
            "synthetic-root",
        )
        .unwrap();
    let wrapper = prepare("query:retained | limit 1", &bindings, |_| Ok(())).unwrap();
    let wrapper = wrapper.query_object().unwrap();
    let dependency = wrapper
        .definitions
        .iter()
        .find(|node| node.question == id)
        .unwrap();
    assert_eq!(dependency.body, migrated.body);
    assert_eq!(dependency.normal_form, migrated.normal_form);
    assert!(dependency.historical_source.is_some());
    // The hash-keyed dependency is backed by the parent's validated closure,
    // even when its current serialization no longer hashes to the original ID.
    let selected = Bindings {
        saved: saved_query_bindings(&wrapper).unwrap(),
        ..Default::default()
    };
    let selected = prepare(&format!("query:{id}"), &selected, |_| Ok(())).unwrap();
    assert_eq!(selected.definitions()[0].question_id, id);
    let selected = selected.query_object().unwrap();
    validate_object(&selected).unwrap();
    assert!(
        selected
            .definitions
            .iter()
            .any(|node| node.question == id && node.historical_source.is_some())
    );
    let mut forged = wrapper;
    forged.definitions[0]
        .body
        .query_mut()
        .unwrap()
        .transforms
        .clear();
    assert!(saved_query_bindings(&forged).is_err());
}

#[test]
fn historical_inline_edges_keep_the_proven_original_subquery_id() {
    use polylogue_model::query_objects::QueryObject;
    use serde_json::json;
    let span = json!({"start":0,"end":1});
    let child = json!({"source":{"kind":"unit","unit":"sessions","implicit":false,"condition":null},"attach":[],"stages":[{"span":span,"kind":{"kind":"fields","fields":["id"]}},{"span":span,"kind":{"kind":"limit","count":1}}]});
    let native = crate::catalog::catalog(&[]);
    let child_form =
        polylogue_model::query_objects::evolution::legacy_normal_form(&child, &native).unwrap();
    let child_id = identity::question_id_from_normal_form(&child_form, &BTreeMap::new()).unwrap();
    let root = json!({"source":{"kind":"unit","unit":"sessions","implicit":false,"condition":{"span":span,"kind":{"kind":"in","operand":{"kind":"field","name":"id"},"query":child}}},"attach":[],"stages":[]});
    let root_form =
        polylogue_model::query_objects::evolution::legacy_normal_form(&root, &native).unwrap();
    let root_id = identity::question_id_from_normal_form(&root_form, &BTreeMap::new()).unwrap();
    let object = QueryObject::from_structured_v1(json!({"id":root_id,"normal_form":root_form,"parameters":{},"identity_ast":root,"source_outputs":[],"definitions":[{"question":child_id,"normal_form":child_form,"parameters":{},"identity_ast":child,"source_outputs":[]}],"dependencies":[child_id],"edges":[{"how":"in","query":child_id}]})).unwrap();
    assert_ne!(
        identity::question_id(
            object.definitions[0].executable_ast().unwrap(),
            &native,
            &BTreeMap::new()
        )
        .unwrap(),
        child_id
    );
    let plan = replay(&object, catalog_fingerprint()).unwrap();
    assert_eq!(plan.question_id(), root_id);
    assert_eq!(plan.query_object().unwrap().edges, object.edges);
}
