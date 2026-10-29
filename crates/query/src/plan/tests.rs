use super::*;
use query_language::identity;

fn bindings() -> Bindings {
    Bindings {
        saved: BTreeMap::from([
            (
                "project".into(),
                vec![crate::SavedQuery {
                    text: "sessions where repo:morrow".into(),
                    revision: "revision-a".into(),
                    first_note: "root-a".into(),
                }],
            ),
            (
                "alias".into(),
                vec![crate::SavedQuery {
                    text: "sessions where repo:morrow".into(),
                    revision: "revision-b".into(),
                    first_note: "root-b".into(),
                }],
            ),
            (
                "nested".into(),
                vec![crate::SavedQuery {
                    text: "query:project | group by origin".into(),
                    revision: "revision-c".into(),
                    first_note: "root-c".into(),
                }],
            ),
        ]),
        ..Bindings::default()
    }
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
        ["origin", "count"]
    );
    let restored = replay(&object, &catalog_fingerprint()).unwrap();
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
    changed.identity_ast.stages.push(Stage {
        span: Span::default(),
        kind: StageKind::Limit { count: 1 },
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
    changed.identity_ast.stages.push(Stage {
        span: Span::default(),
        kind: StageKind::Sort {
            keys: vec![SortKey {
                operand: Operand::field("invented"),
                descending: false,
            }],
        },
    });
    let native = crate::catalog::catalog(&[], &[]);
    changed.normal_form = identity::normal_form(&changed.identity_ast, &native).unwrap();
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
        replay(&object, &catalog_fingerprint()).unwrap();
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
    let restored = replay(&object, &catalog_fingerprint()).unwrap();
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
            replay(&object, &catalog_fingerprint())
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
