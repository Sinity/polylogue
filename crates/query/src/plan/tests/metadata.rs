use super::*;

#[test]
fn frozen_semantic_outputs_ignore_parser_atoms_and_replay_exactly() {
    use query_language::catalog::{Field, FieldKind};
    let mut previous = None;
    for origins in [vec![], vec!["claude-code".into(), "codex".into()]] {
        let mut bindings = Bindings {
            origins,
            ..Bindings::default()
        };
        let definition = prepare(
            "sessions where exists tool_calls(outcome:error)",
            &bindings,
            |_| Ok(()),
        )
        .unwrap();
        bindings
            .add_saved(
                "qualified",
                &definition,
                "synthetic-revision",
                "synthetic-root",
            )
            .unwrap();
        let plan = prepare(
            "query:qualified | matches | sort by step.outcome",
            &bindings,
            |_| Ok(()),
        )
        .unwrap();
        let object = plan.query_object().unwrap();
        let fields = &object.source_outputs[0].schema.fields;
        assert!(fields.iter().any(|field| field.name == "title"));
        assert!(!fields.iter().any(|field| field.name == "codex"));
        let bytes = serde_json::to_vec(&object).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("match_patterns"));
        let decoded = serde_json::from_slice(&bytes).unwrap();
        let restored = replay(&decoded, catalog_fingerprint()).unwrap();
        assert_eq!(restored.question_id(), plan.question_id());
        assert_eq!(
            restored.output_schema().unwrap(),
            plan.output_schema().unwrap()
        );
        if let Some(previous) = &previous {
            assert_eq!(previous, &object);
        }
        previous = Some(object.clone());

        let mut changed = object.clone();
        changed.source_outputs[0].schema.fields.push(Field {
            name: "codex".into(),
            kind: FieldKind::Reference,
            ..Field::default()
        });
        assert!(
            matches!(validate_object(&changed), Err(QueryError::Invalid { .. })),
            "previously frozen parser atoms must be refused, not rewritten"
        );
        let mut changed = object.clone();
        changed.source_outputs[0].schema.fields[0].kind = FieldKind::Json;
        assert!(validate_object(&changed).is_err());
        let mut changed = object.clone();
        changed.source_outputs[0]
            .schema
            .fields
            .iter_mut()
            .find(|field| field.name == "kind")
            .unwrap()
            .values
            .push("invented-kind".into());
        assert!(validate_object(&changed).is_err());
        let mut changed = object.clone();
        changed.source_outputs[0].schema.fields.swap(0, 1);
        assert!(validate_object(&changed).is_err());
        let mut changed = object.clone();
        changed.source_outputs[0].schema.main_time = None;
        assert!(validate_object(&changed).is_err());
    }
}

#[test]
fn metadata_preparation_uses_shared_binding_before_value_and_reference_formatting() {
    let mut bindings = Bindings::default();
    for (name, text) in [
        ("pick", "$id"),
        ("forward", "query:pick(id=$id)"),
        ("nested", "query:forward(id=$id)"),
        ("by-outcome", "tool_calls where outcome:$outcome"),
    ] {
        let plan = prepare(text, &bindings, |_| Ok(())).unwrap();
        bindings
            .add_saved(
                name,
                &plan,
                &format!("{name}-revision"),
                &format!("{name}-root"),
            )
            .unwrap();
    }
    let id = "7f3a2c10-0003-4000-8000-000000000003";
    let direct = prepare(id, &bindings, |_| Ok(())).unwrap();
    for name in ["pick", "forward", "nested"] {
        for text in [
            format!("query:{name}(id={id})"),
            format!("query:{name}(id=\"{id}\")"),
        ] {
            let saved = prepare(&text, &bindings, |_| Ok(())).unwrap();
            assert_eq!(
                identity::normal_form(saved.expanded_query().query(), saved.catalog()).unwrap(),
                identity::normal_form(direct.expanded_query().query(), direct.catalog()).unwrap(),
                "{text}"
            );
            let object = saved.query_object().unwrap();
            let Source::Saved {
                name, arguments, ..
            } = &saved.identity_query().source
            else {
                panic!("saved identity source")
            };
            assert_eq!(
                object.source_outputs[0].source,
                OutputSource::Saved {
                    name: name.clone(),
                    arguments: arguments.clone()
                },
                "frozen output uses the actual bound key"
            );
            let mut duplicate = object.clone();
            let mut snapshot = duplicate.source_outputs[0].clone();
            let OutputSource::Saved { arguments, .. } = &mut snapshot.source else {
                panic!("saved output snapshot")
            };
            let Value::Scalar { quoted, .. } = &mut arguments[0].value else {
                panic!("concrete reference argument")
            };
            *quoted = !*quoted;
            duplicate.source_outputs.push(snapshot);
            assert!(
                validate_object(&duplicate).is_err(),
                "semantic duplicate output snapshot"
            );
            let replayed = replay(&object, catalog_fingerprint()).unwrap();
            assert_eq!(
                identity::normal_form(replayed.expanded_query().query(), replayed.catalog())
                    .unwrap(),
                identity::normal_form(direct.expanded_query().query(), direct.catalog()).unwrap(),
                "frozen {text}"
            );
        }
    }
    let failure = prepare(
        "query:by-outcome(outcome=\"error OR outcome:ok\")",
        &bindings,
        |_| Ok(()),
    )
    .unwrap_err();
    assert_eq!(failure.kind(), "bad_value");
    let QueryError::Language { error, .. } = failure else {
        panic!("shared typed refusal")
    };
    assert_eq!(error.kind, query_language::ErrorKind::BadValue);

    // The fast path inspects the whole tree, including nested compiled bodies.
    let nested = prepare(
        "sessions | fields id | union (sessions where exists tool_calls() | matches | fields parent as id)",
        &bindings,
        |_| Ok(()),
    )
    .unwrap();
    assert!(nested.bound_query().typing().queries.len() > 1);
    nested.query_object().unwrap();
}
