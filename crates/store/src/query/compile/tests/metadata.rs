use super::*;

#[test]
fn output_schema_uses_shared_bucket_names_types_and_closed_values() {
    let options = opts(&[]);
    let schema = options
        .output_schema("tool_calls | group by day(at),outcome | count")
        .unwrap();
    let field = |name: &str| {
        schema
            .fields
            .iter()
            .find(|field| field.name == name)
            .unwrap()
    };
    assert_eq!(field("day").kind, query_language::catalog::FieldKind::Time);
    assert_eq!(
        field("count").kind,
        query_language::catalog::FieldKind::Integer
    );
    assert_eq!(
        field("outcome").values,
        ["ok", "error", "pending", "unknown"]
    );
    assert!(
        !schema
            .fields
            .iter()
            .any(|field| field.name == "session.repo")
    );
}

#[test]
fn matches_declarations_follow_actual_producer_columns_and_replay_frozen_saved_bodies() {
    for text in [
        "sessions where exists tool_call() | matches",
        "sessions where seq(action:shell -> outcome:error) | matches",
        "sessions where exists tool_call() | limit 1 | matches",
        "sessions where exists tool_call() | distinct | matches",
    ] {
        let options = opts(&[]);
        let schema = options.output_schema(text).unwrap();
        let program = compile(text, &options).unwrap();
        assert_eq!(
            schema
                .fields
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            program
                .columns
                .iter()
                .map(|column| column.name.as_str())
                .collect::<Vec<_>>(),
            "{text}"
        );
        assert!(schema.match_units.is_none());
    }
    for (definition, text) in [
        (
            "sessions where exists tool_call(outcome:error) | matches",
            "query:matching | sort by step.command",
        ),
        (
            "sessions where exists tool_call(outcome:error)",
            "query:matching | matches | sort by step.command",
        ),
    ] {
        let options = opts(&[("matching", definition)]);
        let plan = polylogue_query::plan::prepare(text, &options.bindings(), |_| Ok(())).unwrap();
        let object = plan.query_object().unwrap();
        let bytes = serde_json::to_vec(&object).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("match_patterns"));
        assert!(!String::from_utf8_lossy(&bytes).contains("match_outputs"));
        let restored = serde_json::from_slice(&bytes).unwrap();
        let replay =
            polylogue_query::plan::replay(&restored, polylogue_query::plan::catalog_fingerprint())
                .unwrap();
        assert_eq!(replay.question_id(), plan.question_id());
        assert_eq!(
            lower(&replay, &options).unwrap().columns,
            lower(&plan, &options).unwrap().columns,
            "{text}"
        );
        let schema = options.output_schema(text).unwrap();
        assert_eq!(
            schema
                .fields
                .iter()
                .find(|field| field.name == "step.outcome")
                .unwrap()
                .values,
            ["ok", "error", "pending", "unknown"]
        );
    }
    let text = "sessions where seq all(action:shell -> action:shell) | matches";
    let error = compile(text, &opts(&[])).unwrap_err();
    assert_eq!(error.kind(), "not_built", "{text}: {error}");

    use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    for (id, commands) in [("a", vec!["a-first", "a-second"]), ("b", vec!["b-only"])] {
        let session = fixture
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                ..Default::default()
            })
            .unwrap();
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("claude-code:m-{id}"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        for command in commands {
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        command: Some(command),
                        outcome: "ok",
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    fixture.derive().unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (selection, expected) in [
        ("sort by id | limit 1", vec!["a-first", "a-second"]),
        ("distinct", vec!["a-first", "a-second", "b-only"]),
    ] {
        let text = format!(
            "sessions where exists tool_call() | {selection} | matches | fields step.command as command | sort by command"
        );
        let program = compile(&text, &opts(&[])).unwrap();
        let mut rows = reader.query_rows(&program).unwrap();
        for row in &mut rows {
            crate::query::execute::present(&program, row).unwrap();
        }
        assert_eq!(
            rows.iter()
                .map(|row| row["command"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
}

#[test]
fn frozen_saved_semantic_scope_preserves_native_fields_and_refuses_forged_declarations() {
    let mut options = opts(&[
        ("failing", "tool_calls where outcome:error"),
        ("grouped", "tool_calls | group by action | count"),
        ("selected", "tool_calls | fields command"),
        ("scoped", "sessions | tool_calls where outcome:error"),
    ]);
    let plan = polylogue_query::plan::prepare(
        "query:failing session.repo:morrow | sort by command",
        &options.bindings(),
        |_| Ok(()),
    )
    .unwrap();
    let object = plan.query_object().unwrap();
    assert!(
        object.source_outputs[0]
            .schema
            .fields
            .iter()
            .any(|field| field.name == "session.repo")
    );
    let encoded = serde_json::to_vec(&object).unwrap();
    assert!(!String::from_utf8_lossy(&encoded).contains("relation_scope"));
    let restored = serde_json::from_slice(&encoded).unwrap();
    let replay =
        polylogue_query::plan::replay(&restored, polylogue_query::plan::catalog_fingerprint())
            .unwrap();
    let source = &replay.bound_query().query().source;
    assert!(
        replay
            .catalog()
            .source_output(source)
            .unwrap()
            .fields
            .iter()
            .any(|field| field.name == "session.repo")
    );
    assert_eq!(replay.question_id(), plan.question_id());
    let mut forged = restored.clone();
    forged.source_outputs[0]
        .schema
        .fields
        .push(query_language::catalog::Field {
            name: "invented".into(),
            ..Default::default()
        });
    assert!(
        polylogue_query::plan::replay(&forged, polylogue_query::plan::catalog_fingerprint(),)
            .is_err()
    );
    let expected = lower(&replay, &options).unwrap();
    let changed = polylogue_query::plan::prepare(
        "sessions where repo:other",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    let mut current = options.bindings();
    current.saved.remove("failing");
    current
        .add_saved("failing", &changed, "changed-revision", "changed-root")
        .unwrap();
    options.saved = current.saved;
    let actual = lower(&replay, &options).unwrap();
    assert_eq!(actual.sql, expected.sql);
    assert_eq!(actual.params, expected.params);
    for text in [
        "query:grouped | where session.repo:morrow",
        "query:selected | where session.repo:morrow",
    ] {
        assert_eq!(
            compile(text, &options).unwrap_err().kind(),
            "unknown_field",
            "{text}"
        );
    }
    let scoped = polylogue_query::plan::prepare(
        "query:scoped | where session.repo:morrow",
        &options.bindings(),
        |_| Ok(()),
    )
    .unwrap();
    let scoped_object = scoped.query_object().unwrap();
    assert_eq!(
        scoped_object.source_outputs[0].schema.unit.as_deref(),
        Some("tool_calls")
    );
    assert!(
        !scoped_object.source_outputs[0]
            .schema
            .fields
            .iter()
            .any(|field| field.name == "codex")
    );
    polylogue_query::plan::replay(&scoped_object, polylogue_query::plan::catalog_fingerprint())
        .unwrap();
    options.kept.insert(
        "artifact".into(),
        Kept {
            output_metadata: Default::default(),
            shape: None,
            native_identity: false,
            unit: "tool_calls".into(),
            columns: vec![Column {
                name: "command".into(),
                kind: ColKind::Text,
                reference: None,
            }],
            match_columns: vec![],
        },
    );
    let artifact =
        polylogue_query::plan::prepare("result:artifact", &options.bindings(), |_| Ok(())).unwrap();
    let mut current = options.bindings();
    current
        .add_saved(
            "artifact-alias",
            &artifact,
            "artifact-revision",
            "artifact-root",
        )
        .unwrap();
    options.saved = current.saved;
    assert_eq!(
        compile("query:artifact-alias | where session.repo:morrow", &options)
            .unwrap_err()
            .kind(),
        "unknown_field"
    );
    let schema = options
        .output_schema("tool_calls | group by action | agg sum(exit_code)")
        .unwrap();
    assert!(schema.fields.iter().any(|field| field.name == "count"));
}

#[test]
fn forwarded_saved_conditions_keep_suffixes_defaults_and_reject_cycles() {
    let options = opts(&[
        ("pick", "$id"),
        ("forward", "query:pick(id=$id)"),
        ("nested", "query:forward(id=$id) origin:claude-code"),
        ("calls", "tool_calls where outcome:error"),
        ("wrong-unit", "query:calls"),
        ("shaped", "sessions | group by repo | count"),
        ("forward-shaped", "query:shaped"),
        ("cycle-a", "query:cycle-b"),
        ("cycle-b", "query:cycle-a"),
    ]);
    let id = "7f3a2c10-0003-4000-8000-000000000003";
    same(
        &format!("sessions where query:nested(id={id})"),
        &format!("sessions where {id} origin:claude-code"),
        &options,
    );
    for text in [
        "sessions where query:wrong-unit",
        "sessions where query:forward-shaped",
        "sessions where query:cycle-a",
    ] {
        assert!(
            matches!(compile(text, &options), Err(QueryError::Invalid { .. })),
            "{text}"
        );
    }
    // Exercise the backend guard itself, independently of Plan cycle validation.
    let mut compiler = Compiler::new("sessions where query:cycle-a", &options);
    assert!(matches!(
        compiler.saved_relation("cycle-a", &[], Span::default()),
        Err(QueryError::Invalid { .. })
    ));
    assert!(compiler.saved_stack.is_empty());
}

#[test]
fn canonical_replay_preserves_unproven_text_and_json_matching_policy() {
    use polylogue_model::kept::{OutputMetadata, SchemaProof};
    for proof_kind in 0..3 {
        let mut options = opts(&[]);
        let columns = vec![
            Column {
                name: "body".into(),
                kind: ColKind::Text,
                reference: None,
            },
            Column {
                name: "payload".into(),
                kind: ColKind::Json,
                reference: None,
            },
        ];
        let schema = polylogue_query::compile::columns_schema("sessions", &columns);
        let proof = match proof_kind {
            0 => SchemaProof::Unproven,
            1 => SchemaProof::Captured {
                schema,
                unproven_fields: vec!["body".into(), "payload".into()],
            },
            _ => SchemaProof::Captured {
                schema,
                unproven_fields: vec![],
            },
        };
        options.kept.insert(
            "legacy".into(),
            Kept {
                output_metadata: OutputMetadata {
                    rows: proof,
                    ..Default::default()
                },
                unit: "sessions".into(),
                columns,
                match_columns: vec![],
                shape: None,
                native_identity: false,
            },
        );
        let base = polylogue_query::plan::prepare("result:legacy", &options.bindings(), |_| Ok(()))
            .unwrap();
        let mut bindings = options.bindings();
        bindings
            .add_saved("retained", &base, "retained-revision", "retained-root")
            .unwrap();
        options.saved = bindings.saved;
        for (source, field) in ["result:legacy", "query:retained"]
            .into_iter()
            .flat_map(|source| {
                ["body", "payload"]
                    .into_iter()
                    .map(move |field| (source, field))
            })
        {
            let text = format!("{source} | where {field}:needle");
            let plan =
                polylogue_query::plan::prepare(&text, &options.bindings(), |_| Ok(())).unwrap();
            let encoded = serde_json::to_vec(&plan.query_object().unwrap()).unwrap();
            let object = serde_json::from_slice(&encoded).unwrap();
            let replay = polylogue_query::plan::replay(
                &object,
                polylogue_query::plan::catalog_fingerprint(),
            )
            .unwrap();
            assert_eq!(replay.bindings().kept, options.kept);
            // Lower using only the recovered immutable declaration, so consulting
            // a current source cannot conceal a proof upgrade during replay.
            let mut recovered = options.clone();
            recovered.kept = replay.bindings().kept.clone();
            if proof_kind == 2 {
                assert!(lower(&replay, &recovered).is_ok());
                let mut missing = serde_json::to_value(&object).unwrap();
                for output in missing["source_outputs"].as_array_mut().unwrap() {
                    output.as_object_mut().unwrap().remove("unproven_fields");
                }
                for definition in missing["definitions"].as_array_mut().unwrap() {
                    for output in definition["source_outputs"].as_array_mut().unwrap() {
                        output.as_object_mut().unwrap().remove("unproven_fields");
                    }
                }
                let missing = serde_json::from_value(missing).unwrap();
                let conservative = polylogue_query::plan::replay(
                    &missing,
                    polylogue_query::plan::catalog_fingerprint(),
                )
                .unwrap();
                assert_eq!(conservative.question_id(), replay.question_id());
                recovered.kept = conservative.bindings().kept.clone();
                let error = lower(&conservative, &recovered).unwrap_err();
                assert!(
                    error.to_string().contains("logical matching policy"),
                    "{text}: {error}"
                );
            } else {
                let error = lower(&replay, &recovered).unwrap_err();
                assert!(
                    error.to_string().contains("logical matching policy"),
                    "{text}: {error}"
                );
            }
        }
        let text = "result:legacy | fields text(body) as body | where body:needle";
        let plan = polylogue_query::plan::prepare(text, &options.bindings(), |_| Ok(())).unwrap();
        let replay = polylogue_query::plan::replay(
            &plan.query_object().unwrap(),
            polylogue_query::plan::catalog_fingerprint(),
        )
        .unwrap();
        let mut recovered = options.clone();
        recovered.kept = replay.bindings().kept.clone();
        assert!(lower(&replay, &recovered).is_ok());
    }
}
