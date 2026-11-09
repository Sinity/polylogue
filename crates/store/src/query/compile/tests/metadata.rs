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
        "sessions where exists tool_calls() | matches",
        "sessions where seq(action:shell -> outcome:error) | matches",
        "sessions where exists tool_calls() | limit 1 | matches",
        "sessions where exists tool_calls() | distinct | matches",
        "sessions where exists tool_calls() | fields id as source, started as began | matches",
        "sessions where exists tool_calls() | fields repo as source.repo | matches",
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
            "sessions where exists tool_calls(outcome:error) | matches",
            "query:matching | sort by step.command",
        ),
        (
            "sessions where exists tool_calls(outcome:error)",
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
            "sessions where exists tool_calls() | {selection} | matches | fields step.command as command | sort by command"
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
fn canonical_replay_recovers_kept_declarations_and_lowers_matching() {
    use polylogue_model::kept::OutputMetadata;
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
    options.kept.insert(
        "kept".into(),
        Kept {
            output_metadata: OutputMetadata::captured(schema, None),
            unit: "sessions".into(),
            columns,
            match_columns: vec![],
            shape: None,
            native_identity: false,
        },
    );
    let base =
        polylogue_query::plan::prepare("result:kept", &options.bindings(), |_| Ok(())).unwrap();
    let mut bindings = options.bindings();
    bindings
        .add_saved("retained", &base, "retained-revision", "retained-root")
        .unwrap();
    options.saved = bindings.saved;
    for source in ["result:kept", "query:retained"] {
        for field in ["body", "payload"] {
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
            assert_eq!(replay.bindings().kept, options.kept, "{text}");
            // Lower using only the recovered immutable declaration.
            let mut recovered = options.clone();
            recovered.kept = replay.bindings().kept.clone();
            assert!(lower(&replay, &recovered).is_ok(), "{text}");
        }
    }
}

#[test]
fn match_parent_text_alias_keeps_explicit_absence_of_reference_metadata() {
    let text = "sessions where exists tool_calls() | fields text(id) as source | matches | fields parent.source";
    let options = opts(&[]);
    let plan = polylogue_query::compile(text, &options).unwrap();
    let program = lower(&plan, &options).unwrap();
    assert_eq!(program.columns.len(), 1);
    assert_eq!(program.columns[0].name, "parent.source");
    assert_eq!(program.columns[0].kind, ColKind::Text);
    assert_eq!(program.columns[0].reference, None);
    let schema = plan.output_schema_for_columns(&program.columns).unwrap();
    assert_eq!(
        schema.fields[0].kind,
        query_language::catalog::FieldKind::Text
    );
    assert_eq!(schema.fields[0].unit, None);
}

#[test]
fn reference_targets_agree_between_catalog_native_columns_and_projections() {
    use query_language::catalog::FieldKind;
    let options = opts(&[]);
    let catalog = options.catalog();
    for (unit, name, reference, default_column) in [
        ("sessions", "id", ColumnReference::Session, true),
        ("documents", "id", ColumnReference::Document, true),
        ("messages", "id", ColumnReference::Message, true),
        ("messages", "session", ColumnReference::Session, true),
        ("blocks", "message", ColumnReference::Message, true),
        ("blocks", "session", ColumnReference::Session, true),
        ("tool_calls", "message", ColumnReference::Message, true),
        ("tool_calls", "session", ColumnReference::Session, true),
        ("events", "session", ColumnReference::Session, true),
        ("files", "session", ColumnReference::Session, true),
        ("links", "parent", ColumnReference::Session, true),
        ("links", "child", ColumnReference::Session, true),
        ("links", "branch_point", ColumnReference::Message, false),
        ("assertions", "reference", ColumnReference::Assertion, true),
        ("turns", "id", ColumnReference::Message, true),
        ("turns", "answer", ColumnReference::Message, true),
        ("turns", "session", ColumnReference::Session, true),
        ("deliveries", "recipient", ColumnReference::Session, true),
        ("deliveries", "session", ColumnReference::Session, false),
        ("live_intervals", "session", ColumnReference::Session, true),
        ("processes", "session", ColumnReference::Session, true),
    ] {
        let field = catalog.field(unit, name).unwrap();
        assert_eq!(field.kind, FieldKind::Reference, "{unit}.{name}");
        assert_eq!(
            field.unit.as_deref(),
            Some(reference.unit()),
            "{unit}.{name}"
        );
        let native = compile(unit, &options).unwrap();
        let emitted = native.columns.iter().find(|column| column.name == name);
        assert_eq!(emitted.is_some(), default_column, "{unit}.{name}");
        if let Some(column) = emitted {
            assert_eq!(column.reference, Some(reference), "{unit}.{name}");
        }
        let text = format!("{unit} | fields {name}");
        let plan = polylogue_query::compile(&text, &options).unwrap();
        let program = lower(&plan, &options).unwrap();
        assert_eq!(program.columns[0].reference, Some(reference), "{text}");
        let schema = plan.output_schema_for_columns(&program.columns).unwrap();
        assert_eq!(schema.fields[0].kind, FieldKind::Reference, "{text}");
        assert_eq!(
            schema.fields[0].unit.as_deref(),
            Some(reference.unit()),
            "{text}"
        );
    }
    for unit in ["live_intervals", "processes"] {
        assert_eq!(
            catalog
                .unit(unit)
                .unwrap()
                .fields
                .iter()
                .filter(|field| field.name == "session")
                .count(),
            1
        );
    }
    let matches = compile("sessions where exists tool_calls() | matches", &options).unwrap();
    for name in ["parent", "step"] {
        assert_eq!(
            matches
                .columns
                .iter()
                .find(|column| column.name == name)
                .unwrap()
                .reference,
            None
        );
    }
    for (name, reference) in [
        ("step.message", ColumnReference::Message),
        ("step.session", ColumnReference::Session),
    ] {
        assert_eq!(
            matches
                .columns
                .iter()
                .find(|column| column.name == name)
                .unwrap()
                .reference,
            Some(reference)
        );
    }
}

#[test]
fn reference_containers_and_unconstrained_ids_keep_their_physical_shapes() {
    use query_language::catalog::FieldKind;
    let options = opts(&[]);
    let catalog = options.catalog();
    for (name, target) in [("sessions", "sessions"), ("messages", "messages")] {
        let field = catalog.field("changes", name).unwrap();
        assert_eq!(field.kind, FieldKind::List(Box::new(FieldKind::Reference)));
        assert_eq!(field.unit.as_deref(), Some(target));
        let declaration = polylogue_query::fields::lookup(Unit::Changes, name).unwrap();
        assert_eq!(declaration.column_kind(), ColKind::Ids);
        assert_eq!(
            polylogue_query::fields::column_reference(Unit::Changes, name),
            None
        );
    }
    for (unit, name) in [
        (Unit::Assertions, "id"),
        (Unit::Assertions, "root"),
        (Unit::Processes, "tool_call"),
        (Unit::Changes, "ref"),
    ] {
        assert_eq!(polylogue_query::fields::reference_target(unit, name), None);
        assert_eq!(polylogue_query::fields::column_reference(unit, name), None);
        assert_eq!(catalog.field(unit.name(), name).unwrap().unit, None);
    }
    let turns = compile("turns", &options).unwrap();
    let prompts = turns
        .columns
        .iter()
        .find(|column| column.name == "prompts")
        .unwrap();
    assert_eq!(prompts.kind, ColKind::Text);
    assert_eq!(prompts.reference, None);
}
