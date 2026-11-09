use super::*;
use serde_json::json;

fn answer(f: &Fixture, text: &str) -> Vec<Value> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader.prepare_selection(text, &options()).unwrap();
    let mut values = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            values.push(row.values);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    values
}

#[test]
fn kept_stages_use_captured_values_after_native_rows_change() {
    let (_dir, mut f) = fixture();
    let body = {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        let prepared = reader
            .prepare_selection("sessions | fields id, last_active", &options())
            .unwrap();
        let program = prepared.program();
        let mut rows = reader.query_rows(program).unwrap();
        let identities: Vec<_> = rows
            .iter()
            .map(|row| match &program.row_basis {
                crate::query::compile::RowBasis::Native { columns, .. } => columns
                    .iter()
                    .map(|name| row[name].clone())
                    .collect::<Vec<_>>(),
                _ => panic!("native capture"),
            })
            .collect();
        for row in &mut rows {
            crate::query::execute::present(program, row).unwrap();
        }
        json!({"unit":"sessions", "shape":"rows", "columns":program.columns, "rows":rows, "identities":identities})
    };
    let captured = body["rows"].as_array().unwrap().clone();
    f.kept_result(
        "snapshot",
        "sessions | fields id, last_active",
        &body.to_string(),
    )
    .unwrap();
    f.message(
        1,
        MessageFacts {
            id: "test:one:new",
            at: Some(900),
            role: "assistant",
            written_by: "model",
            text: Some("new native content"),
            ..Default::default()
        },
    )
    .unwrap();
    f.derive().unwrap();
    assert_eq!(answer(&f, "result:snapshot"), captured);
    assert_ne!(answer(&f, "sessions | fields id, last_active"), captured);
    assert_eq!(
        answer(
            &f,
            "result:snapshot | fields text(id) as label | where label:ONE"
        ),
        [json!({"label":"test:one"})]
    );
    assert_eq!(
        answer(&f, "result:snapshot | sort by last_active | fields id"),
        [
            json!({"id":"test:one"}),
            json!({"id":"test:two"}),
            json!({"id":"test:three"})
        ]
    );
    assert_eq!(
        answer(
            &f,
            "result:snapshot | group by id | sort by id | fields id, count"
        ),
        [
            json!({"id":"test:one","count":1}),
            json!({"id":"test:three","count":1}),
            json!({"id":"test:two","count":1})
        ]
    );
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(
        reader
            .prepare_selection("result:snapshot | fields origin", &options())
            .is_err()
    );
    let prepared = reader
        .prepare_selection("result:snapshot", &options())
        .unwrap();
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            assert!(matches!(row.identity, Some(RowIdentity::Entity { .. })));
            assert!(row.target.is_some());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
}

#[test]
fn duplicate_kept_items_have_distinct_stable_cursor_identity_without_entity_proof() {
    let (_dir, mut f) = fixture();
    f.kept_result(
        "duplicates",
        "synthetic capture",
        &json!({
            "unit":"sessions", "shape":"rows", "columns":[{"name":"label","kind":"text"}],
            "rows":[{"label":"same"},{"label":"same"}]
        })
        .to_string(),
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let prepared = reader
        .prepare_selection("result:duplicates", &options())
        .unwrap();
    let mut identities = vec![];
    let mut after = None;
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            assert_eq!(row.values, json!({"label":"same"}));
            assert!(row.target.is_none());
            assert!(matches!(row.identity, Some(RowIdentity::Distinct { .. })));
            identities.push(row.identity);
            after = Some(row.position);
            Ok(ControlFlow::Break(()))
        })
        .unwrap();
    reader
        .visit_selection::<SelectionError>(&prepared, after.as_ref(), None, |row| {
            assert_eq!(row.values, json!({"label":"same"}));
            assert!(row.target.is_none());
            identities.push(row.identity);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(identities.len(), 2);
    assert_ne!(identities[0], identities[1]);
    let mut repeated = vec![];
    reader
        .visit_selection::<SelectionError>(&prepared, None, None, |row| {
            repeated.push(row.identity);
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    assert_eq!(identities, repeated);
}

#[test]
fn kept_group_shape_and_private_aliases_are_installed_before_stages() {
    let (_dir, mut f) = fixture();
    f.kept_result("groups", "synthetic capture", &json!({
        "unit":"sessions", "shape":"groups", "columns":[{"name":"_pk","kind":"text"},{"name":"count","kind":"number"}],
        "rows":[{"_pk":"west","count":2},{"_pk":"east","count":5}]
    }).to_string()).unwrap();
    assert_eq!(
        answer(&f, "result:groups | sort by count desc | fields _pk, count"),
        [
            json!({"_pk":"east","count":5}),
            json!({"_pk":"west","count":2})
        ]
    );
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .prepare_selection("result:groups", &options())
            .unwrap()
            .program()
            .shape,
        polylogue_query::Shape::Groups
    );
}

#[test]
fn kept_join_and_aggregate_use_captured_numeric_slots() {
    let (_dir, mut f) = fixture();
    f.kept_result("numbers", "synthetic capture", &json!({
        "unit":"sessions", "shape":"rows", "columns":[{"name":"id","kind":"text"},{"name":"input_tokens","kind":"number"}],
        "rows":[{"id":"test:one","input_tokens":2},{"id":"test:two","input_tokens":7}]
    }).to_string()).unwrap();
    assert_eq!(
        answer(&f, "result:numbers | agg sum(input_tokens) as total"),
        [json!({"total":9,"unknown_input_tokens":0})]
    );
    assert_eq!(
        answer(
            &f,
            "sessions | join (result:numbers) on id = id | agg sum(match.input_tokens) as total"
        ),
        [json!({"total":9,"unknown_match_input_tokens":0})]
    );
}

#[test]
fn kept_same_name_text_casts_retain_their_bound_matching_policy() {
    use query_language::catalog::FieldKind;
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    for (id, cwd) in [
        ("test:one", "/work/module/src"),
        ("test:two", "/work/module-other"),
    ] {
        fixture
            .session(SessionFacts {
                id,
                cwd: Some(cwd),
                ..Default::default()
            })
            .unwrap();
    }
    let mut physical_paths = vec![];
    for (id, text, id_kind, path_kind) in [
        (
            "native-policy",
            "sessions | fields id, cwd",
            FieldKind::Reference,
            FieldKind::Subtree,
        ),
        (
            "cast-policy",
            "sessions | fields text(id) as id, text(cwd) as cwd",
            FieldKind::Text,
            FieldKind::Text,
        ),
    ] {
        let body = {
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let prepared = reader.prepare_selection(text, &options()).unwrap();
            let schema = prepared.plan().output_schema().unwrap();
            assert_eq!(schema.fields[0].kind, id_kind);
            assert_eq!(schema.fields[1].kind, path_kind);
            let program = prepared.program();
            physical_paths.push(program.columns[1].clone());
            let mut rows = reader.query_rows(program).unwrap();
            for row in &mut rows {
                crate::query::execute::present(program, row).unwrap();
            }
            json!({"unit":"sessions", "shape":"rows", "columns":program.columns, "rows":rows,
                "output_metadata":polylogue_model::kept::OutputMetadata::captured(schema, None)})
        };
        fixture.kept_result(id, text, &body.to_string()).unwrap();
    }
    // These physically identical columns need the producer's logical declaration.
    assert_eq!(physical_paths[0], physical_paths[1]);
    assert!(answer(&fixture, "result:native-policy | where id:one").is_empty());
    assert_eq!(
        answer(&fixture, "result:cast-policy | where id:one | fields id"),
        [json!({"id":"test:one"})]
    );
    assert_eq!(
        answer(
            &fixture,
            "result:native-policy | where cwd:/work/module | fields id"
        ),
        [json!({"id":"test:one"})]
    );
    assert_eq!(
        answer(
            &fixture,
            "result:cast-policy | where cwd:/work/module | fields id | sort by id"
        ),
        [json!({"id":"test:one"}), json!({"id":"test:two"})]
    );
}

#[test]
fn captured_logical_declarations_must_agree_with_physical_columns_and_reference_hints() {
    use query_language::catalog::{Field, FieldKind, OutputSchema};
    let mut kept = polylogue_query::Kept {
        output_metadata: polylogue_model::kept::OutputMetadata::captured(
            OutputSchema {
                fields: vec![Field {
                    name: "id".into(),
                    kind: FieldKind::Text,
                    ..Default::default()
                }],
                ..Default::default()
            },
            None,
        ),
        shape: Some(polylogue_query::Shape::Rows),
        native_identity: false,
        unit: "sessions".into(),
        columns: vec![polylogue_query::Column {
            name: "id".into(),
            kind: polylogue_query::ColKind::Text,
            reference: Some(polylogue_query::ColumnReference::Session),
        }],
        match_columns: vec![],
    };
    assert!(kept.validate_output_metadata().is_err());
    kept.columns[0].reference = None;
    assert!(kept.validate_output_metadata().is_ok());
    kept.columns[0].kind = polylogue_query::ColKind::Money;
    assert!(kept.validate_output_metadata().is_err());
    kept.columns[0].kind = polylogue_query::ColKind::Text;
    kept.columns[0].name = "other".into();
    assert!(kept.validate_output_metadata().is_err());
}
