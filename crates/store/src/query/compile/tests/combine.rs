use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use serde_json::{Value, json};

fn fixture() -> (tempfile::TempDir, Fixture) {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    for (id, repo, tokens) in [("a", "east", 1), ("b", "east", 3), ("c", "west", 5)] {
        let session = fixture
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                repo: Some(repo),
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
        fixture
            .usage(
                message,
                &polylogue_model::Usage {
                    input: Some(tokens),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    (directory, fixture)
}

fn read(reader: &crate::Reader<'_>, text: &str) -> Vec<Value> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let mut rows = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    rows
}

#[test]
fn fusion_keeps_projected_values_order_and_full_right_operand_ranks() {
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    for (left, right, labels) in [
        (
            "desc",
            "where repo:east",
            ["claude-code:b", "claude-code:a"],
        ),
        ("asc", "", ["claude-code:a", "claude-code:b"]),
    ] {
        let text = format!(
            "sessions where repo:east | fields id as label,input_tokens as n | sort by n {left} | fuse (sessions {right} | fields id as label,input_tokens as n | sort by n desc) | fields label,n"
        );
        let rows = read(&reader, &text);
        assert_eq!(
            rows.iter()
                .map(|row| row["label"].as_str().unwrap())
                .collect::<Vec<_>>(),
            labels
        );
        assert!(rows.iter().all(|row| row["n"]
            == if row["label"] == "claude-code:a" {
                1
            } else {
                3
            }));
    }
    let union = read(
        &reader,
        "sessions where id:\"claude-code:a\" | fields id as label,input_tokens as n | union (sessions where repo:east | fields id as label,input_tokens + 10 as n)",
    );
    assert_eq!(
        union,
        [
            json!({"label":"claude-code:a","n":1}),
            json!({"label":"claude-code:b","n":13})
        ]
    );
    let after_projection = read(
        &reader,
        "sessions | fields id as label,input_tokens as n | fuse (sessions | sort by id) | fields label,n | sort by n desc | fuse (sessions | sort by input_tokens desc) | fields label,n",
    );
    assert_eq!(
        after_projection,
        [
            json!({"label":"claude-code:c","n":5}),
            json!({"label":"claude-code:b","n":3}),
            json!({"label":"claude-code:a","n":1})
        ]
    );
}

#[test]
fn a_union_fuse_chain_scores_each_original_operand_once() {
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let rows = read(
        &reader,
        "sessions where repo:east | fields id as label,input_tokens as n | sort by n | union (sessions where input_tokens>=3 | fields id as label,input_tokens as n | sort by n desc) | fuse (sessions where repo:east | sort by input_tokens desc)",
    );
    assert_eq!(
        rows.iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["claude-code:b", "claude-code:a", "claude-code:c"]
    );
    for (row, score) in rows.iter().zip([
        -(1.0 / 62.0 + 1.0 / 62.0 + 1.0 / 61.0),
        -(1.0 / 61.0 + 1.0 / 62.0),
        -1.0 / 61.0,
    ]) {
        assert!(
            (row["rank"].as_f64().unwrap() - score).abs() < 1e-12,
            "{rows:?}"
        );
    }
}

#[test]
fn volatile_operand_selection_is_shared_by_membership_values_and_ranking() {
    use std::sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    };
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let calls = Arc::new(AtomicI64::new(0));
    let random_calls = calls.clone();
    reader
        .conn()
        .unwrap()
        .create_scalar_function(
            "random",
            0,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8,
            move |_| Ok(random_calls.fetch_add(1, Ordering::SeqCst)),
        )
        .unwrap();
    let source = "sessions | fields id as label,input_tokens as n | sort by random() | limit 2";
    let selected = read(&reader, source);
    let source_calls = calls.swap(0, Ordering::SeqCst);
    assert!(source_calls > 0);
    let rows = read(
        &reader,
        &format!("{source} | fuse (sessions | sort by input_tokens desc)"),
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        source_calls,
        "the volatile source must be evaluated once"
    );
    let mut expected = selected
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let right = match row["label"].as_str().unwrap() {
                "claude-code:c" => 1,
                "claude-code:b" => 2,
                _ => 3,
            };
            (
                row["label"].as_str().unwrap(),
                row["n"].clone(),
                -(1.0 / (61.0 + index as f64) + 1.0 / (60.0 + right as f64)),
            )
        })
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| left.2.total_cmp(&right.2).then(left.0.cmp(right.0)));
    assert_eq!(rows.len(), 2);
    for (row, (label, value, score)) in rows.iter().zip(expected) {
        assert_eq!(row["label"], label);
        assert_eq!(row["n"], value);
        assert!((row["rank"].as_f64().unwrap() - score).abs() < 1e-12);
    }
}

#[test]
fn measurements_share_the_scoped_combination_selection() {
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let program = compile("sessions where repo:east | fields id as label,input_tokens as n | sort by n | fuse (sessions | sort by input_tokens desc) | measure time", &opts(&[])).unwrap();
    use crate::query::execute::QueryPopulation;
    let answer = reader
        .query_keys(&program, QueryPopulation::Answer)
        .unwrap();
    let selected = reader
        .query_keys(&program, QueryPopulation::MeasureSelection)
        .unwrap();
    let population = reader
        .query_keys(&program, QueryPopulation::MeasurePopulation)
        .unwrap();
    assert_eq!(answer.len(), 2);
    assert_eq!(selected, answer);
    assert_eq!(population.len(), 3);
}

#[test]
fn snapshot_operands_refuse_without_reconstructing_current_native_rows() {
    let mut options = opts(&[]);
    options.kept.insert(
        "captured".into(),
        Kept {
            output_metadata: Default::default(),
            shape: Some(Shape::Rows),
            native_identity: true,
            unit: "sessions".into(),
            columns: c("sessions").columns,
            match_columns: vec![],
        },
    );
    for text in [
        "sessions | fuse (result:captured)",
        "result:captured | fuse (sessions)",
    ] {
        let error = compile(text, &options).unwrap_err();
        assert_eq!(error.kind(), "not_built", "{text}: {error}");
    }
}

#[test]
fn projected_set_operands_refuse_reference_text_and_reference_target_mismatches() {
    let options = opts(&[]);
    for (left, right) in [
        (
            "sessions where origin:claude-code | fields id",
            "sessions where origin:codex | fields text(id) as id",
        ),
        (
            "sessions where origin:claude-code | fields text(id) as id",
            "sessions where origin:codex | fields id",
        ),
        (
            "messages | fields id as ref",
            "messages | fields session as ref",
        ),
    ] {
        compile(left, &options).unwrap();
        compile(right, &options).unwrap();
        for operation in ["union", "diff"] {
            let text = format!("{left} | {operation} ({right})");
            let error = compile(&text, &options).unwrap_err();
            assert_eq!(error.kind(), "invalid_query", "{text}: {error}");
        }
        // Over one native row domain, intersect and except compare row
        // identities and keep the left operand's columns.
        for operation in ["intersect", "except"] {
            let text = format!("{left} | {operation} ({right})");
            let left_columns = compile(left, &options).unwrap().columns;
            assert_eq!(
                compile(&text, &options).unwrap().columns,
                left_columns,
                "{text}"
            );
        }
    }
}

#[test]
fn compatible_projected_sets_keep_plain_text_and_matching_reference_metadata() {
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let options = opts(&[]);
    let text = "sessions where repo:east | fields text(id) as id | union (sessions where repo:west | fields text(id) as id) | fields id";
    let plan = polylogue_query::compile(text, &options).unwrap();
    let program = lower(&plan, &options).unwrap();
    assert_eq!(program.columns.len(), 1);
    assert_eq!(program.columns[0].kind, ColKind::Text);
    assert_eq!(program.columns[0].reference, None);
    let schema = plan.output_schema_for_columns(&program.columns).unwrap();
    assert_eq!(
        schema.fields[0].kind,
        query_language::catalog::FieldKind::Text
    );
    assert_eq!(schema.fields[0].unit, None);
    let mut rows = reader.query_rows(&program).unwrap();
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    assert_eq!(
        rows,
        [
            json!({"id":"claude-code:a"}),
            json!({"id":"claude-code:b"}),
            json!({"id":"claude-code:c"})
        ]
    );

    let plan = polylogue_query::compile(
        "messages | fields id as ref | union (messages | fields id as ref) | fields ref",
        &options,
    )
    .unwrap();
    let program = lower(&plan, &options).unwrap();
    assert_eq!(program.columns[0].reference, Some(ColumnReference::Message));
    let schema = plan.output_schema_for_columns(&program.columns).unwrap();
    assert_eq!(
        schema.fields[0].kind,
        query_language::catalog::FieldKind::Reference
    );
    assert_eq!(schema.fields[0].unit.as_deref(), Some("messages"));
}

fn mixed_duration_archive() -> crate::fixture::writer::WriterArchive {
    use crate::fixture::writer::{WriterArchive, bash};
    let mut archive = WriterArchive::new();
    let mut messages = bash("mixed-duration", 1, 1_000, "first", true, "first result");
    let mut last = bash("mixed-duration", 2, 4_000, "last", true, "last result");
    last[1].at = polylogue_model::Timestamp::from_unix_ms(6_500);
    for block in &mut last[1].blocks {
        if let polylogue_model::Block::ToolResult {
            reported_duration_ms,
            ..
        } = block
        {
            *reported_duration_ms = Some(2_500);
        }
    }
    messages.extend(last);
    archive.write("mixed-duration", messages);
    archive
}

#[test]
fn compatible_duration_operands_normalize_real_values_in_both_directions() {
    let archive = mixed_duration_archive();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let reported = "tool_calls where command:last | fields reported_duration as d";
    let distance = "tool_calls where command:first AND before(tool_calls where command:last) within 5s | fields match.distance as d";
    for (left, right, kind, expected) in [
        (reported, distance, ColKind::Duration, [2_500.0, 3_000.0]),
        (distance, reported, ColKind::DurationSeconds, [2.5, 3.0]),
    ] {
        let text = format!("{left} | union ({right}) | sort by d");
        let plan = polylogue_query::compile(&text, &opts(&[])).unwrap();
        let program = lower(&plan, &opts(&[])).unwrap();
        assert_eq!(program.columns[0].kind, kind);
        let schema = plan.output_schema_for_columns(&program.columns).unwrap();
        assert_eq!(
            schema.fields[0].kind,
            query_language::catalog::FieldKind::Duration
        );
        let rows = read(&reader, &text);
        assert_eq!(rows.len(), 2, "{text}: {rows:?}");
        assert_eq!(
            rows.iter()
                .map(|row| row["d"].as_f64().unwrap())
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn union_members_without_an_optional_witness_keep_unknown_values_and_provenance() {
    let archive = mixed_duration_archive();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let witnessed =
        "tool_calls where command:first AND before(tool_calls where command:last) within 5s";
    let plain = "tool_calls where command:last";
    assert_eq!(
        read(
            &reader,
            &format!("{witnessed} | fields command,match.command,match.message.id")
        ),
        [
            json!({"command":"first", "match.command":"last", "match.message.id":"claude-code:mixed-duration-c2"})
        ]
    );
    let text = format!(
        "{witnessed} | union ({plain}) | sort by command | fields command,match.command,match.message.id"
    );
    assert_eq!(
        read(&reader, &text),
        [
            json!({"command":"first", "match.command":"last", "match.message.id":"claude-code:mixed-duration-c2"}),
            json!({"command":"last", "match.command":null, "match.message.id":null}),
        ]
    );
    let unknown =
        format!("{witnessed} | union ({plain}) | where match.command:unknown | fields command");
    assert_eq!(read(&reader, &unknown), [json!({"command":"last"})]);
}

#[test]
fn equivalent_message_targets_bind_and_keep_their_reference_metadata() {
    let archive = mixed_duration_archive();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let text = "tool_calls | fields message as ref | union (tool_calls | fields message.id as ref) | fields ref";
    let plan = polylogue_query::compile(text, &opts(&[])).unwrap();
    let program = lower(&plan, &opts(&[])).unwrap();
    assert_eq!(program.columns[0].reference, Some(ColumnReference::Message));
    let schema = plan.output_schema_for_columns(&program.columns).unwrap();
    assert_eq!(
        schema.fields[0].kind,
        query_language::catalog::FieldKind::Reference
    );
    assert_eq!(schema.fields[0].unit.as_deref(), Some("messages"));
    assert_eq!(
        read(&reader, text),
        [
            json!({"ref":"claude-code:mixed-duration-c1"}),
            json!({"ref":"claude-code:mixed-duration-c2"}),
        ]
    );
}

#[test]
fn diff_normalizes_removed_duration_values_without_changing_identity_rank_or_time() {
    let archive = mixed_duration_archive();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    for alias in ["d", "clustered_time"] {
        let reported =
            format!("tool_calls where command:last | fields reported_duration as {alias},at");
        let distance = format!(
            "tool_calls where command:first AND before(tool_calls where command:last) within 5s | fields match.distance as {alias},at"
        );
        for (after, before, kind, joined, removed, joined_at, removed_at) in [
            (
                &reported,
                &distance,
                ColKind::Duration,
                2_500.0,
                3_000.0,
                4_000,
                1_000,
            ),
            (
                &distance,
                &reported,
                ColKind::DurationSeconds,
                3.0,
                2.5,
                1_000,
                4_000,
            ),
        ] {
            let text = format!(
                "{after} | diff ({before}) | sort by change | fields {alias},at,change,old_rank,new_rank"
            );
            let program = compile(&text, &opts(&[])).unwrap();
            assert_eq!(program.columns[0].kind, kind);
            assert!(
                !program
                    .notes
                    .iter()
                    .any(|note| note.contains("clustered_time uses")),
                "an alias does not establish a clustering construct"
            );
            let rows = read(&reader, &text);
            assert_eq!(rows.len(), 2, "{text}: {rows:?}");
            assert_eq!(rows[0]["change"], "joined");
            assert_eq!(rows[0]["old_rank"], Value::Null);
            assert_eq!(rows[0]["new_rank"], 1);
            assert_eq!(rows[0][alias].as_f64(), Some(joined));
            assert_eq!(rows[1]["change"], "left");
            assert_eq!(rows[1]["old_rank"], 1);
            assert_eq!(rows[1]["new_rank"], Value::Null);
            assert_eq!(rows[1][alias].as_f64(), Some(removed));
            for (row, at) in rows.iter().zip([joined_at, removed_at]) {
                assert_eq!(
                    row["at"]
                        .as_str()
                        .unwrap()
                        .parse::<jiff::Timestamp>()
                        .unwrap()
                        .as_millisecond(),
                    at
                );
            }
        }
    }
}

#[test]
fn diff_moved_rows_keep_after_values_and_unknowns_over_before_values() {
    let archive = mixed_duration_archive();
    let reads = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = reads.reader().unwrap();
    let text = "tool_calls | fields message as id,reported_duration as d | sort by id desc | diff (tool_calls where near(tool_calls where command:first) within 5s | fields message as id,match.distance as d | sort by id) | sort by id | fields id,d,change,old_rank,new_rank";
    assert_eq!(
        read(&reader, text),
        [
            json!({"id":"claude-code:mixed-duration-c1", "d":null, "change":"moved", "old_rank":1, "new_rank":2}),
            json!({"id":"claude-code:mixed-duration-c2", "d":2_500, "change":"moved", "old_rank":2, "new_rank":1}),
        ]
    );
}

#[test]
fn fused_count_and_membership_keep_rank_dependent_operand_selections() {
    let (_directory, fixture) = fixture();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let text = "sessions | sort by input_tokens desc | limit 1 | union (sessions | sort by input_tokens | limit 1) | fuse (sessions | sort by input_tokens desc)";
    let program = compile(text, &opts(&[])).unwrap();
    let parameters = crate::selection::values(&program.params);
    assert_eq!(
        reader
            .query_counts(&program.statements.count(program.counts_near), &parameters)
            .unwrap(),
        (2, 0)
    );
    let membership = reader
        .query(
            &format!(
                "SELECT s.id FROM ({}) k JOIN session s ON s.pk=k._pk ORDER BY s.id",
                program.statements.members(program.counts_near)
            ),
            &parameters,
        )
        .unwrap();
    assert_eq!(
        membership,
        [json!({"id":"claude-code:a"}), json!({"id":"claude-code:c"})]
    );
    let rows = read(&reader, text);
    assert_eq!(
        rows.iter().map(|row| row["id"].clone()).collect::<Vec<_>>(),
        [json!("claude-code:c"), json!("claude-code:a")]
    );
}
