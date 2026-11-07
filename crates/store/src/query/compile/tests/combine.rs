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
        "sessions where id:\"claude-code:a\" | fields id as label,input_tokens as n | union (sessions where repo:east | fields id as label,input_tokens+10 as n)",
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
