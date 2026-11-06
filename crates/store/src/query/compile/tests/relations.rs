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

fn rows(fixture: &Fixture, text: &str) -> Vec<Value> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let mut rows = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    rows
}

#[test]
fn projection_aliases_do_not_overwrite_native_identity_or_order_slots() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(
            &fixture,
            "sessions | sort by id desc | limit 2 | fields id as _pk, input_tokens as _id, repo as _o0 | where _id < 4 | fields _pk, _id, _o0"
        ),
        [json!({"_pk":"claude-code:b", "_id":3, "_o0":"east"})]
    );
}

#[test]
fn selected_groups_aggregate_their_original_rows_after_projection() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields repo as region, input_tokens as n | group by region | where count > 1 | agg sum(n) as total"
        ),
        [json!({"region":"east", "count":2, "total":4.0, "unknown_n":0})]
    );
}

#[test]
fn key_join_preserves_pair_multiplicity_and_projected_aliases() {
    let (_directory, fixture) = fixture();
    let answer = rows(
        &fixture,
        "sessions | fields id as left_id, repo as region | join (sessions | fields id as right_id, repo as region) on region = region | fields left_id, match.right_id as right_id | sort by left_id, right_id",
    );
    assert_eq!(
        answer,
        [
            json!({"left_id":"claude-code:a", "right_id":"claude-code:a"}),
            json!({"left_id":"claude-code:a", "right_id":"claude-code:b"}),
            json!({"left_id":"claude-code:b", "right_id":"claude-code:a"}),
            json!({"left_id":"claude-code:b", "right_id":"claude-code:b"}),
            json!({"left_id":"claude-code:c", "right_id":"claude-code:c"}),
        ]
    );
}

#[test]
fn grouped_first_returns_the_selected_input_rows() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(
            &fixture,
            "sessions | sort by id | fields id, repo | group by repo | first 1 | fields id | sort by id"
        ),
        [json!({"id":"claude-code:a"}), json!({"id":"claude-code:c"})]
    );
}
