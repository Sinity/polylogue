use super::*;
use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
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
            "sessions | fields repo as region, input_tokens as n | group by region | agg sum(n) as total | where count > 1"
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

#[test]
fn step_order_counts_every_call_before_filtering_and_ignores_message_sequence_gaps() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:steps",
            ..Default::default()
        })
        .unwrap();
    for (message_id, seq, commands) in [
        ("claude-code:first", 1, vec!["first", "middle"]),
        ("claude-code:last", 1000, vec!["last"]),
    ] {
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: message_id,
                    role: "assistant",
                    written_by: "model",
                    seq: Some(seq),
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
    assert!(rows(&fixture, "tool_calls where command = first AND before(tool_calls where command = last) within 1 step").is_empty());
    assert_eq!(
        rows(
            &fixture,
            "tool_calls where command = first AND before(tool_calls where command = last) within 2 steps | fields command, match.distance"
        ),
        [json!({"command":"first", "match.distance":2})]
    );
    assert_eq!(
        rows(
            &fixture,
            "tool_calls where command = first AND before(tool_calls where command = middle) within 1 step | fields command, match.distance"
        ),
        [json!({"command":"first", "match.distance":1})]
    );
}

#[test]
fn distinct_uses_declared_row_identity_after_projection() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(&fixture, "sessions | sort by id | fields repo | distinct"),
        [
            json!({"repo":"east"}),
            json!({"repo":"east"}),
            json!({"repo":"west"})
        ]
    );
}

#[test]
fn diff_aligns_logical_values_when_the_two_queries_use_different_sql_slots() {
    let (_directory, fixture) = fixture();
    let answer = rows(
        &fixture,
        "sessions | sort by id | limit 2 | fields id as label, input_tokens as n | diff (sessions | sort by id | offset 1 | fields input_tokens as n, id as label) | fields label, n, change, old_rank, new_rank",
    );
    assert_eq!(
        answer,
        [
            json!({"label":"claude-code:a", "n":1, "change":"joined", "old_rank":null, "new_rank":1}),
            json!({"label":"claude-code:b", "n":3, "change":"moved", "old_rank":1, "new_rank":2}),
            json!({"label":"claude-code:c", "n":5, "change":"left", "old_rank":2, "new_rank":null}),
        ]
    );
}

#[test]
fn historical_read_positions_refuse_without_substituting_current_membership() {
    for text in [
        "sessions | as of seq 42",
        "sessions | fields id | as of 2026-10-02",
        "sessions | group by repo | as of seq 42",
    ] {
        let error = compile(text, &opts(&[])).unwrap_err();
        assert_eq!(error.kind(), "not_yet_available", "{text}: {error}");
        assert!(error.to_string().contains("evidence_until:"), "{error}");
    }
}

#[test]
fn selections_accept_the_full_unsigned_count_domain() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(
            &fixture,
            "sessions | limit 18446744073709551615 | fields id | sort by id"
        )
        .len(),
        3
    );
    assert!(rows(&fixture, "sessions | offset 18446744073709551615").is_empty());
}

#[test]
fn explicit_witness_choice_keeps_the_outer_row_and_uses_only_its_requested_order() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:outer",
            ..Default::default()
        })
        .unwrap();
    for (id, role, at) in [
        ("claude-code:early", "user", 1000),
        ("claude-code:late", "user", 9000),
        ("claude-code:outer-message", "assistant", 10000),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role,
                    written_by: "model",
                    at: Some(at),
                    cost_nusd: (id == "claude-code:early").then_some(1_000_000_000),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    assert_eq!(
        rows(
            &fixture,
            "sessions where id = claude-code:outer AND after(messages where role:user) within 20s | fields id, match.id, match.distance"
        ),
        [json!({"id":"claude-code:outer", "match.id":"claude-code:late", "match.distance":-1.0})]
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions where id = claude-code:outer AND after(messages where role:user) within 20s pick first by at | fields id, match.id, match.distance"
        ),
        [json!({"id":"claude-code:outer", "match.id":"claude-code:early", "match.distance":-9.0})]
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions where id = claude-code:outer AND after(messages where role:user) within 20s pick first by abs(match.distance) desc | fields match.id, abs(match.distance) as elapsed | where elapsed > 5s"
        ),
        [json!({"match.id":"claude-code:early", "elapsed":9.0})]
    );
    for direction in ["asc", "desc"] {
        assert_eq!(
            rows(
                &fixture,
                &format!(
                    "sessions where id = claude-code:outer AND after(messages where role:user) within 20s pick first by cost {direction} | fields match.id"
                )
            ),
            [json!({"match.id":"claude-code:early"})]
        );
    }
    assert_eq!(
        rows(
            &fixture,
            "sessions where id = claude-code:outer AND after(messages where role:user | fields at as moment) within 20s pick first by match.id desc | where match.moment = 1970-01-01T00:00:09Z | fields id"
        ),
        [json!({"id":"claude-code:outer"})]
    );
}

#[test]
fn explicit_absolute_distance_choice_keeps_exact_nanosecond_ties() {
    use polylogue_model::{Precision, TimeEvidence, TimeQuality, Timestamp};
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:precise",
            ..Default::default()
        })
        .unwrap();
    for (id, role, at) in [
        ("claude-code:outer", "assistant", "1970-01-01T00:00:00Z"),
        (
            "claude-code:z-near",
            "user",
            "9999-12-30T00:00:00.000000001Z",
        ),
        (
            "claude-code:a-far",
            "user",
            "9999-12-30T00:00:00.000000002Z",
        ),
    ] {
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role,
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        fixture
            .message_time(
                message,
                Some(Timestamp::parse_rfc3339(at).unwrap()),
                TimeEvidence {
                    quality: TimeQuality::Intrinsic,
                    precision: Some(Precision::Instant),
                    until: None,
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    assert_eq!(
        rows(
            &fixture,
            "messages where id = claude-code:outer AND before(messages where role:user) within 3000000d pick first by abs(match.distance) | fields match.id"
        ),
        [json!({"match.id":"claude-code:z-near"})]
    );
}

#[test]
fn filters_read_a_grouped_join_column_from_its_materialized_slot() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(
            &fixture,
            "sessions | join (sessions | group by repo) on repo = repo | where match.count > 1 | fields id | sort by id"
        ),
        [json!({"id":"claude-code:a"}), json!({"id":"claude-code:b"})]
    );
}
