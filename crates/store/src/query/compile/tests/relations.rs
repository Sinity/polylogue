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
        [json!({"region":"east", "count":2, "total":4, "unknown_n":0})]
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
    assert_eq!(
        rows(
            &fixture,
            "sessions where exists tool_call(command:last) | matches | fields step.command as command, step.position as ordinal"
        ),
        [json!({"command":"last", "ordinal":3})]
    );
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

#[test]
fn order_conditions_read_the_input_witness_and_publish_only_the_shared_exposure() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:outer",
            ..Default::default()
        })
        .unwrap();
    for (id, role, at) in [
        ("claude-code:prior", "user", 9000),
        ("claude-code:a", "user", 8000),
        ("claude-code:b", "user", 7000),
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
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    let initial = "sessions where id = claude-code:outer AND after(messages where id = claude-code:prior) within 20s";
    let a = "after(messages where id = claude-code:a) within 20s";
    let b = "after(messages where id = claude-code:b) within 20s";
    let absent = "after(messages where id = claude-code:missing) within 20s";
    let selected = |condition: &str| {
        rows(
            &fixture,
            &format!("{initial} | where {condition} | fields id, match.id"),
        )
    };
    let prior = [json!({"id":"claude-code:outer", "match.id":"claude-code:prior"})];
    for condition in [
        format!("{a} AND {b}"),
        format!("{b} AND {a}"),
        format!("{a} OR {b}"),
        format!("{b} OR {a}"),
        format!("NOT {absent}"),
    ] {
        assert_eq!(selected(&condition), prior, "{condition}");
    }
    let next = [json!({"id":"claude-code:outer", "match.id":"claude-code:a"})];
    for condition in [
        format!("{a} AND match.id = claude-code:prior"),
        format!("match.id = claude-code:prior AND {a}"),
        format!("{a} AND ({b} OR NOT {absent})"),
    ] {
        assert_eq!(selected(&condition), next, "{condition}");
    }
    assert!(selected(&format!("{a} AND match.id = claude-code:a")).is_empty());
}

#[test]
fn projected_natural_matches_follow_the_bound_field_kind_and_unicode_case_rules() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    for (id, cwd) in [
        ("claude-code:École-A", "/work/src"),
        ("claude-code:ecole-B", "/work2/src"),
        ("claude-code:third", "/work"),
    ] {
        let session = fixture
            .session(SessionFacts {
                id,
                cwd: Some(cwd),
                ..Default::default()
            })
            .unwrap();
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("{id}-message"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        fixture
            .tool_call(
                session,
                message,
                CallFacts {
                    tool: "Bash",
                    action: "shell",
                    command: Some("cat"),
                    outcome: "ok",
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields text(id) as label | where label:éCOLE"
        ),
        [json!({"label":"claude-code:École-A"})]
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields text(id) as label | where label:ecole"
        ),
        [json!({"label":"claude-code:ecole-B"})]
    );
    assert!(
        rows(
            &fixture,
            r#"sessions | fields id as refid | where refid:"claude-code:école-A""#
        )
        .is_empty()
    );
    assert_eq!(
        rows(
            &fixture,
            r#"sessions | fields id as refid | where refid:"claude-code:École-A""#
        ),
        [json!({"refid":"claude-code:École-A"})]
    );
    assert!(
        rows(
            &fixture,
            "tool_calls | fields tool as exact | where exact:bash"
        )
        .is_empty()
    );
    assert_eq!(
        rows(
            &fixture,
            "tool_calls | fields tool as exact | where exact:Bash"
        )
        .len(),
        3
    );
    assert_eq!(
        rows(
            &fixture,
            "tool_calls | fields text(tool) as plain | where plain:ASH"
        )
        .len(),
        3
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields text(id) as outer_id | join (sessions | fields text(id) as label | where label:éCOLE) on outer_id = label | fields outer_id, match.label"
        ),
        [json!({"outer_id":"claude-code:École-A", "match.label":"claude-code:École-A"})]
    );
    let paths = [json!({"p":"/work"}), json!({"p":"/work/src"})];
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields cwd as p | where p ^ /work | sort by p"
        ),
        paths
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields cwd as p | where p:/work | sort by p"
        ),
        paths
    );
    assert_eq!(
        rows(
            &fixture,
            "sessions | fields text(cwd) as p | where p ^ /work"
        )
        .len(),
        3
    );
}

#[test]
fn natural_text_matches_read_the_input_alias_beside_a_new_order_witness() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:outer",
            ..Default::default()
        })
        .unwrap();
    for (id, role, at) in [
        ("claude-code:prior", "user", 9000),
        ("claude-code:next", "user", 8000),
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
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    let initial = "sessions where id = claude-code:outer AND after(messages where id = claude-code:prior) within 20s | fields id, text(match.id) as prior";
    let order = "after(messages where id = claude-code:next) within 20s";
    for predicate in [
        format!("{order} AND prior:PRIOR"),
        format!("prior:PRIOR AND {order}"),
    ] {
        assert_eq!(
            rows(
                &fixture,
                &format!("{initial} | where {predicate} | fields prior, match.id")
            ),
            [json!({"prior":"claude-code:prior", "match.id":"claude-code:next"})]
        );
    }
}

#[test]
fn grouped_reference_values_keep_field_metadata_without_native_row_targets() {
    use crate::selection::{RowIdentity, SelectionError};
    use std::ops::ControlFlow;
    let (_directory, fixture) = fixture();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    for text in ["sessions | group by id", "sessions | agg min(id)"] {
        let prepared = reader.prepare_selection(text, &opts(&[])).unwrap();
        assert_eq!(
            prepared.program().columns[0].reference,
            Some(ColumnReference::Session)
        );
        let mut seen = 0;
        reader
            .visit_selection::<SelectionError>(&prepared, None, None, |row| {
                assert!(row.target.is_none(), "{text}");
                assert!(
                    matches!(
                        row.identity,
                        Some(RowIdentity::Group { .. } | RowIdentity::Summary)
                    ),
                    "{text}"
                );
                seen += 1;
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        assert!(seen > 0);
    }
}

#[test]
fn native_aggregates_keep_child_rows_and_unknown_values() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:aggregates",
            ..Default::default()
        })
        .unwrap();
    for (id, cost, exits) in [
        ("claude-code:one", Some(2_000_000_000), vec![Some(0), None]),
        ("claude-code:two", None, vec![Some(1)]),
    ] {
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    cost_nusd: cost,
                    ..Default::default()
                },
            )
            .unwrap();
        for exit_code in exits {
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        outcome: "ok",
                        exit_code,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    assert_eq!(
        rows(&fixture, "messages | agg count(), sum(cost) as total"),
        [json!({"count":2, "total":2.0, "unknown_cost":1})]
    );
    assert_eq!(
        rows(
            &fixture,
            "tool_calls | sort by exit_code desc | limit 2 | agg count(), sum(exit_code) as total"
        ),
        [json!({"count":2, "total":1, "unknown_exit_code":0})]
    );
    assert_eq!(
        rows(
            &fixture,
            "tool_calls | group by message | agg count(), sum(exit_code) as total | sort by message"
        ),
        [
            json!({"message":"claude-code:one", "count":2, "total":0, "unknown_exit_code":1}),
            json!({"message":"claude-code:two", "count":1, "total":1, "unknown_exit_code":0})
        ]
    );
}

#[test]
fn explanation_clauses_use_bound_syntax_instead_of_original_byte_spans() {
    let text = "messages where active:* (((active:false AND (active:false OR active:false)) AND active:false)) walrus";
    let program = compile(text, &opts(&[])).unwrap();
    assert!(
        program
            .why
            .iter()
            .any(|clause| clause.clause.contains("active"))
    );
    assert!(
        program
            .why
            .iter()
            .any(|clause| clause.clause.contains("walrus"))
    );
    let unicode = compile("messages where model:\"Żółw\" AND active:false", &opts(&[])).unwrap();
    assert!(
        unicode
            .why
            .iter()
            .any(|clause| clause.clause.contains("Żółw"))
    );
}

#[test]
fn expression_lowering_does_not_slice_formatted_text_at_original_positions() {
    let text = format!(
        "sessions where {}repo:east{} | fields input_tokens",
        "(".repeat(80),
        ")".repeat(80)
    );
    let (_directory, fixture) = fixture();
    assert_eq!(
        rows(&fixture, &text),
        [json!({"input_tokens":3}), json!({"input_tokens":1})]
    );
}
