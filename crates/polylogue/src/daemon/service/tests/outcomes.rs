//! Each answer's outcome is decided where it is built: a query that matched
//! nothing is empty (exit 1); an incomplete query, a read over an index that
//! is not current, and an import whose user state awaits confirmation are
//! degraded (exit 5) and say why in their notes.
use super::*;
use crate::daemon::query::{Outcome, QueryAnswer};
use crate::test_answers;
use polylogue_model::freshness::{IndexFreshness, ReadCompleteness, ReadFreshness};

fn query(outcome: Outcome, notes: &[&str]) -> QueryAnswer {
    QueryAnswer {
        logical_sessions: Vec::new(),
        value: crate::protocol::query::Answer::Rows(Box::new(crate::protocol::query::Rows {
            read_position: crate::protocol::query::Position {
                freshness: None,
                index_instance: "synthetic".into(),
                index: 1,
                archive: 1,
                cursor: "synthetic:1:1".into(),
            },
            unit: "sessions".into(),
            shape: polylogue_query::Shape::Rows,
            columns: vec![],
            rows: vec![],
            references: None,
            count: Some(0),
            next: None,
            ranked: false,
            notes: vec![],
            incomplete: vec![],
            resolved: crate::protocol::query::Resolved {
                context: synthetic_context().run_context(),
                times: vec![],
                run: None,
                query: None,
            },
            definitions: vec![],
            unknown: Default::default(),
            why: None,
        })),
        outcome,
        notes: notes.iter().map(|note| (*note).to_owned()).collect(),
    }
}

fn synthetic_context() -> polylogue_reading::operation::Context {
    polylogue_reading::operation::Context::standalone(
        polylogue_reading::operation::Invocation::new(
            polylogue_model::Timestamp::parse_rfc3339("2026-03-01T12:00:00Z").unwrap(),
            "synthetic-outcomes".into(),
            polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
        ),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    )
}

fn current() -> ReadFreshness {
    ReadFreshness {
        local: ReadCompleteness::empty(),
        sinex: IndexFreshness::Standalone,
    }
}
fn behind() -> ReadFreshness {
    ReadFreshness {
        local: ReadCompleteness::empty(),
        sinex: IndexFreshness::primary(Some(5), Some(8), Some(0.5), Some(true), None),
    }
}

#[test]
fn a_query_that_matched_nothing_is_empty_with_exit_one() {
    let answer = query_outcome(query(Outcome::Empty, &[]), &current());
    assert_eq!(answer.exit_code(), 1);
    assert_eq!(answer.http_status(), Some(200));
    test_answers::empty(answer);
}

#[test]
fn a_query_with_rows_or_a_measure_over_a_current_index_is_ok() {
    for outcome in [Outcome::Rows, Outcome::Measure] {
        let answer = query_outcome(query(outcome, &[]), &current());
        assert_eq!(answer.exit_code(), 0, "{outcome:?}");
        assert!(
            crate::protocol::notes(&answer)
                .collect::<Vec<_>>()
                .is_empty(),
            "{outcome:?}"
        );
        test_answers::ok(answer);
    }
}

#[test]
fn an_incomplete_query_is_degraded_and_names_the_missing_part() {
    let answer = query_outcome(
        query(Outcome::Incomplete, &["semantic vectors unavailable"]),
        &current(),
    );
    assert_eq!(answer.exit_code(), 5);
    assert_eq!(answer.http_status(), Some(200));
    let (_, notes) = test_answers::degraded(answer);
    assert_eq!(notes, ["semantic vectors unavailable"]);
}

#[test]
fn a_query_over_an_index_that_is_behind_is_degraded_even_with_rows_or_none() {
    for outcome in [Outcome::Rows, Outcome::Empty, Outcome::Measure] {
        let answer = query_outcome(query(outcome, &[]), &behind());
        assert_eq!(answer.exit_code(), 5, "{outcome:?}");
        let (mut value, notes) = test_answers::degraded(answer);
        assert_eq!(notes, behind().notes(), "{outcome:?}");
        assert_eq!(
            value.read_position_mut().freshness.as_ref(),
            Some(&behind())
        );
    }
}

#[test]
fn an_import_with_pending_user_state_is_degraded_with_its_note() {
    let mut report = polylogue_engine::ImportReport::default();
    report.user_state.pending = 1;
    let answer = import_answer(report.clone(), false);
    assert_eq!(answer.exit_code(), 5);
    assert_eq!(answer.http_status(), Some(200));
    let (value, notes) = test_answers::degraded(answer);
    assert_eq!(value.user_state.pending, 1);
    assert_eq!(notes, [crate::protocol::AWAITING_CONFIRMATION]);

    report.user_state = polylogue_engine::UserStateReport {
        pending: 0,
        applied: 1,
        present: 2,
    };
    let answer = import_answer(report.clone(), false);
    assert_eq!(answer.exit_code(), 0);
    assert!(
        crate::protocol::notes(&answer)
            .collect::<Vec<_>>()
            .is_empty()
    );
    test_answers::ok(answer);
}

#[test]
fn an_import_answered_during_a_build_says_the_file_is_stored_and_not_yet_readable() {
    let build = polylogue_store::archive::index_build::IndexBuild {
        reason: polylogue_store::index::file::RebuildReason::NoIndex,
        chains: 3,
        chains_read: 1,
        bytes: 300,
        bytes_read: 100,
    };
    let report = polylogue_engine::ImportReport {
        readable: polylogue_engine::Readable::AfterBuild { build },
        imported: vec!["/home/op/export.json".into()],
        ..Default::default()
    };
    let answer = import_answer(report, false);
    assert_eq!(answer.exit_code(), 5);
    let (report, notes) = test_answers::degraded(answer);
    assert_eq!(
        notes,
        [
            "stored, not yet readable: the index is being built (the archive had no index): 100 of 300 bytes parsed; 1 of 3 input chains read"
        ]
    );
    assert_eq!(
        crate::cli::print::import(
            &crate::protocol::import::Answer::Files(Box::new(report)),
            command_surface::TextStyle::default(),
        ),
        "stored, not yet readable: /home/op/export.json\n",
    );
}

#[test]
fn answer_reports_pending_search_text_and_parse_lag_with_counts_and_origins() {
    use polylogue_model::freshness::{CompletenessDimension, CompletenessReason, CountedReason};
    let mut freshness = current();
    freshness.local.search = CompletenessDimension::Incomplete(vec![CountedReason::new(
        CompletenessReason::SearchPending,
    )]);
    freshness.local.parsing = CompletenessDimension::Incomplete(vec![
        CountedReason::inputs(CompletenessReason::ParserBehind, "codex", 1),
        CountedReason::inputs(CompletenessReason::MissingCheckpoint, "claude-code", 12),
    ]);
    let answer = query_outcome(query(Outcome::Rows, &[]), &freshness);
    assert_eq!(answer.exit_code(), 5);
    let (mut value, notes) = test_answers::degraded(answer);
    assert_eq!(
        notes,
        [
            "1 codex input has bytes not parsed yet; status counts them, and no query lists stored inputs yet",
            "12 claude-code inputs have not been parsed yet; status counts them, and no query lists stored inputs yet",
            "search text is pending",
        ]
    );
    assert_eq!(
        value.read_position_mut().freshness.as_ref(),
        Some(&freshness)
    );
    // --json carries each count beside its reason's name.
    assert_eq!(
        serde_json::to_value(&freshness.local.parsing).unwrap(),
        serde_json::json!({"state": "incomplete", "reasons": [
            {"reason": "parser_behind", "origin": "codex", "count": 1, "waiting_ends": true},
            {"reason": "missing_checkpoint", "origin": "claude-code", "count": 12, "waiting_ends": true},
        ]})
    );
    let answer = crate::protocol::read(value.clone(), &freshness);
    let (same_value, same_notes) = test_answers::degraded(answer);
    assert_eq!(
        serde_json::to_value(same_value).unwrap(),
        serde_json::to_value(value).unwrap()
    );
    assert_eq!(same_notes, notes);
}

/// docs/interfaces.md, Exit codes: a failed or refused parse stands until
/// its bytes or its parser change, so waiting does not end it and it does
/// not make a reply degraded; with no rows the reply is empty.
#[test]
fn a_failed_parse_leaves_a_query_ok_or_empty() {
    use polylogue_model::freshness::{CompletenessReason, CountedReason};
    let mut freshness = current();
    freshness.local.failed_parses = vec![
        CountedReason::inputs(CompletenessReason::ParseFailed, "codex", 2),
        CountedReason::inputs(CompletenessReason::Refused, "chatgpt", 1),
    ];
    assert!(freshness.up_to_date());
    assert!(!freshness.catching_up());
    test_answers::ok(query_outcome(query(Outcome::Rows, &[]), &freshness));
    test_answers::empty(query_outcome(query(Outcome::Empty, &[]), &freshness));
}
