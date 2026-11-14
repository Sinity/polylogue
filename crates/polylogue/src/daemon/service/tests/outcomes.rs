//! Each answer's outcome is decided where it is built: a query that matched
//! nothing is empty (exit 1); an incomplete query, a read over an index that
//! is not current, and an import whose user state awaits confirmation are
//! degraded (exit 5) and say why in their notes.
use super::*;
use crate::daemon::query::{Outcome, QueryAnswer};
use crate::test_answers;
use polylogue_model::currency::{IndexCurrency, ReadCompleteness, ReadFreshness};

fn query(outcome: Outcome, notes: &[&str]) -> QueryAnswer {
    QueryAnswer {
        value: crate::protocol::query::Answer::Rows(Box::new(crate::protocol::query::Rows {
            read_at: crate::protocol::query::Position {
                freshness: None,
                generation: "synthetic".into(),
                index: 1,
                archive: 1,
                cursor: "synthetic:1:1".into(),
            },
            unit: "sessions".into(),
            shape: polylogue_query::Shape::Rows,
            columns: vec![],
            rows: vec![],
            count: Some(0),
            next: None,
            ranked: false,
            notes: vec![],
            incomplete: vec![],
            resolved: crate::protocol::query::Resolved {
                context: synthetic_context().provenance(),
                times: vec![],
            },
            definitions: vec![],
            unknown: Default::default(),
            why: None,
        })),
        outcome,
        notes: notes.iter().map(|note| (*note).to_owned()).collect(),
    }
}

fn synthetic_context() -> crate::operation::Context {
    crate::operation::Context::standalone(
        crate::operation::Invocation::new(
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
        owner: IndexCurrency::Standalone,
    }
}
fn behind() -> ReadFreshness {
    ReadFreshness {
        local: ReadCompleteness::empty(),
        owner: IndexCurrency::primary(Some(5), Some(8), Some(0.5), Some(true), None),
    }
}

#[test]
fn a_query_that_matched_nothing_is_empty_with_exit_one() {
    let answer = query_outcome(query(Outcome::Empty, &[]), &current());
    assert_eq!(answer.exit_code(), 1);
    assert_eq!(answer.http_status(), Some(200));
    let value = test_answers::empty(answer);
    assert!(matches!(value,crate::protocol::query::Answer::Rows(rows) if rows.count==Some(0)));
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
        assert_eq!(value.read_at_mut().freshness.as_ref(), Some(&behind()));
    }
}

#[test]
fn an_import_with_pending_user_state_is_degraded_with_its_note() {
    let mut report = polylogue_engine::ImportReport::default();
    report.user_state.pending = 1;
    let answer = import_answer(report.clone());
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
    let answer = import_answer(report);
    assert_eq!(answer.exit_code(), 0);
    assert!(
        crate::protocol::notes(&answer)
            .collect::<Vec<_>>()
            .is_empty()
    );
    test_answers::ok(answer);
}

#[test]
fn answer_reports_pending_search_text_and_parse_lag() {
    use polylogue_model::currency::{CompletenessDimension, CompletenessReason};
    let mut freshness = current();
    freshness.local.search =
        CompletenessDimension::Incomplete(vec![CompletenessReason::SearchPending]);
    freshness.local.parsing =
        CompletenessDimension::Incomplete(vec![CompletenessReason::ParserBehind]);
    let answer = query_outcome(query(Outcome::Rows, &[]), &freshness);
    assert_eq!(answer.exit_code(), 5);
    let (mut value, notes) = test_answers::degraded(answer);
    assert_eq!(
        notes,
        [
            "parsing is behind the stored bytes",
            "search text is pending"
        ]
    );
    assert_eq!(value.read_at_mut().freshness.as_ref(), Some(&freshness));
    let answer = crate::protocol::read(value.clone(), &freshness);
    let (same_value, same_notes) = test_answers::degraded(answer);
    assert_eq!(
        serde_json::to_value(same_value).unwrap(),
        serde_json::to_value(value).unwrap()
    );
    assert_eq!(same_notes, notes);
}
