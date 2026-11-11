//! A `| where` stage straight after the source joins its selection, except
//! where the conjunction would mean something else (docs/query.md, Stages).
use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use serde_json::Value;

/// Two sessions: `older` in east with a user prompt answered 2 s later, and
/// `newer` in west with a user prompt answered 20 s later.
fn fixture() -> (tempfile::TempDir, Fixture) {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(directory.path()).unwrap();
    for (id, repo, start, answer) in [
        ("older", "east", 1_000, 3_000),
        ("newer", "west", 50_000, 70_000),
    ] {
        let session = fixture
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                repo: Some(repo),
                ..Default::default()
            })
            .unwrap();
        for (suffix, role, written_by, at) in [
            ("prompt", "user", "human", start),
            ("answer", "assistant", "model", answer),
        ] {
            fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("claude-code:{id}-{suffix}"),
                        role,
                        written_by,
                        at: Some(at),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    fixture.derive().unwrap();
    (directory, fixture)
}

fn ids(fixture: &Fixture, text: &str) -> Vec<String> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .into_iter()
        .map(|mut row| {
            crate::query::execute::present(&program, &mut row).unwrap();
            row
        })
        .map(|row| match &row["id"] {
            Value::String(id) => id.clone(),
            other => panic!("{text}: id {other}"),
        })
        .collect()
}

/// `last` is the newest session meeting the conditions beside it, so a later
/// stage narrows the newest session instead of choosing among its own rows.
#[test]
fn a_stage_after_last_narrows_the_newest_session() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        ids(&fixture, "sessions where last repo:east"),
        ["claude-code:older"]
    );
    assert!(ids(&fixture, "sessions where last | where repo:east").is_empty());
    assert_eq!(
        ids(&fixture, "sessions where last | where repo:west"),
        ["claude-code:newer"]
    );
}

/// An order condition in a stage still filters: only the prompt answered
/// within 5 s is kept, never every user message.
#[test]
fn an_order_condition_in_a_stage_still_selects() {
    let (_directory, fixture) = fixture();
    assert_eq!(
        ids(
            &fixture,
            "messages where role:user | where before(messages where role:assistant) within 5s | fields id"
        ),
        ["claude-code:older-prompt"]
    );
}

/// `| matches` reads its pattern from the stage that names it.
#[test]
fn matches_read_the_pattern_of_a_where_stage() {
    let (_directory, fixture) = fixture();
    let program = compile(
        "sessions where repo:east | where exists messages(role:assistant) | matches",
        &opts(&[]),
    )
    .unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let rows = reader.query_rows(&program).unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
}
