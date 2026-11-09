//! Snippets look for the query's positive top-level terms, each in its own
//! index, and mark them by character.
use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

fn fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:snippets",
            kind: "main",
            started: Some(1000),
            ..Default::default()
        })
        .unwrap();
    for (n, text) in [
        "alpha omega",
        "alpha beta",
        "we refactored the parser",
        "zażółć gęślą jaźń quartz",
    ]
    .into_iter()
    .enumerate()
    {
        fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("claude-code:snippets/{n}"),
                    role: "user",
                    written_by: "human",
                    prompt_kind: Some("prose"),
                    at: Some(2000 + n as i64),
                    text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    (dir, fixture)
}

/// Each row's snippet text and its marked words.
fn snippets(fixture: &Fixture, text: &str) -> Vec<(String, Vec<String>)> {
    let program = compile(text, &opts(&[])).unwrap_or_else(|error| panic!("{text}: {error}"));
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut rows = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    crate::query::execute::snippets(&reader, &program, &mut rows).unwrap();
    rows.iter()
        .map(|row| {
            let snippet = &row["snippet"];
            let shown = snippet["text"]
                .as_str()
                .unwrap_or_else(|| panic!("{text}: no snippet in {row}"))
                .to_owned();
            let marks = snippet["marks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|mark| {
                    let (start, end) = (
                        mark[0].as_u64().unwrap() as usize,
                        mark[1].as_u64().unwrap() as usize,
                    );
                    shown.chars().skip(start).take(end - start).collect()
                })
                .collect();
            (shown, marks)
        })
        .collect()
}

#[test]
fn snippet_ignores_negated_terms() {
    let (_dir, fixture) = fixture();
    let got = snippets(&fixture, "messages where alpha NOT omega");
    assert_eq!(got, [("alpha beta".to_owned(), vec!["alpha".to_owned()])]);
}

#[test]
fn snippet_for_or_branch_marks_the_matching_branch() {
    let (_dir, fixture) = fixture();
    let got = snippets(&fixture, "messages where (beta OR zeta)");
    assert_eq!(got, [("alpha beta".to_owned(), vec!["beta".to_owned()])]);
}

#[test]
fn prefix_term_is_highlighted() {
    let (_dir, fixture) = fixture();
    let got = snippets(&fixture, "messages where refactor*");
    assert_eq!(
        got,
        [(
            "we refactored the parser".to_owned(),
            vec!["refactored".to_owned()]
        )]
    );
}

#[test]
fn highlight_offsets_are_characters() {
    let (_dir, fixture) = fixture();
    let program = compile("messages where quartz", &opts(&[])).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    crate::query::execute::snippets(&reader, &program, &mut rows).unwrap();
    let snippet = &rows[0]["snippet"];
    assert_eq!(snippet["text"], "zażółć gęślą jaźń quartz");
    // "zażółć gęślą jaźń " is 18 characters and 26 bytes.
    assert_eq!(snippet["marks"], serde_json::json!([[18, 24]]));
}
