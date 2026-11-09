//! One displayed title everywhere: the authored title, else the recorded
//! one, else the derived one; a conflict shows every current value.
use super::*;
use crate::fixture::{Fixture, SessionFacts};
use polylogue_model::SessionId;

fn fixture() -> (tempfile::TempDir, Fixture) {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "claude-code:titled",
            kind: "main",
            title: Some("Recorded title"),
            started: Some(1000),
            ..Default::default()
        })
        .unwrap();
    (dir, fixture)
}

/// The title as the list, the `title` field, a single read and the SQL view
/// each give it.
fn shown_titles(fixture: &Fixture) -> [String; 4] {
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = compile("sessions", &opts(&[])).unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    crate::query::execute::present(&program, &mut rows[0]).unwrap();
    let listed = rows[0]["title"].as_str().unwrap().to_owned();
    let program = compile("sessions | fields id, title", &opts(&[])).unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    crate::query::execute::present(&program, &mut rows[0]).unwrap();
    let field = rows[0]["title"].as_str().unwrap().to_owned();
    let read = reader
        .display_title(&SessionId::parse("claude-code:titled").unwrap())
        .unwrap()
        .unwrap();
    let view: String = reader
        .conn()
        .unwrap()
        .query_row("SELECT title FROM v1_sessions", [], |row| row.get(0))
        .unwrap();
    [listed, field, read, view]
}

fn matches(fixture: &Fixture, text: &str) -> usize {
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .query_rows(&compile(text, &opts(&[])).unwrap())
        .unwrap()
        .len()
}

#[test]
fn authored_title_wins_in_list_read_and_title_field() {
    let (_dir, mut fixture) = fixture();
    assert_eq!(
        shown_titles(&fixture),
        ["Recorded title"; 4].map(String::from)
    );
    fixture
        .note(
            "title",
            None,
            Some("Authored title"),
            "session:claude-code:titled",
        )
        .unwrap();
    assert_eq!(
        shown_titles(&fixture),
        ["Authored title"; 4].map(String::from)
    );
    assert_eq!(matches(&fixture, "sessions where title:authored"), 1);
    assert_eq!(matches(&fixture, "sessions where title:recorded"), 0);
}

#[test]
fn conflicting_authored_titles_are_both_shown() {
    let (_dir, mut fixture) = fixture();
    // Two first titles for one session, as two archives each write one.
    for title in ["Second view", "First view"] {
        fixture
            .note("title", None, Some(title), "session:claude-code:titled")
            .unwrap();
    }
    assert_eq!(
        shown_titles(&fixture),
        ["First view | Second view"; 4].map(String::from)
    );
    assert_eq!(matches(&fixture, "sessions where title:second"), 1);
}
