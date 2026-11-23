//! Several families in one single-value slot (each written first where the
//! other was not yet seen, then merged) are a conflict on every reader, and
//! one write or revision naming every head resolves them.
use super::*;
use crate::fixture::{Fixture, SessionFacts};
use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
use polylogue_model::user::UserWrite;

fn count(fixture: &Fixture, text: &str) -> usize {
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader
        .query_rows(&compile(text, &opts(&[])).unwrap())
        .unwrap()
        .len()
}

#[test]
fn two_titles_written_apart_show_conflict_until_one_write_names_every_head() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "claude-code:titled",
            kind: "main",
            started: Some(1000),
            ..Default::default()
        })
        .unwrap();
    // Two first titles for the same session, as two archives would each have
    // written them, both through the apply path.
    for title in ["From the laptop", "From the desktop"] {
        fixture
            .note("title", None, Some(title), "session:claude-code:titled")
            .unwrap();
    }
    assert_eq!(
        count(&fixture, "annotations where kind:title conflict:true"),
        2
    );
    assert_eq!(count(&fixture, "sessions where has:conflict"), 1);
    // The next title written from a command revises every head of both.
    fixture
        .user(UserWrite::Assert {
            assertions: vec![NewAssertion {
                kind: AssertionKind::Title,
                key: None,
                value: Some(serde_json::json!("Settled")),
                valid_until: None,
                links: vec![NewAssertionLink {
                    role: LinkRole::About,
                    reference: polylogue_model::Reference::parse("session:claude-code:titled")
                        .unwrap(),
                }],
            }],
        })
        .unwrap();
    assert_eq!(
        count(&fixture, "annotations where kind:title conflict:true"),
        0
    );
    assert_eq!(count(&fixture, "annotations where kind:title"), 1);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = compile("annotations where kind:title | fields value", &opts(&[])).unwrap();
    let mut rows = reader.query_rows(&program).unwrap();
    for row in &mut rows {
        crate::query::execute::present(&program, row).unwrap();
    }
    assert_eq!(rows, [serde_json::json!({"value": "Settled"})]);
    assert_eq!(count(&fixture, "sessions where has:conflict"), 0);
}
