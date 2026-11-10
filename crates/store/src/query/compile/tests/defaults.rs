use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

fn ids(fixture: &Fixture, options: &Options, text: &str) -> Vec<String> {
    let program = compile(text, options).unwrap_or_else(|error| panic!("{text}: {error}"));
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut ids: Vec<String> = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .into_iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}

#[test]
fn active_unknown_matches() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:paths",
            ..Default::default()
        })
        .unwrap();
    for (id, on_active_path) in [
        ("claude-code:on", Some(true)),
        ("claude-code:off", Some(false)),
        ("claude-code:unknown", None),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    on_active_path,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let options = opts(&[]);
    let ids = |text| ids(&fixture, &options, text);
    // The default keeps the active path and the messages whose path is unknown.
    assert_eq!(ids("messages"), ["claude-code:on", "claude-code:unknown"]);
    assert_eq!(ids("messages where active:true"), ["claude-code:on"]);
    assert_eq!(ids("messages where active:false"), ["claude-code:off"]);
    assert_eq!(
        ids("messages where active:unknown"),
        ["claude-code:unknown"]
    );
    // Unknown matches neither a condition nor its NOT.
    assert_eq!(
        ids("messages where active:* NOT active:true"),
        ["claude-code:off"]
    );
    assert_eq!(
        ids("messages where active:*"),
        ["claude-code:off", "claude-code:on", "claude-code:unknown"]
    );
}

#[test]
fn saved_query_hidden_condition_cannot_widen() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for id in ["claude-code:shown", "claude-code:hidden"] {
        fixture
            .session(SessionFacts {
                id,
                repo: Some("morrow"),
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .note("hidden", None, None, "session:claude-code:hidden")
        .unwrap();
    let options = opts(&[
        ("morrow", "sessions where repo:morrow"),
        ("hidden-morrow", "sessions where repo:morrow hidden:true"),
        ("over-morrow", "query:morrow"),
    ]);
    let ids = |text| ids(&fixture, &options, text);
    // Inline, a top-level condition on `hidden` lifts the default.
    assert_eq!(
        ids("sessions where repo:morrow hidden:true"),
        ["claude-code:hidden"]
    );
    // A saved relation bound its own defaults; the caller only narrows it,
    // as the source and as a top-level condition alike.
    assert_eq!(ids("query:morrow"), ["claude-code:shown"]);
    assert!(ids("query:morrow hidden:true").is_empty());
    assert_eq!(ids("query:morrow hidden:*"), ["claude-code:shown"]);
    assert!(ids("sessions where query:morrow hidden:true").is_empty());
    assert!(ids("query:over-morrow hidden:true").is_empty());
    // A saved relation that lifts a default itself keeps what it lifted.
    assert_eq!(ids("query:hidden-morrow"), ["claude-code:hidden"]);
    assert_eq!(
        ids("sessions where query:hidden-morrow"),
        ["claude-code:hidden"]
    );
    // Under OR it is membership in its own population; the outer defaults
    // still apply around it.
    assert_eq!(
        ids("sessions where query:morrow OR repo:none"),
        ["claude-code:shown"]
    );
    assert!(ids("sessions where query:hidden-morrow OR repo:none").is_empty());
}
