use super::*;
use crate::fixture::{MessageFacts, SessionFacts};
use std::ops::ControlFlow;

#[test]
fn breaking_message_visit_does_not_decode_later_bodies() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:bounded",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:first",
                role: "user",
                written_by: "human",
                text: Some("first"),
                ..Default::default()
            },
        )
        .unwrap();
    let later = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:later",
                role: "user",
                written_by: "human",
                text: Some("later"),
                ..Default::default()
            },
        )
        .unwrap();
    let conn =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute("UPDATE block SET body=X'00' WHERE message_pk=?1", [later])
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let id = SessionId::parse("claude-code:bounded").unwrap();
    let mut visited = 0;
    reader
        .visit_messages(&id, false, |message| {
            visited += 1;
            assert_eq!(message.id.to_string(), "claude-code:first");
            ControlFlow::Break(())
        })
        .unwrap();
    assert_eq!(visited, 1);
    assert!(
        reader.messages(&id, false).is_err(),
        "the reference case must decode the corrupt later record"
    );
    assert!(
        reader.message_fingerprint(&id, false).is_ok(),
        "fingerprinting does not decode packed bodies"
    );
}

#[test]
fn message_visits_and_fingerprints_include_the_same_logical_history() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let parent = fixture
        .session(SessionFacts {
            id: "claude-code:parent",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let child = fixture
        .session(SessionFacts {
            id: "claude-code:child",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            parent,
            MessageFacts {
                id: "claude-code:p1",
                role: "user",
                written_by: "human",
                text: Some("inherited"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .message(
            child,
            MessageFacts {
                id: "claude-code:c1",
                role: "user",
                written_by: "human",
                text: Some("own"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .link(
            "claude-code:parent",
            "claude-code:child",
            "continuation",
            "fork",
        )
        .unwrap();
    crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE link SET branch_point='claude-code:p1'", [])
        .unwrap();
    let pool = fixture.reads().unwrap();
    let id = SessionId::parse("claude-code:child").unwrap();
    let reader = pool.reader().unwrap();
    let physical = reader.message_fingerprint(&id, false).unwrap();
    let logical = reader.message_fingerprint(&id, true).unwrap();
    let mut messages = Vec::new();
    reader
        .visit_messages(&id, true, |m| {
            messages.push(m.text());
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(messages, ["inherited", "own"]);
    drop(reader);
    fixture
        .message(
            parent,
            MessageFacts {
                id: "claude-code:p2",
                role: "assistant",
                written_by: "model",
                text: Some("new inherited"),
                ..Default::default()
            },
        )
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(reader.message_fingerprint(&id, false).unwrap(), physical);
    assert_eq!(reader.message_fingerprint(&id, true).unwrap(), logical);
    drop(reader);
    crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE link SET branch_point='claude-code:p2'", [])
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_ne!(reader.message_fingerprint(&id, true).unwrap(), logical);
}
