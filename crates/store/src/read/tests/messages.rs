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
    let own = reader.message_fingerprint(&id, false).unwrap();
    let with_parents = reader.message_fingerprint(&id, true).unwrap();
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
    assert_eq!(reader.message_fingerprint(&id, false).unwrap(), own);
    assert_eq!(reader.message_fingerprint(&id, true).unwrap(), with_parents);
    drop(reader);
    crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE link SET branch_point='claude-code:p2'", [])
        .unwrap();
    let reader = pool.reader().unwrap();
    assert_ne!(reader.message_fingerprint(&id, true).unwrap(), with_parents);
}

#[test]
fn visible_tool_outcomes_use_original_source_block_indices() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:outcome",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:outcome-0",
                role: "assistant",
                text: Some("before tool"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .tool_call(
            session,
            message,
            crate::fixture::CallFacts {
                tool: "Bash",
                command: Some("synthetic-check"),
                outcome: "error",
                output: Some("synthetic failure"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture.event(session, "compaction", None).unwrap();
    let conn = rusqlite::Connection::open(fixture.archive.index_db()).unwrap();
    conn.execute(
        "UPDATE event SET after_message_pk=?1 WHERE session_pk=?2",
        params![message, session],
    )
    .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let id = MessageId::parse("claude-code:outcome-0").unwrap();
    assert_eq!(
        reader.tool_outcomes(&id).unwrap(),
        vec![(1, polylogue_model::CallOutcome::Error)]
    );
    let selected = reader
        .select_messages(&SessionId::parse("claude-code:outcome").unwrap(), false)
        .unwrap();
    let outline = selected.outline(None, 10).unwrap();
    assert_eq!(outline.len(), 1);
    assert_eq!(outline[0].kinds, vec!["tool failure", "compaction"]);
    assert_eq!(outline[0].events.len(), 1);
}
