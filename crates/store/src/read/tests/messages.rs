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
}

#[test]
fn message_visits_include_the_same_logical_history() {
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
        .link("claude-code:parent", "claude-code:child", "fork", None)
        .unwrap();
    crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE link SET branch_point='claude-code:p1'", [])
        .unwrap();
    let pool = fixture.reads().unwrap();
    let id = SessionId::parse("claude-code:child").unwrap();
    let reader = pool.reader().unwrap();
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
    // A parent's message after the branch point is not inherited.
    let reader = pool.reader().unwrap();
    let mut messages = Vec::new();
    reader
        .visit_messages(&id, true, |m| {
            messages.push(m.text());
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(messages, ["inherited", "own"]);
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
    // The compaction event is one entry of the session's item index, after
    // the message it follows.
    let items = reader
        .item_index(&SessionId::parse("claude-code:outcome").unwrap())
        .unwrap()
        .unwrap();
    let events: Vec<_> = items
        .entries
        .iter()
        .filter_map(|entry| entry.event_kind.as_deref())
        .collect();
    assert_eq!(events, ["compaction"]);
}
