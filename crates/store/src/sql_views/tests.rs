use crate::fixture::{Fixture, MessageFacts, SessionFacts};

/// A view's columns are part of its version: changing them means a new
/// version beside it.
#[test]
fn each_view_keeps_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let conn = reader.conn().unwrap();
    for (view, columns) in [
        (
            "v1_sessions",
            "id origin kind title cwd repo branch harness started_at last_activity_at",
        ),
        (
            "v1_messages",
            "id session role written_by model at text input_tokens output_tokens cache_read_tokens cache_write_tokens cost_usd duplicate_of on_active_path",
        ),
        (
            "v1_tool_calls",
            "session call_id at tool action command path outcome exit_code",
        ),
        ("v1_links", "parent child kind how"),
        (
            "v1_notes",
            "id first_note kind key value author created_at targets current withdrawn",
        ),
    ] {
        let stmt = conn.prepare(&format!("SELECT * FROM {view}")).unwrap();
        assert_eq!(stmt.column_names().join(" "), columns, "{view}");
    }
}

#[test]
fn the_views_read_sessions_messages_and_notes() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:s1",
            title: Some("parser work"),
            ..SessionFacts::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "m1",
            role: "user",
            written_by: "human",
            text: Some("fix the parser"),
            ..MessageFacts::default()
        },
    )
    .unwrap();
    f.note("tag", None, Some("research"), "session:test:s1")
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let sessions = reader
        .query("SELECT id, title FROM v1_sessions", &[])
        .unwrap();
    assert_eq!(sessions[0]["id"], "test:s1");
    assert_eq!(sessions[0]["title"], "parser work");
    let messages = reader
        .query("SELECT session, text FROM v1_messages", &[])
        .unwrap();
    assert_eq!(messages[0]["session"], "test:s1");
    assert_eq!(messages[0]["text"], "fix the parser");
    let notes = reader
        .query(
            "SELECT value, targets, current, withdrawn FROM v1_notes",
            &[],
        )
        .unwrap();
    assert_eq!(notes[0]["targets"], "session:test:s1");
    assert_eq!(
        (&notes[0]["current"], &notes[0]["withdrawn"]),
        (&1.into(), &0.into())
    );
}
