use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use polylogue_model::{Reference, user::UserWrite};

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
            "id session role written_by model at text input_tokens output_tokens cache_read_tokens cache_write_tokens cost_nusd duplicate_of on_active_path",
        ),
        (
            "v1_tool_calls",
            "session call_id at tool action command path outcome exit_code",
        ),
        ("v1_links", "parent child kind reason"),
        (
            "v1_annotations",
            "id root kind key value author via machine created_at valid_until about current withdrawn conflict",
        ),
    ] {
        let stmt = conn.prepare(&format!("SELECT * FROM {view}")).unwrap();
        assert_eq!(stmt.column_names().join(" "), columns, "{view}");
    }
}

#[test]
fn the_views_read_sessions_messages_and_annotations() {
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
    f.note("tag", Some("research"), None, "session:test:s1")
        .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let sessions = reader
        .query("SELECT id, title FROM v1_sessions", &[])
        .unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["id"], "test:s1");
    assert_eq!(sessions[0]["title"], "parser work");
    let messages = reader
        .query("SELECT session, text FROM v1_messages", &[])
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["session"], "test:s1");
    assert_eq!(messages[0]["text"], "fix the parser");
    let annotations = reader
        .query(
            "SELECT key, value, about, current, withdrawn FROM v1_annotations",
            &[],
        )
        .unwrap();
    assert_eq!(annotations[0]["about"], "session:test:s1");
    assert_eq!(annotations[0]["key"], "research");
    assert!(annotations[0]["value"].is_null());
    assert_eq!(
        (&annotations[0]["current"], &annotations[0]["withdrawn"]),
        (&1.into(), &0.into())
    );
}

/// An unknown token count or cost is NULL, never 0; known values pass through.
#[test]
fn the_views_keep_unknown_values_null_and_read_calls_and_links() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:s1",
            ..SessionFacts::default()
        })
        .unwrap();
    f.session(SessionFacts {
        id: "test:s2",
        ..SessionFacts::default()
    })
    .unwrap();
    let priced = f
        .message(
            s,
            MessageFacts {
                id: "m1",
                role: "assistant",
                written_by: "model",
                cost_nusd: Some(2500),
                ..MessageFacts::default()
            },
        )
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "m2",
            role: "user",
            written_by: "human",
            ..MessageFacts::default()
        },
    )
    .unwrap();
    f.tool_call(
        s,
        priced,
        CallFacts {
            tool: "Bash",
            action: "shell",
            command: Some("ls"),
            outcome: "error",
            exit_code: Some(2),
            ..CallFacts::default()
        },
    )
    .unwrap();
    f.link("test:s1", "test:s2", "fork", None).unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let messages = reader
        .query(
            "SELECT id, input_tokens, output_tokens, cost_nusd FROM v1_messages ORDER BY id",
            &[],
        )
        .unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["cost_nusd"], 2500);
    assert!(messages[0]["input_tokens"].is_null());
    assert!(messages[0]["output_tokens"].is_null());
    assert!(messages[1]["cost_nusd"].is_null());
    let calls = reader
        .query(
            "SELECT session, tool, command, outcome, exit_code FROM v1_tool_calls",
            &[],
        )
        .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["session"], "test:s1");
    assert_eq!(calls[0]["tool"], "Bash");
    assert_eq!(calls[0]["command"], "ls");
    assert_eq!(calls[0]["outcome"], "error");
    assert_eq!(calls[0]["exit_code"], 2);
    let links = reader
        .query("SELECT parent, child, kind, reason FROM v1_links", &[])
        .unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0]["parent"], "test:s1");
    assert_eq!(links[0]["child"], "test:s2");
    assert_eq!(links[0]["kind"], "fork");
    assert!(links[0]["reason"].is_null());
}

#[test]
fn a_withdrawn_annotation_shows_as_withdrawn_in_its_view() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    f.session(SessionFacts {
        id: "test:s1",
        ..SessionFacts::default()
    })
    .unwrap();
    let kept = f
        .note("tag", Some("keep"), None, "session:test:s1")
        .unwrap();
    let dropped = f
        .note("tag", Some("drop"), None, "session:test:s1")
        .unwrap();
    f.user(UserWrite::WithdrawAnnotations {
        targets: vec![Reference::parse(&format!("annotation:{dropped}@{dropped}")).unwrap()],
        reason: None,
    })
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query(
            "SELECT id, withdrawn, conflict FROM v1_annotations WHERE key IN ('keep','drop') ORDER BY key DESC",
            &[],
        )
        .unwrap();
    assert_eq!(rows[0]["id"], kept);
    assert_eq!(rows[0]["withdrawn"], 0);
    assert_eq!(rows[1]["id"], dropped);
    assert_eq!(rows[1]["withdrawn"], 1);
    assert_eq!(rows[1]["conflict"], 0);
}

/// A stub is a hook-named session with no transcript yet; the view lists
/// sessions only.
#[test]
fn the_sessions_view_leaves_out_stubs() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(&dir.path().join("archive")).unwrap();
    f.session(SessionFacts {
        id: "test:real",
        ..SessionFacts::default()
    })
    .unwrap();
    let stub = f
        .session(SessionFacts {
            id: "test:stub",
            ..SessionFacts::default()
        })
        .unwrap();
    crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute("UPDATE session SET stub=1 WHERE pk=?1", [stub])
        .unwrap();
    let pool = f.reads().unwrap();
    let sessions = pool
        .reader()
        .unwrap()
        .query("SELECT id FROM v1_sessions", &[])
        .unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0]["id"], "test:real");
}
