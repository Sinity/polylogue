use super::*;

#[test]
fn five_minute_clustering_uses_interval_bounds_and_absolute_gaps() {
    assert_eq!(clustered(&[]), 0);
    assert_eq!(clustered(&[0, 300_000, 600_000]), 600_000);
    assert_eq!(clustered(&[0, 360_000, 720_000]), 0);
    assert_eq!(clustered(&[200, 0, 100]), 200);
    assert_eq!(clustered(&[0, 120_000, 900_000, 1_020_000]), 240_000);
}

#[test]
fn topic_strips_known_prefixes_and_preserves_unicode() {
    assert_eq!(clean_topic("[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code. Fix the parser.",120).as_deref(),Some("Fix the parser."));
    assert_eq!(clean_topic("   ", 120), None);
    assert_eq!(clean_topic("ééé", 2).as_deref(), Some("éé"));
}

#[test]
fn profile_reads_prose_without_decoding_tool_output() {
    use crate::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let session = f
        .session(SessionFacts {
            id: "codex:output",
            ..Default::default()
        })
        .unwrap();
    let message = f
        .message(
            session,
            MessageFacts {
                id: "a",
                role: "assistant",
                written_by: "model",
                text: Some("A substantive reply"),
                at: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    f.tool_call(
        session,
        message,
        CallFacts {
            tool: "Bash",
            action: "shell",
            outcome: "ok",
            at: Some(0),
            output: Some("synthetic output"),
            ..Default::default()
        },
    )
    .unwrap();
    let mut conn =
        crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    // Reading this block would fail decoding; no profile fact depends on it.
    conn.execute("UPDATE block SET body=X'FF' WHERE kind='tool_result'", [])
        .unwrap();
    let tx = conn.transaction().unwrap();
    run(&tx, session).unwrap();
    let facts: String = tx
        .query_row(
            "SELECT facts FROM session_profile WHERE session_pk=?1",
            [session],
            |r| r.get(0),
        )
        .unwrap();
    let facts: Facts = serde_json::from_str(&facts).unwrap();
    assert_eq!(facts.tool_call_count, 1);
    assert_eq!(facts.message_count, 1);
}
