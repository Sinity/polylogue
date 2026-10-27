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
    let (calls, messages): (u64, u64) = tx
        .query_row(
            "SELECT tool_call_count,message_count FROM session_profile WHERE session_pk=?1",
            [session],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(messages, 1);
}
