use super::*;
use crate::archive::open_write;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

#[test]
fn item_commit_uses_prepared_content_without_reading_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let mut conn = open_write(&directory.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&conn).unwrap();
    let tx = conn.transaction().unwrap();
    let mut message = text_message(
        "message",
        "session",
        Role::Assistant,
        &"long output\n".repeat(10000),
        Some("end_turn"),
    );
    message.blocks.push(Block::ToolCall {
        call_id: Some("call".into()),
        tool: Some("Bash".into()),
        action: polylogue_model::ToolAction::Shell,
        input: serde_json::json!({"command":"printf ready"}),
        command: Some("printf ready".into()),
        path: None,
        written_text: None,
        run_by_script: false,
    });
    let mut b = batch("input", Route::Live, 0, vec![row(0, message)], &["session"]);
    resolve::prepare_index(&tx, &mut b, 1, &resolve::IndexUpkeep::PerTransaction).unwrap();
    let texts = crate::text_rows::TextIds::new(
        &tx,
        false,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    write_rows(
        &tx,
        &b,
        &b.parse[0],
        rows::ReadingWrite {
            texts: &texts,
            resolution: b.parse[0].resolution.as_ref().unwrap(),
            counter: 1,
            upkeep: &resolve::IndexUpkeep::PerTransaction,
            source_commit: Some(&fixture_commit()),
            changes: &changes,
            retraction: false,
            replacement: false,
        },
    )
    .unwrap();
    tx.flush_prepared_statement_cache();
    tx.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "message_version_block" | "text" | "message_version_body",
            column_name: "body",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    crate::items::refresh(&tx, &changes).unwrap();
    let session = tx
        .query_row("SELECT pk FROM session", [], |row| row.get::<_, i64>(0))
        .unwrap();
    assert!(
        crate::items::computed_entries(&tx, session).is_err(),
        "the old row reconstruction must hit the denied payload columns"
    );
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    let expected = crate::items::computed_entries(&tx, session).unwrap();
    let stored = crate::items::stored_entries(&tx, session, 0).unwrap();
    assert_eq!(stored, expected);
    assert_eq!(stored[1].body_bytes, 120000);
    assert_eq!(stored[2].call_id.as_deref(), Some("call"));
    // A placement-only update uses the existing metadata, with no body decode.
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let message = tx
        .query_row("SELECT pk FROM message", [], |row| row.get::<_, i64>(0))
        .unwrap();
    changes.message_metadata(&tx, message).unwrap();
    tx.execute("UPDATE message SET on_active_path=0", [])
        .unwrap();
    tx.flush_prepared_statement_cache();
    tx.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "message_version_block" | "text" | "message_version_body",
            column_name: "body",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    crate::items::refresh(&tx, &changes).unwrap();
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    let expected = crate::items::computed_entries(&tx, session).unwrap();
    assert!(
        expected
            .iter()
            .all(|entry| entry.active_path == Some(false))
    );
    let stored = crate::items::stored_entries(&tx, session, 0).unwrap();
    assert_eq!(stored, expected);
}
