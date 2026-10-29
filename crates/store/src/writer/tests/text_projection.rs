use super::*;
use rusqlite::hooks::{AuthAction, AuthContext, Authorization};

#[test]
fn current_blocks_reuse_stored_text_identities() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = open_write(&dir.path().join("index.db"), 1).unwrap();
    crate::schema::create_index(&db).unwrap();
    let tx = db.transaction().unwrap();
    let rows = (0..32)
        .map(|n| {
            row(
                n,
                text_message(
                    &format!("message-{n}"),
                    "session",
                    Role::User,
                    &format!("Synthetic unique body {n}"),
                    None,
                ),
            )
        })
        .collect();
    let mut b = batch("input", 0, 0, rows, &["session"]);
    resolve::prepare_index(&tx, &mut b, 1, false).unwrap();
    tx.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "text",
            column_name: "hash",
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    assert_eq!(write_rows(&tx, &b, &b.parse[0], 1, 1, None).unwrap(), 32);
    tx.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert_eq!(
        tx.query_row("SELECT count(DISTINCT text_id) FROM block", [], |row| row
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        32
    );
}
