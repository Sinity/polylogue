use super::*;
use rusqlite::StatementStatus;

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
    let b = batch("input", 0, 100, rows, &["session"]);
    assert_eq!(write_rows(&tx, &b, &b.parse[0], 1, 1, true).unwrap(), 32);
    let reads = tx
        .prepare_cached("SELECT id, indexed FROM text WHERE hash = ?1")
        .unwrap()
        .get_status(StatementStatus::Run);
    println!("text identity lookups for 32 distinct blocks: {reads}");
    assert_eq!(
        reads, 32,
        "current projection repeated historical text identity lookup"
    );
}
