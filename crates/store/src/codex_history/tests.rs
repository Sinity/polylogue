use super::*;
use std::io::Read;

#[test]
fn schema_check_refuses_missing_or_newer_migrations_and_changed_columns() {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch(include_str!("../../tests/fixtures/codex-history-v0162.sql"))
        .unwrap();
    let path = Path::new("synthetic-history");
    assert!(
        validate(&c, path)
            .unwrap_err()
            .to_string()
            .contains("no_parser_yet")
    );
    c.execute_batch(
        "CREATE TABLE _sqlx_migrations(version INTEGER PRIMARY KEY,success INTEGER NOT NULL)",
    )
    .unwrap();
    for n in 1..=8 {
        c.execute("INSERT INTO _sqlx_migrations VALUES(?1,1)", [n])
            .unwrap();
    }
    validate(&c, path).unwrap();
    c.execute("INSERT INTO _sqlx_migrations VALUES(9,0)", [])
        .unwrap();
    assert!(
        validate(&c, path)
            .unwrap_err()
            .to_string()
            .contains("no_parser_yet")
    );
    c.execute("DELETE FROM _sqlx_migrations WHERE version=9", [])
        .unwrap();
    c.execute("ALTER TABLE thread_items ADD COLUMN unexpected TEXT", [])
        .unwrap();
    assert!(
        validate(&c, path)
            .unwrap_err()
            .to_string()
            .contains("no_parser_yet")
    );
}

#[test]
fn physical_snapshot_includes_committed_wal_and_excludes_an_open_write() {
    let root = tempfile::tempdir().unwrap();
    let source_path = root.path().join("source.sqlite");
    let source = Connection::open(&source_path).unwrap();
    source.execute_batch("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE evidence(value INTEGER); INSERT INTO evidence VALUES(1); BEGIN IMMEDIATE; UPDATE evidence SET value=2").unwrap();
    let archive = root.path().join("archive");
    let writer = crate::Writer::open(&archive).unwrap();
    let captured = writer
        .blob_capture()
        .unwrap()
        .capture_history_snapshot(&source_path, &AtomicBool::new(false))
        .unwrap();
    let reader = hashed_blobs::Reader::new(&archive, vec![]);
    let mut raw = Vec::new();
    reader
        .read(&hashed_blobs::Hash(*captured.hash().bytes()))
        .unwrap()
        .read_to_end(&mut raw)
        .unwrap();
    assert_eq!(raw.len() as u64, captured.length());
    let snapshot_path = root.path().join("snapshot.sqlite");
    std::fs::write(&snapshot_path, raw).unwrap();
    let snapshot =
        Connection::open_with_flags(&snapshot_path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    assert_eq!(
        snapshot
            .query_row("SELECT value FROM evidence", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        source
            .query_row("SELECT value FROM evidence", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    source.execute_batch("ROLLBACK").unwrap();
}
