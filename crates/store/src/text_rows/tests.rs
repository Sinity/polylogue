use super::*;
use crate::archive::{cache, open_write};
use crate::fixture::Fixture;

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[test]
fn repeated_text_is_stored_once_and_indexed_in_both_searches() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "walrus", 1, None, false).unwrap();
    assert_eq!(intern(&tx, "walrus", 1, None, false).unwrap(), id);
    assert_eq!(intern(&tx, "walrus", 2, None, false).unwrap(), id);
    tx.commit().unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert_eq!(index_pending(&conn, 1).unwrap(), 2);
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'walrus'"
        ),
        1
    );
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM output_fts WHERE output_fts MATCH 'walrus'"
        ),
        1
    );
    assert_eq!(intern(&conn, "walrus", 1, None, false).unwrap(), id);
    assert_eq!(index_pending(&conn, 1).unwrap(), 0);
}

#[test]
fn removal_and_rollback_do_not_leave_search_work() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        intern(&tx, "rolled back", 1, None, false).unwrap();
    }
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(index_pending(&conn, 1).unwrap(), 0);
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "removed", 1, None, false).unwrap();
    remove(&tx, id, 1).unwrap();
    tx.commit().unwrap();
    assert_eq!(index_pending(&conn, 1).unwrap(), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
}

#[test]
fn a_failed_decode_keeps_the_pending_entry() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let mut prepared = HashMap::new();
    prepared.insert(*blake3::hash(b"walrus").as_bytes(), b"damaged".to_vec());
    intern(&conn, "walrus", 1, Some(&prepared), false).unwrap();
    assert!(index_pending(&conn, 1).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
}
