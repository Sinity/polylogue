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
    let id = intern(&tx, "walrus", 1, None, None, None).unwrap();
    assert_eq!(intern(&tx, "walrus", 1, None, None, None).unwrap(), id);
    assert_eq!(intern(&tx, "walrus", 2, None, None, None).unwrap(), id);
    tx.commit().unwrap();
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 2);
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
    assert_eq!(intern(&conn, "walrus", 1, None, None, None).unwrap(), id);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
}

#[test]
fn removal_and_rollback_do_not_leave_search_work() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        intern(&tx, "rolled back", 1, None, None, None).unwrap();
    }
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "removed", 1, None, None, None).unwrap();
    remove(&tx, id, 1).unwrap();
    tx.commit().unwrap();
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
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
    intern(&conn, "walrus", 1, Some(&prepared), None, None).unwrap();
    assert!(index_pending(&conn, || ()).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
}

#[test]
fn pending_byte_limit_commits_complete_entries_and_retry_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let first = intern(&conn, "walrus", 1, None, None, None).unwrap();
    let second = intern(&conn, "narwhal", 2, None, None, None).unwrap();
    intern(&conn, "dolphin", 1, None, None, None).unwrap();
    conn.execute("UPDATE text SET body = X'00' WHERE id = ?1", [second])
        .unwrap();
    assert_eq!(pending_batch(&conn, 1).unwrap(), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert!(index_pending(&conn, || ()).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM output_fts"), 0);
    assert_eq!(count(&conn, "SELECT rowid FROM text_fts"), first);
    conn.execute(
        "UPDATE text SET body = ?1 WHERE id = ?2",
        params![pack_text("narwhal").unwrap(), second],
    )
    .unwrap();
    assert_eq!(index_pending(&conn, || ()).unwrap(), 2);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 2);
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM output_fts WHERE output_fts MATCH 'narwhal'"
        ),
        1
    );
}

#[test]
fn a_later_decode_failure_rolls_back_earlier_search_entries_in_its_transaction() {
    for bulk in [false, true] {
        let first_build = bulk.then(crate::first_build::FirstBuild::default);
        let dir = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
        intern(&conn, "walrus", 1, None, first_build.as_ref(), None).unwrap();
        let second = intern(&conn, "narwhal", 1, None, first_build.as_ref(), None).unwrap();
        conn.execute("UPDATE text SET body = X'00' WHERE id = ?1", [second])
            .unwrap();
        if bulk {
            assert!(index_marked(&conn).is_err());
        } else {
            assert!(index_pending(&conn, || ()).is_err());
            assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
        }
        assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
    }
}

#[test]
fn bulk_and_pending_searches_keep_ascii_unicode_and_search_kinds() {
    for bulk in [false, true] {
        let first_build = bulk.then(crate::first_build::FirstBuild::default);
        let dir = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
        let tx = conn.transaction().unwrap();
        for n in 0..SEARCH_ROWS + 3 {
            intern(
                &tx,
                &format!("walrus{n}"),
                1,
                None,
                first_build.as_ref(),
                None,
            )
            .unwrap();
        }
        let shared = "Łódź Straße Øresund";
        let id = intern(&tx, shared, 1, None, first_build.as_ref(), None).unwrap();
        assert_eq!(
            intern(&tx, shared, 2, None, first_build.as_ref(), None).unwrap(),
            id
        );
        intern(&tx, "notindexed", 0, None, first_build.as_ref(), None).unwrap();
        tx.commit().unwrap();
        if bulk {
            index_marked(&conn).unwrap();
        } else {
            assert_eq!(
                index_pending(&conn, || ()).unwrap(),
                (SEARCH_ROWS + 5) as u64
            );
        }
        assert_eq!(
            count(&conn, "SELECT count(*) FROM text_fts"),
            SEARCH_ROWS + 4
        );
        assert_eq!(count(&conn, "SELECT count(*) FROM output_fts"), 1);
        assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
        for table in ["text_fts", "output_fts"] {
            assert_eq!(
                count(
                    &conn,
                    &format!(
                        "SELECT count(*) FROM {table} WHERE {table} MATCH 'lodz strasse oresund'"
                    )
                ),
                1
            );
            assert_eq!(
                count(
                    &conn,
                    &format!("SELECT count(*) FROM {table} WHERE {table} MATCH 'notindexed'")
                ),
                0
            );
        }
        assert_eq!(
            count(
                &conn,
                &format!(
                    "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'walrus{}'",
                    SEARCH_ROWS + 2
                )
            ),
            1
        );
    }
}
