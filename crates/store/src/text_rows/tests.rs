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
    let id = intern(&tx, "walrus", 1, None, None).unwrap();
    assert_eq!(intern(&tx, "walrus", 1, None, None).unwrap(), id);
    assert_eq!(intern(&tx, "walrus", 2, None, None).unwrap(), id);
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
    assert_eq!(intern(&conn, "walrus", 1, None, None).unwrap(), id);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
}

#[test]
fn removal_and_rollback_do_not_leave_search_work() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        intern(&tx, "rolled back", 1, None, None).unwrap();
    }
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "removed", 1, None, None).unwrap();
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
    intern(&conn, "walrus", 1, Some(&prepared), None).unwrap();
    assert!(index_pending(&conn, || ()).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
}

#[test]
fn pending_byte_limit_commits_complete_entries_and_retry_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let first = intern(&conn, "walrus", 1, None, None).unwrap();
    let second = intern(&conn, "narwhal", 2, None, None).unwrap();
    intern(&conn, "dolphin", 1, None, None).unwrap();
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
        intern(&conn, "walrus", 1, None, first_build.as_ref()).unwrap();
        let second = intern(&conn, "narwhal", 1, None, first_build.as_ref()).unwrap();
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
            intern(&tx, &format!("walrus{n}"), 1, None, first_build.as_ref()).unwrap();
        }
        let shared = "Łódź Straße Øresund";
        let id = intern(&tx, shared, 1, None, first_build.as_ref()).unwrap();
        assert_eq!(
            intern(&tx, shared, 2, None, first_build.as_ref()).unwrap(),
            id
        );
        intern(&tx, "notindexed", 0, None, first_build.as_ref()).unwrap();
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

#[test]
fn first_build_texts_share_rows_without_sql_readback_or_prefix_aliasing() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = FirstBuild::load(&conn).unwrap();
    let one = [7; 32];
    let mut two = one;
    two[31] = 8;
    let a = resolve(&conn, &one, Some(b"first"), Some(&first)).unwrap();
    let b = resolve(&conn, &two, Some(b"second"), Some(&first)).unwrap();
    assert_ne!(a, b);
    // Both hits must use exact hashes in writer memory, even when the
    // eight-byte prefixes are equal. SQLite refuses any text-table read.
    conn.flush_prepared_statement_cache();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "text", ..
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    assert_eq!(resolve(&conn, &one, None, Some(&first)).unwrap(), a);
    assert_eq!(resolve(&conn, &two, None, Some(&first)).unwrap(), b);
    assert!(resolve(&conn, &[9; 32], None, Some(&first)).is_err());
}

#[test]
fn failed_text_transaction_reloads_after_rollback_and_row_reuse() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = FirstBuild::load(&conn).unwrap();
    let gone = [1; 32];
    let kept = [2; 32];
    let old;
    {
        let _texts = first.text_transaction();
        let tx = conn.transaction().unwrap();
        old = resolve(&tx, &gone, Some(b"rolled back"), Some(&first)).unwrap();
        // The transaction and its text-map changes are both discarded.
    }
    let texts = first.text_transaction();
    let tx = conn.transaction().unwrap();
    let new = resolve(&tx, &kept, Some(b"committed"), Some(&first)).unwrap();
    assert_eq!(old, new);
    assert!(resolve(&tx, &gone, None, Some(&first)).is_err());
    tx.commit().unwrap();
    texts.commit();
    assert_eq!(retained(&conn, &kept, Some(&first)).unwrap(), new);
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
}
