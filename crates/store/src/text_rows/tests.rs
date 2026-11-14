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
    prepared.insert(
        *blake3::hash(b"walrus").as_bytes(),
        crate::encoding::PreparedText::new("walrus", b"damaged".to_vec()),
    );
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
    let first = TextIds::new(&conn, true).unwrap();
    let one = [7; 32];
    let mut two = one;
    two[31] = 8;
    let a = first
        .resolve(
            &conn,
            &one,
            Some(&crate::encoding::PreparedText::new(
                "first",
                b"first".to_vec(),
            )),
        )
        .unwrap()
        .id;
    let b = first
        .resolve(
            &conn,
            &two,
            Some(&crate::encoding::PreparedText::new(
                "second",
                b"second".to_vec(),
            )),
        )
        .unwrap()
        .id;
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
    assert_eq!(first.resolve(&conn, &one, None).unwrap().id, a);
    assert_eq!(first.resolve(&conn, &two, None).unwrap().id, b);
    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert!(matches!(
        first.resolve(&conn, &[9; 32], None),
        Err(crate::StoreError::Invalid(_))
    ));
}

#[test]
fn failed_text_transaction_reloads_presence_without_reassigning_known_ids() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = TextIds::new(&conn, true).unwrap();
    let gone = [1; 32];
    let kept = [2; 32];
    let old;
    {
        let _texts = first.transaction();
        let tx = conn.transaction().unwrap();
        old = first
            .resolve(
                &tx,
                &gone,
                Some(&crate::encoding::PreparedText::new(
                    "rolled back",
                    b"rolled back".to_vec(),
                )),
            )
            .unwrap()
            .id;
        // The transaction and its text-map changes are both discarded.
    }
    let texts = first.transaction();
    let tx = conn.transaction().unwrap();
    let new = first
        .resolve(
            &tx,
            &kept,
            Some(&crate::encoding::PreparedText::new(
                "committed",
                b"committed".to_vec(),
            )),
        )
        .unwrap()
        .id;
    assert_ne!(old, new);
    assert!(first.resolve(&tx, &gone, None).is_err());
    tx.commit().unwrap();
    texts.commit();
    assert_eq!(first.resolve(&conn, &kept, None).unwrap().id, new);
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
    let transaction = first.transaction();
    let tx = conn.transaction().unwrap();
    assert_eq!(
        first
            .resolve(
                &tx,
                &gone,
                Some(&crate::encoding::PreparedText::new(
                    "restored",
                    b"restored".to_vec()
                ))
            )
            .unwrap()
            .id,
        old
    );
    tx.commit().unwrap();
    transaction.commit();
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 2);
    let normalized: Vec<u8> = conn
        .query_row("SELECT normalized_hash FROM text WHERE id=?1", [old], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        normalized,
        crate::encoding::normalized_text_hash("restored")
    );
}

#[test]
fn cached_text_identity_does_not_claim_a_payload_write() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let identities = TextIds::new(&conn, true).unwrap();
    let text = "shared synthetic text";
    let hash = *blake3::hash(text.as_bytes()).as_bytes();
    let prepared =
        crate::encoding::PreparedText::new(text, crate::encoding::pack_text(text).unwrap());
    let tx = conn.transaction().unwrap();
    let written = identities.resolve(&tx, &hash, Some(&prepared)).unwrap();
    assert!(written.written);
    let reused = identities.resolve(&tx, &hash, Some(&prepared)).unwrap();
    assert_eq!(reused.id, written.id);
    assert!(!reused.written);
    tx.commit().unwrap();
    let cached = identities.resolve(&conn, &hash, Some(&prepared)).unwrap();
    assert_eq!(cached.id, written.id);
    assert!(!cached.written);
    let preloaded = TextIds::new(&conn, true)
        .unwrap()
        .resolve(&conn, &hash, None)
        .unwrap();
    assert_eq!(preloaded.id, written.id);
    assert!(!preloaded.written);
}
