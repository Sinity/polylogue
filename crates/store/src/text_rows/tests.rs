use super::*;
use crate::archive::{cache, open_write};
use crate::fixture::Fixture;

fn hash_of(text: &str) -> [u8; 32] {
    *blake3::hash(text.as_bytes()).as_bytes()
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[test]
fn repeated_text_is_stored_once_and_indexed_in_both_searches() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    assert_eq!(
        intern(&tx, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap(),
        id
    );
    assert_eq!(
        intern(&tx, "walrus", 2, None, &IndexUpkeep::PerTransaction).unwrap(),
        id
    );
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
    assert_eq!(
        intern(&conn, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap(),
        id
    );
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
}

#[test]
fn removal_and_rollback_do_not_leave_search_work() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        intern(&tx, "rolled back", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    }
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 0);
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
    let tx = conn.transaction().unwrap();
    let id = intern(&tx, "removed", 1, None, &IndexUpkeep::PerTransaction).unwrap();
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
    let id = intern(&conn, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    replace_group_body(&conn, id, b"damaged");
    assert!(index_pending(&conn, || ()).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
}

#[test]
fn pending_byte_limit_commits_complete_entries_and_retry_keeps_the_rest() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let first = intern(&conn, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    let second = intern(&conn, "narwhal", 2, None, &IndexUpkeep::PerTransaction).unwrap();
    intern(&conn, "dolphin", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    replace_group_body(&conn, second, b"\0");
    assert_eq!(pending_batch(&conn, 1).unwrap(), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
    assert!(index_pending(&conn, || ()).is_err());
    assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 1);
    assert_eq!(count(&conn, "SELECT count(*) FROM output_fts"), 0);
    assert_eq!(count(&conn, "SELECT rowid FROM text_fts"), first);
    replace_group_body(&conn, second, &alone("narwhal"));
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
    for at_build_end in [false, true] {
        let upkeep = if at_build_end {
            IndexUpkeep::AtBuildEnd(std::sync::Arc::new(crate::key_maps::KeyMaps::default()))
        } else {
            IndexUpkeep::PerTransaction
        };
        let dir = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        let conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
        intern(&conn, "walrus", 1, None, &upkeep).unwrap();
        let second = intern(&conn, "narwhal", 1, None, &upkeep).unwrap();
        replace_group_body(&conn, second, b"\0");
        if at_build_end {
            assert!(build_search_indexes(&conn).is_err());
        } else {
            assert!(index_pending(&conn, || ()).is_err());
            assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 2);
        }
        assert_eq!(count(&conn, "SELECT count(*) FROM text_fts"), 0);
    }
}

#[test]
fn build_and_pending_searches_keep_ascii_unicode_and_search_kinds() {
    for at_build_end in [false, true] {
        let upkeep = if at_build_end {
            IndexUpkeep::AtBuildEnd(std::sync::Arc::new(crate::key_maps::KeyMaps::default()))
        } else {
            IndexUpkeep::PerTransaction
        };
        let dir = tempfile::tempdir().unwrap();
        let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
        let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
        let tx = conn.transaction().unwrap();
        for n in 0..SEARCH_ROWS + 3 {
            intern(&tx, &format!("walrus{n}"), 1, None, &upkeep).unwrap();
        }
        let shared = "Łódź Straße Øresund";
        let id = intern(&tx, shared, 1, None, &upkeep).unwrap();
        assert_eq!(intern(&tx, shared, 2, None, &upkeep).unwrap(), id);
        intern(&tx, "notindexed", 0, None, &upkeep).unwrap();
        tx.commit().unwrap();
        if at_build_end {
            build_search_indexes(&conn).unwrap();
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
fn build_texts_share_rows_without_sql_readback_or_prefix_aliasing() {
    use rusqlite::hooks::{AuthAction, AuthContext, Authorization};
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = TextIds::new(
        &conn,
        true,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let one = [7; 32];
    let mut two = one;
    two[31] = 8;
    let a = first
        .resolve(
            &conn,
            &one,
            PreparedTexts::pack(["first"])
                .unwrap()
                .get(&hash_of("first")),
            None,
        )
        .unwrap();
    let b = first
        .resolve(
            &conn,
            &two,
            PreparedTexts::pack(["second"])
                .unwrap()
                .get(&hash_of("second")),
            None,
        )
        .unwrap();
    assert_ne!(a, b);
    // Both hits must compare full hashes in writer memory, even when the
    // eight-byte prefixes are equal. SQLite refuses any text-table read.
    conn.flush_prepared_statement_cache();
    conn.authorizer(Some(|context: AuthContext<'_>| match context.action {
        AuthAction::Read {
            table_name: "text", ..
        } => Authorization::Deny,
        _ => Authorization::Allow,
    }))
    .unwrap();
    assert_eq!(first.resolve(&conn, &one, None, None).unwrap(), a);
    assert_eq!(first.resolve(&conn, &two, None, None).unwrap(), b);
    conn.authorizer(None::<fn(AuthContext<'_>) -> Authorization>)
        .unwrap();
    assert!(matches!(
        first.resolve(&conn, &[9; 32], None, None),
        Err(crate::StoreError::Invalid(_))
    ));
}

#[test]
fn failed_text_transaction_reloads_presence_without_reassigning_known_ids() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let first = TextIds::new(
        &conn,
        true,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
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
                PreparedTexts::pack(["rolled back"])
                    .unwrap()
                    .get(&hash_of("rolled back")),
                None,
            )
            .unwrap();
        // The transaction and its text-map changes are both discarded.
    }
    let texts = first.transaction();
    let tx = conn.transaction().unwrap();
    let new = first
        .resolve(
            &tx,
            &kept,
            PreparedTexts::pack(["committed"])
                .unwrap()
                .get(&hash_of("committed")),
            None,
        )
        .unwrap();
    assert_ne!(old, new);
    assert!(first.resolve(&tx, &gone, None, None).is_err());
    tx.commit().unwrap();
    texts.commit();
    assert_eq!(first.resolve(&conn, &kept, None, None).unwrap(), new);
    assert_eq!(count(&conn, "SELECT count(*) FROM text"), 1);
    let transaction = first.transaction();
    let tx = conn.transaction().unwrap();
    assert_eq!(
        first
            .resolve(
                &tx,
                &gone,
                PreparedTexts::pack(["restored"])
                    .unwrap()
                    .get(&hash_of("restored")),
                None
            )
            .unwrap(),
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
fn cached_text_identity_records_no_payload_write() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let identities = TextIds::new(
        &conn,
        true,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let text = "shared synthetic text";
    let hash = *blake3::hash(text.as_bytes()).as_bytes();
    let prepared = PreparedTexts::pack([text]).unwrap();
    let prepared = prepared.get(&hash);
    let tx = conn.transaction().unwrap();
    let written = identities.resolve(&tx, &hash, prepared, None).unwrap();
    let reused = identities.resolve(&tx, &hash, prepared, None).unwrap();
    assert_eq!(reused, written);
    tx.commit().unwrap();
    let cached = identities.resolve(&conn, &hash, prepared, None).unwrap();
    assert_eq!(cached, written);
    let preloaded = TextIds::new(
        &conn,
        true,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap()
    .resolve(&conn, &hash, None, None)
    .unwrap();
    assert_eq!(preloaded, written);
}

#[test]
fn new_text_membership_follows_unindexing_and_rollback_within_the_writer() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    let identities = TextIds::new(
        &conn,
        true,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let text = "synthetic tool output";
    let hash = *blake3::hash(text.as_bytes()).as_bytes();
    let prepared = PreparedTexts::pack([text]).unwrap();
    let bits = std::collections::HashMap::from([(hash, 2)]);
    let membership = NewMembership {
        bits: &bits,
        at_build_end: false,
    };
    let state = |conn: &Connection, id: i64| {
        (
            count(conn, &format!("SELECT indexed FROM text WHERE id={id}")),
            count(
                conn,
                &format!("SELECT count(*) FROM search_pending WHERE text_id={id}"),
            ),
        )
    };

    // A text a step's blocks name joins their index as it is written.
    let transaction = identities.transaction();
    let tx = conn.transaction().unwrap();
    let id = identities
        .resolve(&tx, &hash, prepared.get(&hash), Some(&membership))
        .unwrap();
    assert_eq!(state(&tx, id), (2, 1));
    join_search_index(&tx, &identities, id, 2, false).unwrap();
    assert_eq!(state(&tx, id), (2, 1));
    // No block uses it, so unindexing removes it; activating restores it.
    unindex_unused(&tx, &identities, id, false).unwrap();
    assert_eq!(state(&tx, id), (0, 0));
    join_search_index(&tx, &identities, id, 2, false).unwrap();
    assert_eq!(state(&tx, id), (2, 1));
    unindex_unused(&tx, &identities, id, false).unwrap();
    tx.commit().unwrap();
    transaction.commit();
    assert_eq!(state(&conn, id), (0, 0));

    // Membership set in a transaction that rolls back is not assumed after it.
    let transaction = identities.transaction();
    let tx = conn.transaction().unwrap();
    join_search_index(&tx, &identities, id, 1, false).unwrap();
    assert_eq!(state(&tx, id), (1, 1));
    drop(tx);
    drop(transaction);
    assert_eq!(state(&conn, id), (0, 0));
    let tx = conn.transaction().unwrap();
    join_search_index(&tx, &identities, id, 1, false).unwrap();
    tx.commit().unwrap();
    assert_eq!(state(&conn, id), (1, 1));
}

mod search_build;

#[test]
fn search_work_follows_the_queue_instead_of_text_history() {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    const HISTORY: u64 = 8192;
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let mut conn = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    for number in 0..HISTORY {
        intern(
            &tx,
            &format!("historical text {number}"),
            0,
            None,
            &IndexUpkeep::PerTransaction,
        )
        .unwrap();
    }
    tx.commit().unwrap();
    // A completed bulk build has history and an empty queue when analyzed.
    conn.execute_batch("ANALYZE").unwrap();
    let id = intern(&conn, "walrus", 1, None, &IndexUpkeep::PerTransaction).unwrap();
    assert_eq!(
        intern(&conn, "walrus", 2, None, &IndexUpkeep::PerTransaction).unwrap(),
        id
    );
    // A text removed before indexing must not strand the queue or enter FTS.
    conn.execute(
        "INSERT INTO search_pending(text_id,fts_table) VALUES (?1,1)",
        [i64::MAX],
    )
    .unwrap();
    let work = Arc::new(AtomicU64::new(0));
    let seen = Arc::clone(&work);
    conn.progress_handler(
        100,
        Some(move || {
            seen.fetch_add(100, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    assert_eq!(index_pending(&conn, || ()).unwrap(), 2);
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    // Even one VM instruction per unrelated historical text exceeds the
    // complete budget for two queued entries and the orphan's removal.
    assert!(
        work.load(Ordering::Relaxed) < HISTORY,
        "search drain visited history: {} VM instructions",
        work.load(Ordering::Relaxed)
    );
    assert_eq!(count(&conn, "SELECT count(*) FROM search_pending"), 0);
    for table in ["text_fts", "output_fts"] {
        assert_eq!(
            count(
                &conn,
                &format!("SELECT rowid FROM {table} WHERE {table} MATCH 'walrus'")
            ),
            id
        );
    }
    assert_eq!(index_pending(&conn, || ()).unwrap(), 0);
}
