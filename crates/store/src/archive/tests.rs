use super::*;

#[test]
fn a_second_open_is_refused_while_the_lock_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
    assert!(matches!(
        Archive::open_or_create(&root),
        Err(StoreError::Locked { .. })
    ));
    drop(_lock);
    Archive::open_or_create(&root).unwrap();
}

#[test]
fn an_index_left_unsynced_is_set_aside_and_rebuilt() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    writer.flush().unwrap();
    drop(writer);
    // A stop in the middle of a first build leaves the mark set.
    open_write(&archive.index_db(), cache::SETUP)
        .unwrap()
        .execute("UPDATE meta SET value = '1' WHERE key = 'unsynced'", [])
        .unwrap();
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    assert!(!root.join("index.db.unsynced").exists());
    assert!(!schema::unsynced(&open_read(&archive.index_db()).unwrap()).unwrap());
}

#[test]
fn conflicting_history_refuses_before_blob_only_cleanup() {
    for retired in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
        let hash = store
            .put(b"synthetic unresolved history".as_slice(), 1)
            .unwrap();
        let canonical = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        canonical.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('removed','session:claude-code:first',1,1)",[]).unwrap();
        let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
        crate::excise::recovery::acknowledge(&canonical, &index).unwrap();
        canonical
            .execute(
                "UPDATE excision SET scope='session:claude-code:other' WHERE id='removed'",
                [],
            )
            .unwrap();
        canonical
            .execute(
                "INSERT INTO pending_blob_excision VALUES (?1,'pending')",
                [&hash.0[..]],
            )
            .unwrap();
        assert!(!crate::excise::vector_cleanup::pending(&canonical).unwrap());
        drop(index);
        if retired {
            fs::rename(archive.index_db(), archive.root.join("index.db.v26")).unwrap();
        }
        drop(canonical);
        drop(lock);
        assert!(matches!(
            Archive::open_or_create(dir.path()),
            Err(StoreError::ExcisionConflict { .. })
        ));
        assert!(store.contains(&hash));
        let canonical = open_read(&archive.archive_db()).unwrap();
        assert!(crate::excise::blob_cleanup::pending(&canonical).unwrap());
    }
}

#[test]
fn previous_row_format_is_retired_before_a_new_index_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.pragma_update(None, "user_version", 26).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('old-format-marker', 'retained')",
            [],
        )
        .unwrap();
    drop(index);
    let canonical = open_archive_writer(&archive).unwrap();
    let value = "canonical note value independent of row dictionary";
    canonical.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES ('note','note','note',?1,'operator','synthetic',1,1)", [&value]).unwrap();
    drop(canonical);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(schema::version(&index).unwrap(), schema::INDEX_VERSION);
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='old-format-marker'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(!dir.path().join("index.db.v26").exists());
    let canonical = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        canonical
            .query_row("SELECT value FROM note WHERE id='note'", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        value
    );
}

#[test]
fn an_exhausted_counter_refuses_instead_of_reusing_a_position() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); INSERT INTO meta VALUES ('write_counter','9223372036854775807')").unwrap();
    let tx = conn.transaction().unwrap();
    assert!(
        bump_counter(&tx)
            .unwrap_err()
            .to_string()
            .contains("counter exhausted")
    );
    assert_eq!(
        tx.query_row("SELECT value FROM meta", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "9223372036854775807"
    );
}

mod hash_order;
