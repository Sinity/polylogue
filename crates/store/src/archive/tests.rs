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
    assert!(root.join("index.db.unsynced").exists());
    assert!(!schema::unsynced(&open_read(&archive.index_db()).unwrap()).unwrap());
}

#[test]
fn excision_capture_spills_large_unit_keys_and_rollback_keeps_the_archive() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let mut conn = open_archive_writer(&archive).unwrap();
    {
        let tx = conn.transaction().unwrap();
        let mut insert = tx.prepare("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note','retained','operator','synthetic',1,?2)").unwrap();
        for i in 0..12000 {
            insert
                .execute(rusqlite::params![
                    format!("{i:05}-{}", "k".repeat(256)),
                    i + 1
                ])
                .unwrap();
        }
        drop(insert);
        tx.commit().unwrap();
    }
    let tx = conn.transaction().unwrap();
    let _capture = crate::sinex_rows::Capture::start(&tx, "synthetic-removal").unwrap();
    tx.execute("DELETE FROM note", []).unwrap();
    let count: i64 = tx
        .query_row("SELECT count(*) FROM excision_units", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 12000);
    let key: String = tx
        .query_row(
            "SELECT key FROM excision_units ORDER BY key LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(key, format!("00000-{}", "k".repeat(256)));
    let pages: i64 = tx
        .pragma_query_value(Some("temp"), "page_count", |r| r.get(0))
        .unwrap();
    let page_size: i64 = tx
        .pragma_query_value(Some("temp"), "page_size", |r| r.get(0))
        .unwrap();
    let cache: i64 = tx
        .pragma_query_value(Some("temp"), "cache_size", |r| r.get(0))
        .unwrap();
    let storage: i64 = tx
        .pragma_query_value(None, "temp_store", |r| r.get(0))
        .unwrap();
    assert_eq!(storage, 1, "capture must use file-backed temporary storage");
    assert!(
        cache < 0 && pages * page_size > -cache * 1024,
        "the captured key set must exceed its temporary page-cache target"
    );
    tx.rollback().unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM note WHERE value='retained'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        12000
    );
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_temp_master WHERE name LIKE 'excision_%'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn conflicting_history_refuses_before_blob_only_cleanup() {
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
    let old = open_read(&dir.path().join("index.db.v26")).unwrap();
    assert_eq!(
        old.query_row(
            "SELECT value FROM meta WHERE key='old-format-marker'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "retained"
    );
    let canonical = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        canonical
            .query_row("SELECT value FROM note WHERE id='note'", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        value
    );
}
