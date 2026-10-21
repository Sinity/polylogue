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
