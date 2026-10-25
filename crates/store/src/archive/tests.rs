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
fn an_index_left_unsynced_is_rebuilt() {
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
fn an_index_missing_an_excision_is_rebuilt_and_marked_blobs_go_on_open() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    let bytes = vec![7u8; crate::blobs::INLINE_LIMIT + 1];
    let hash = store.put(bytes.as_slice(), 1).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('stale-marker', 'from before')",
            [],
        )
        .unwrap();
    drop(index);
    // A stop after the archive's commit: the excision is recorded and its
    // blob marked, but the index and the file were not reached.
    let archive_db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    archive_db.execute("INSERT INTO excision(id, target,created_at,commit_seq) VALUES ('removed','session:claude-code:first',1,1)",[]).unwrap();
    archive_db
        .execute(
            "INSERT INTO blob(hash,size,stored_size,place,encoding,state) VALUES (?1,?2,?2,'file','plain','deleting')",
            rusqlite::params![&hash.0[..], bytes.len() as i64],
        )
        .unwrap();
    drop(archive_db);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!store.contains(&hash));
    let archive_db = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        archive_db
            .query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='stale-marker'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(!crate::excise::index_behind(&archive_db, &index).unwrap());
}

#[test]
fn an_index_of_another_version_is_replaced_and_the_archive_kept() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.pragma_update(None, "user_version", 26).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('previous-index-marker', 'present')",
            [],
        )
        .unwrap();
    drop(index);
    let archive_conn = open_archive_writer(&archive).unwrap();
    let value = "archive note value independent of row dictionary";
    archive_conn.execute("INSERT INTO note(id, first_note,kind,value,author,machine,created_at,commit_seq) VALUES ('note','note','note',?1,'operator','synthetic',1,1)", [&value]).unwrap();
    drop(archive_conn);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(schema::version(&index).unwrap(), schema::INDEX_VERSION);
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM meta WHERE key='previous-index-marker'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert!(!dir.path().join("index.db.v26").exists());
    let archive_conn = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        archive_conn
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

#[test]
fn opening_clears_blob_files_a_stop_left_before_their_move() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let left = archive.blobs_dir().join("tmp").join("4242-0.part");
    std::fs::write(&left, b"bytes a stop left before the move").unwrap();
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!left.exists());
    assert_eq!(
        std::fs::read_dir(archive.blobs_dir().join("tmp"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn opening_settles_every_blob_row_left_writing() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    let bytes = |fill: u8| vec![fill; crate::blobs::INLINE_LIMIT + 1];
    let put = |fill: u8| polylogue_model::BlobHash(store.put(bytes(fill).as_slice(), 1).unwrap().0);
    let absent = |fill: u8| polylogue_model::BlobHash::of(&bytes(fill));
    // Stored bytes with their file, and without it.
    let used_whole = put(1);
    let used_missing = absent(2);
    // Decoded bytes whose parsed rows never committed, with and without
    // their file.
    let decoded_whole = put(3);
    let decoded_missing = absent(4);
    let conn = open_archive_writer(&archive).unwrap();
    for hash in [used_whole, used_missing, decoded_whole, decoded_missing] {
        conn.execute(
            "INSERT INTO blob(hash,size,stored_size,place,encoding,state) VALUES (?1,?2,0,'file','plain','writing')",
            rusqlite::params![&hash.0[..], bytes(0).len() as i64],
        )
        .unwrap();
    }
    conn.execute(
        "INSERT INTO input(id,format) VALUES ('stored','export-file')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('stored',1,'first',1,?1,1)",
        [2 * bytes(0).len() as i64],
    )
    .unwrap();
    for (offset, hash) in [(0, used_whole), (bytes(0).len() as i64, used_missing)] {
        conn.execute(
            "INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES ('stored',1,?1,?2,?3,0)",
            rusqlite::params![offset, bytes(0).len() as i64, &hash.0[..]],
        )
        .unwrap();
    }
    drop(conn);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_read(&archive.archive_db()).unwrap();
    let state = |hash: polylogue_model::BlobHash| -> Option<String> {
        conn.query_row(
            "SELECT state FROM blob WHERE hash = ?1",
            [&hash.0[..]],
            |r| r.get(0),
        )
        .optional()
        .unwrap()
    };
    assert_eq!(state(used_whole).as_deref(), Some("present"));
    assert_eq!(state(used_missing).as_deref(), Some("writing"));
    assert_eq!(state(decoded_whole), None);
    assert_eq!(state(decoded_missing), None);
    assert!(store.contains(&hashed_blobs::Hash(used_whole.0)));
    assert!(!store.contains(&hashed_blobs::Hash(decoded_whole.0)));
    assert_eq!(
        crate::staging::pending_blobs(&conn, "").unwrap()[0].blob,
        used_missing
    );
}
