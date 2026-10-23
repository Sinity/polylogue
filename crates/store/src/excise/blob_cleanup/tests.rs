use super::*;
use crate::archive::{cache, open_write};

#[test]
fn replay_drains_more_than_one_key_batch_and_keeps_unqueued_files() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    let kept = store.put(b"synthetic unqueued file".as_slice(), 1).unwrap();
    let mut conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let tx = conn.transaction().unwrap();
    for n in 0u64..1003 {
        let mut hash = [0; 32];
        hash[..8].copy_from_slice(&n.to_be_bytes());
        tx.execute(
            "INSERT INTO pending_blob_excision VALUES (?1,'missing-files')",
            [&hash[..]],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    assert!(matches!(
        crate::Writer::start(&archive),
        Err(StoreError::ExcisionPending)
    ));
    recover(&archive, &conn).unwrap();
    assert!(!pending(&conn).unwrap());
    recover(&archive, &conn).unwrap();
    assert!(store.contains(&kept));
}

#[test]
fn recording_retirement_preserves_files_until_commit_and_replay() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    let hash = store.put(b"synthetic staged file".as_slice(), 1).unwrap();
    let hash = BlobHash(hash.0);
    let mut conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    {
        let tx = conn.transaction().unwrap();
        record(&tx, "rolled-back", &hash).unwrap();
        assert!(pending(&tx).unwrap());
        assert!(store.contains(&hashed_blobs::Hash(hash.0)));
    }
    assert!(!pending(&conn).unwrap());
    assert!(store.contains(&hashed_blobs::Hash(hash.0)));
    {
        let tx = conn.transaction().unwrap();
        record(&tx, "committed", &hash).unwrap();
        assert!(store.contains(&hashed_blobs::Hash(hash.0)));
        tx.commit().unwrap();
    }
    assert!(store.contains(&hashed_blobs::Hash(hash.0)));
    recover(&archive, &conn).unwrap();
    assert!(!store.contains(&hashed_blobs::Hash(hash.0)));
    assert!(!pending(&conn).unwrap());
}

#[test]
fn a_later_retirement_visit_error_rolls_back_staged_intent_and_retries() {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::schema::create_archive(&conn).unwrap();
    conn.execute_batch(include_str!(
        "../../schema/migrations/004-local-blob-excision.sql"
    ))
    .unwrap();
    let mut publication = crate::sinex_rows::ExcisionPublication {
        version: 1,
        replacements: vec![],
        retired_batches: vec![],
        digest: String::new(),
    };
    for n in 0u64..4097 {
        let mut hash = [0; 32];
        hash[..8].copy_from_slice(&n.to_be_bytes());
        publication
            .retired_batches
            .push(format!("blake3:{}", hashed_blobs::Hash(hash).hex()));
    }
    let prior = BlobHash([255; 32]);
    let tx = conn.transaction().unwrap();
    record(&tx, "prior", &prior).unwrap();
    tx.commit().unwrap();
    publication.retired_batches.push("invalid batch".into());
    {
        let tx = conn.transaction().unwrap();
        assert!(
            crate::sinex_rows::visit_retired_local_batches(&tx, &publication, |hash| {
                record(&tx, "failed", &hash)
            })
            .is_err()
        );
        assert_eq!(
            tx.query_row("SELECT count(*) FROM pending_blob_excision", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            4098
        );
    }
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pending_blob_excision", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    publication.retired_batches.pop();
    let tx = conn.transaction().unwrap();
    crate::sinex_rows::visit_retired_local_batches(&tx, &publication, |hash| {
        record(&tx, "retry", &hash)
    })
    .unwrap();
    record(&tx, "latest", &BlobHash([0; 32])).unwrap();
    tx.commit().unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pending_blob_excision", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        4098
    );
    assert_eq!(
        conn.query_row(
            "SELECT excision_id FROM pending_blob_excision WHERE hash=?1",
            [&[0u8; 32][..]],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "latest"
    );
    assert_eq!(
        conn.query_row(
            "SELECT excision_id FROM pending_blob_excision WHERE hash=?1",
            [&prior.0[..]],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "prior"
    );
}

#[test]
fn physical_cleanup_keeps_intent_visible_without_blocking_read_admission() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let hash = [7u8; 32];
    conn.execute(
        "INSERT INTO pending_blob_excision VALUES (?1,'synthetic')",
        [&hash[..]],
    )
    .unwrap();
    drop(conn);
    let pool = std::sync::Arc::new(crate::ReadPool::open(&archive).unwrap());
    let (physical, reached) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let location = archive.clone();
    let deletion = std::thread::spawn(move || {
        let conn = open_write(&location.archive_db(), cache::SETUP).unwrap();
        delete_pending(&conn, |_| {
            physical.send(()).unwrap();
            resume.recv().unwrap();
            Ok(())
        })
    });
    reached.recv().unwrap();
    let (reply, result) = std::sync::mpsc::channel();
    let reads = pool.clone();
    let reading = std::thread::spawn(move || {
        let outcome = reads.reader().unwrap().change_position();
        reply
            .send(matches!(outcome, Err(StoreError::ExcisionPending)))
            .unwrap();
    });
    let prompt = result.recv_timeout(std::time::Duration::from_secs(5));
    release.send(()).unwrap();
    deletion.join().unwrap().unwrap();
    reading.join().unwrap();
    assert!(
        prompt.unwrap(),
        "pending cleanup must refuse instead of waiting for physical work"
    );
    pool.reader().unwrap().change_position().unwrap();
}

#[test]
fn acknowledgment_cannot_erase_a_changed_cleanup_intent() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let hash = [8u8; 32];
    conn.execute(
        "INSERT INTO pending_blob_excision VALUES (?1,'before')",
        [&hash[..]],
    )
    .unwrap();
    let outcome = delete_pending(&conn, |_| {
        conn.execute("UPDATE pending_blob_excision SET excision_id='changed'", [])?;
        Ok(())
    });
    assert!(
        matches!(outcome,Err(StoreError::Invalid(message)) if message.contains("changed during deletion"))
    );
    assert_eq!(
        conn.query_row("SELECT excision_id FROM pending_blob_excision", [], |row| {
            row.get::<_, String>(0)
        })
        .unwrap(),
        "changed"
    );
}
