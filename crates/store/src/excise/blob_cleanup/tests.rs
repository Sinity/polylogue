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
