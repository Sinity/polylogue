use super::*;
use crate::archive::{cache, open_write};

#[test]
fn both_database_handles_use_the_existing_canonical_file() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let canonical = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let guard = write(&index).unwrap();
    let file = File::open(archive.archive_db()).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    let guard = write(&canonical).unwrap();
    assert!(matches!(
        file.try_lock_shared(),
        Err(TryLockError::WouldBlock)
    ));
    drop(guard);
    file.try_lock_shared().unwrap();
}

#[test]
fn failed_or_dropped_transactions_release_publication() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _writer) = Archive::open_or_create(dir.path()).unwrap();
    let mut canonical = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    {
        let _guard = write(&canonical).unwrap();
        let tx = canonical.transaction().unwrap();
        assert!(
            tx.execute("INSERT INTO no_such_table VALUES (1)", [])
                .is_err()
        );
    }
    admit(&archive, || panic!("publication was not released")).unwrap();
}

#[test]
fn in_memory_derivations_have_no_shared_archive_file() {
    let conn = Connection::open_in_memory().unwrap();
    let _guard = write(&conn).unwrap();
    conn.execute("CREATE TABLE synthetic(value)", []).unwrap();
}
