use super::*;
use crate::{
    archive::{cache, open_write},
    schema,
};

fn index(path: &std::path::Path) -> Connection {
    let db = open_write(path, cache::SETUP).unwrap();
    schema::create_index(&db).unwrap();
    db
}

#[test]
fn only_the_recorded_index_names_are_owned() {
    for name in [
        "index.db.unsynced",
        "index.db.v26",
        "index.db.v0",
        "index.db.excision-00000000-0000-4000-8000-000000000001",
    ] {
        assert!(owned_name(name), "{name}");
    }
    for name in [
        "index.db",
        "archive.db",
        "embeddings.db",
        "index.db.backup",
        "index.db.v",
        "index.db.v01",
        "index.db.v-1",
        "index.db.v1.extra",
        "index.db.excision-not-an-id",
    ] {
        assert!(!owned_name(name), "{name}");
    }
}

#[test]
fn obsolete_indexes_are_removed_and_neighbors_survive() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    for name in [
        "index.db.v26",
        "index.db.unsynced",
        "index.db.excision-00000000-0000-4000-8000-000000000001",
    ] {
        drop(index(&dir.path().join(name)));
    }
    let neighbor = dir.path().join("index.db.backup");
    fs::write(&neighbor, b"operator-owned file").unwrap();
    let canonical = open_read(&archive.archive_db()).unwrap();
    remove(&archive, &canonical).unwrap();
    remove(&archive, &canonical).unwrap();
    assert!(checked(&archive, &canonical).unwrap().is_empty());
    assert_eq!(fs::read(neighbor).unwrap(), b"operator-owned file");
    assert!(archive.index_db().exists());
    assert!(archive.embeddings_db().exists());
}

#[test]
fn a_conflicting_old_policy_preserves_every_copy() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    drop(index(&dir.path().join("index.db.v1")));
    let conflict = index(&dir.path().join("index.db.v2"));
    crate::excise::recovery::set(
        &conflict,
        &crate::excise::recovery::Revision {
            version: 1,
            id: "unresolved-history".into(),
        },
    )
    .unwrap();
    drop(conflict);
    let canonical = open_read(&archive.archive_db()).unwrap();
    assert!(matches!(
        remove(&archive, &canonical),
        Err(StoreError::ExcisionConflict { .. })
    ));
    assert!(dir.path().join("index.db.v1").exists());
    assert!(dir.path().join("index.db.v2").exists());
}

#[test]
fn invalid_files_and_orphan_sidecars_refuse_before_any_unlink() {
    for name in ["index.db.v2", "index.db.v2-wal"] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        drop(index(&dir.path().join("index.db.v1")));
        fs::write(dir.path().join(name), b"unresolved bytes").unwrap();
        let canonical = open_read(&archive.archive_db()).unwrap();
        assert!(remove(&archive, &canonical).is_err());
        assert!(dir.path().join("index.db.v1").exists());
        assert_eq!(
            fs::read(dir.path().join(name)).unwrap(),
            b"unresolved bytes"
        );
    }
}

#[cfg(unix)]
#[test]
fn symlinks_are_preserved_without_reading_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    std::os::unix::fs::symlink(archive.archive_db(), dir.path().join("index.db.v2")).unwrap();
    let canonical = open_read(&archive.archive_db()).unwrap();
    assert!(remove(&archive, &canonical).is_err());
    assert!(
        fs::symlink_metadata(dir.path().join("index.db.v2"))
            .unwrap()
            .is_symlink()
    );
    assert!(archive.archive_db().exists());
}

#[test]
fn a_reader_blocking_wal_checkpoint_preserves_the_database_and_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("index.db.v2");
    let db = index(&path);
    let reader = open_read(&path).unwrap();
    reader.execute_batch("BEGIN; SELECT * FROM meta;").unwrap();
    db.execute("INSERT INTO meta VALUES ('later', 'retained')", [])
        .unwrap();
    let canonical = open_read(&archive.archive_db()).unwrap();
    assert!(remove(&archive, &canonical).is_err());
    assert!(path.exists());
    assert!(dir.path().join("index.db.v2-wal").exists());
    drop(reader);
    remove(&archive, &canonical).unwrap();
    assert!(!path.exists());
    assert!(!dir.path().join("index.db.v2-wal").exists());
    assert!(!dir.path().join("index.db.v2-shm").exists());
}
