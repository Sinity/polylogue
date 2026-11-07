use super::*;
use crate::{
    Archive, ReadPool, StoreError,
    completion::{CatalogReader, CatalogWriter, Metadata},
};
use std::sync::Arc;
fn build(pointer: &CatalogPointer, generation: &str, value: &str) -> CatalogWriter {
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: polylogue_model::ChangePosition {
            generation: generation.into(),
            index: 1,
            archive: 1,
        },
        written_at: 100,
    };
    pointer
        .build(&metadata, |writer| {
            writer.put_catalog(&query_language::catalog::Catalog::new("sessions"))?;
            writer.put_values(
                "synthetic",
                &std::collections::BTreeMap::from([(
                    "repo".into(),
                    std::collections::BTreeMap::from([(value.into(), 1)]),
                )]),
            )
        })
        .unwrap()
}
#[test]
fn revoked_pointer_rejects_open_snapshots_and_a_late_old_generation_publisher() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let pointer = Arc::new(CatalogPointer::new(directory.path().join("catalog")));
    let invalidate = Arc::clone(&pointer);
    let _registration =
        Archive::register_scope_invalidator(&root, move || invalidate.invalidate()).unwrap();
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let reads = ReadPool::open(&archive).unwrap();
    let generation = reads
        .reader()
        .unwrap()
        .change_position()
        .unwrap()
        .generation;
    let old = archive.scope_token().unwrap();
    let writer = build(&pointer, &generation, "old-value");
    pointer.activate(&writer, &old).unwrap();
    let snapshot = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.values("repo", "").unwrap()[0].value, "old-value");
    let state = crate::read::scope::state(&root);
    {
        let mut reset = state.begin_reset().unwrap();
        reset.invalidate().unwrap();
    }
    assert!(matches!(snapshot.check(), Err(StoreError::ScopeChanged)));
    assert!(
        CatalogReader::open(&pointer.path, "synthetic")
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        pointer.activate(&writer, &old),
        Err(StoreError::ScopeChanged)
    ));
    state.finish_reset();
    let writer = build(&pointer, "new-generation", "new-value");
    pointer
        .activate(&writer, &archive.scope_token().unwrap())
        .unwrap();
    let current = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(current.values("repo", "").unwrap()[0].value, "new-value");
    assert!(matches!(snapshot.check(), Err(StoreError::ScopeChanged)));
}
#[test]
fn standalone_generation_change_revokes_pointer_before_reopen_finishes() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let pointer = Arc::new(CatalogPointer::new(directory.path().join("catalog")));
    let invalidator = Arc::clone(&pointer);
    let _registration =
        Archive::register_scope_invalidator(&root, move || invalidator.invalidate()).unwrap();
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let token = archive.scope_token().unwrap();
    let writer = build(&pointer, "first", "cached");
    pointer.activate(&writer, &token).unwrap();
    let conn = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    conn.execute(
        "UPDATE meta SET value='synthetic-earlier-boot' WHERE key='boot_id'",
        [],
    )
    .unwrap();
    drop(conn);
    drop(lock);
    drop(archive);
    let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
    assert!(!pointer.path.exists());
    assert!(matches!(token.check(), Err(StoreError::ScopeChanged)));
}

#[test]
fn same_source_generation_rebuild_never_reuses_a_pinned_wal_namespace() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let token = archive.scope_token().unwrap();
    let pointer = CatalogPointer::new(directory.path().join("catalog"));
    let mut first = build(&pointer, "same-source-generation", "first");
    pointer.activate(&first, &token).unwrap();
    let old = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    let next = Metadata {
        archive: "synthetic".into(),
        position: polylogue_model::ChangePosition {
            generation: "same-source-generation".into(),
            index: 2,
            archive: 2,
        },
        written_at: 200,
    };
    first
        .update(&next, |writer| {
            writer.put_values(
                "synthetic",
                &std::collections::BTreeMap::from([(
                    "repo".into(),
                    std::collections::BTreeMap::from([("newer-commit".into(), 1)]),
                )]),
            )
        })
        .unwrap();
    let first_path = first.path.clone();
    drop(first);
    // The pinned old reader keeps the first materialization's WAL alive.
    assert_eq!(old.values("repo", "").unwrap()[0].value, "first");
    let latest = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(latest.values("repo", "").unwrap()[0].value, "newer-commit");
    assert_eq!(latest.metadata.position, next.position);
    let second = build(&pointer, "same-source-generation", "rebuilt");
    assert_ne!(second.path, first_path);
    pointer.activate(&second, &token).unwrap();
    for suffix in ["", "-wal", "-shm"] {
        assert!(!Path::new(&format!("{}{suffix}", first_path.display())).exists());
    }
    assert_eq!(old.values("repo", "").unwrap()[0].value, "first");
    assert_eq!(latest.values("repo", "").unwrap()[0].value, "newer-commit");
    let rebuilt = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(rebuilt.values("repo", "").unwrap()[0].value, "rebuilt");
    assert_eq!(rebuilt.metadata.position.index, 1);
    assert!(matches!(old.check(), Err(StoreError::ScopeChanged)));
    assert!(matches!(latest.check(), Err(StoreError::ScopeChanged)));
    let second_path = second.path.clone();
    drop(second);
    // Recreating a deleted derived database is another new physical build,
    // even while a reader still owns the old materialization's open files.
    std::fs::remove_file(&second_path).unwrap();
    let third = build(&pointer, "same-source-generation", "recreated");
    assert_ne!(third.path, second_path);
    assert_ne!(third.path, first_path);
    pointer.activate(&third, &token).unwrap();
    for suffix in ["", "-wal", "-shm"] {
        assert!(!Path::new(&format!("{}{suffix}", second_path.display())).exists());
    }
    let recreated = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(recreated.values("repo", "").unwrap()[0].value, "recreated");
    assert!(matches!(rebuilt.check(), Err(StoreError::ScopeChanged)));
}

#[test]
fn namespace_retirement_stays_local_and_cannot_fail_an_activated_pointer() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let token = archive.scope_token().unwrap();
    let cache = directory.path().join("cache");
    std::fs::create_dir(&cache).unwrap();
    let pointer = CatalogPointer::new(cache.join("catalog"));
    let outside = directory.path().join("catalog.outside.db");
    std::fs::write(&outside, "unrelated").unwrap();
    std::os::unix::fs::symlink("../catalog.outside.db", &pointer.path).unwrap();
    let first = build(&pointer, "source", "first");
    pointer.activate(&first, &token).unwrap();
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "unrelated");
    pointer.invalidate().unwrap();
    assert!(!first.path.exists());
    let unremovable = cache.join("catalog.unremovable.db");
    std::fs::create_dir(&unremovable).unwrap();
    std::os::unix::fs::symlink("catalog.unremovable.db", &pointer.path).unwrap();
    let second = build(&pointer, "source", "second");
    pointer.activate(&second, &token).unwrap();
    assert!(unremovable.is_dir(), "cleanup cannot remove a directory");
    let reader = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(reader.values("repo", "").unwrap()[0].value, "second");
}
