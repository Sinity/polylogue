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
fn revoked_pointer_rejects_open_snapshots_and_old_writers() {
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
    let (archive, lock, registration) =
        Archive::open_or_create_with_invalidator(&root, move || invalidator.invalidate()).unwrap();
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
    drop(writer);
    drop(registration);
    drop(lock);
    drop(archive);
    let invalidator = Arc::clone(&pointer);
    let (_archive, _lock, _registration) =
        Archive::open_or_create_with_invalidator(&root, move || invalidator.invalidate()).unwrap();
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
    // The pinned old reader keeps the first catalog file's WAL alive.
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
    // Recreating a deleted derived database is another new build,
    // even while a reader still holds the old catalog's open files.
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

#[test]
fn resume_validates_the_publication_identity_before_reusing_its_database() {
    use std::os::unix::fs::MetadataExt;
    for case in [
        "current",
        "aliased_parent",
        "version6",
        "archive",
        "generation",
        "index",
        "archive_position",
        "negative_index",
        "negative_archive",
        "grammar",
        "machine",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let parent = if case == "aliased_parent" {
            let real = directory.path().join("real");
            fs::create_dir(&real).unwrap();
            let alias = directory.path().join("alias");
            std::os::unix::fs::symlink(&real, &alias).unwrap();
            alias
        } else {
            directory.path().to_owned()
        };
        let pointer = CatalogPointer::new(parent.join("catalog"));
        let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
        let catalog = query_language::catalog::Catalog::new("sessions");
        let metadata = Metadata {
            archive: "synthetic".into(),
            position: polylogue_model::ChangePosition {
                generation: "generation".into(),
                index: 10,
                archive: 20,
            },
            written_at: 123,
        };
        let writer = pointer
            .build(&metadata, |writer| {
                writer.put_catalog(&catalog)?;
                writer.put_publisher_machine("machine")?;
                writer.put_values(
                    "source",
                    &std::collections::BTreeMap::from([(
                        "model".into(),
                        std::collections::BTreeMap::from([("retained-model".into(), 3)]),
                    )]),
                )
            })
            .unwrap();
        pointer
            .activate(&writer, &archive.scope_token().unwrap())
            .unwrap();
        if matches!(case, "negative_index" | "negative_archive") {
            let mut invalid = metadata.position.clone();
            if case == "negative_index" {
                invalid.index = -1;
            } else {
                invalid.archive = -1;
            }
            writer
                .connection()
                .execute(
                    "UPDATE metadata SET value=?1 WHERE key='position'",
                    [serde_json::to_string(&invalid).unwrap()],
                )
                .unwrap();
        }
        if case == "version6" {
            writer
                .connection()
                .execute("UPDATE metadata SET value='6' WHERE key='version'", [])
                .unwrap();
        }
        drop(writer);
        let inode = fs::metadata(&pointer.path).unwrap().ino();
        let mut current = metadata.clone();
        let mut expected_catalog = catalog;
        let machine = if case == "machine" {
            "other"
        } else {
            "machine"
        };
        match case {
            "archive" => current.archive = "other".into(),
            "generation" => current.position.generation = "other".into(),
            "index" => current.position.index -= 1,
            "archive_position" => current.position.archive -= 1,
            "grammar" => expected_catalog = query_language::catalog::Catalog::new("messages"),
            _ => {}
        }
        let resumed = pointer
            .resume(&current, &expected_catalog, machine)
            .unwrap();
        assert_eq!(
            resumed.is_some(),
            matches!(case, "current" | "aliased_parent"),
            "{case}"
        );
        assert_eq!(fs::metadata(&pointer.path).unwrap().ino(), inode);
        if let Some((mut writer, saved, _vocabulary)) = resumed {
            assert_eq!(saved.position, metadata.position);
            assert_eq!(
                CatalogReader::open(&pointer.path, "synthetic")
                    .unwrap()
                    .unwrap()
                    .values("model", "retained")
                    .unwrap()[0]
                    .count,
                3
            );
            writer
                .update(&saved, |writer| {
                    writer.put_values(
                        "changed",
                        &std::collections::BTreeMap::from([(
                            "model".into(),
                            std::collections::BTreeMap::from([("new-model".into(), 1)]),
                        )]),
                    )
                })
                .unwrap();
            assert_eq!(
                CatalogReader::open(&pointer.path, "synthetic")
                    .unwrap()
                    .unwrap()
                    .values("model", "new")
                    .unwrap()[0]
                    .count,
                1
            );
        }
        if case == "version6" {
            let rebuilt = pointer
                .build(&metadata, |writer| {
                    writer.put_catalog(&expected_catalog)?;
                    writer.put_publisher_machine("machine")?;
                    writer.put_values(
                        "new-source",
                        &std::collections::BTreeMap::from([(
                            "model".into(),
                            std::collections::BTreeMap::from([("rebuilt-model".into(), 1)]),
                        )]),
                    )
                })
                .unwrap();
            pointer
                .activate(&rebuilt, &archive.scope_token().unwrap())
                .unwrap();
            assert_ne!(fs::metadata(&pointer.path).unwrap().ino(), inode);
            let cache = CatalogReader::open(&pointer.path, "synthetic")
                .unwrap()
                .unwrap();
            assert_eq!(cache.values("model", "").unwrap()[0].value, "rebuilt-model");
        }
    }
}

#[test]
fn an_offline_reader_cannot_leave_a_resumed_publisher_unwritable() {
    use std::os::unix::fs::MetadataExt;
    let directory = tempfile::tempdir().unwrap();
    let pointer = CatalogPointer::new(directory.path().join("catalog"));
    let (archive, _lock) = Archive::open_or_create(&directory.path().join("archive")).unwrap();
    let catalog = query_language::catalog::Catalog::new("sessions");
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: polylogue_model::ChangePosition {
            generation: "generation".into(),
            index: 1,
            archive: 1,
        },
        written_at: 123,
    };
    let populate = |writer: &mut CatalogWriter, value: &str| {
        writer.put_catalog(&catalog)?;
        writer.put_publisher_machine("machine")?;
        writer.put_values(
            "source",
            &std::collections::BTreeMap::from([(
                "model".into(),
                std::collections::BTreeMap::from([(value.into(), 1)]),
            )]),
        )
    };
    let writer = pointer
        .build(&metadata, |writer| populate(writer, "retained-model"))
        .unwrap();
    pointer
        .activate(&writer, &archive.scope_token().unwrap())
        .unwrap();
    drop(writer);
    let pinned = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(
        pinned.values("model", "").unwrap()[0].value,
        "retained-model"
    );
    let target = fs::read_link(&pointer.path).unwrap();
    let identity = fs::metadata(&pointer.path).unwrap().ino();
    let resumed = match pointer.resume(&metadata, &catalog, "machine") {
        Ok(Some((writer, _, _))) => Some(writer),
        Err(StoreError::Sqlite(error))
            if error.sqlite_error_code() == Some(rusqlite::ErrorCode::ReadOnly) =>
        {
            None
        }
        Err(error) => panic!("expected transient readonly mapping: {error}"),
        Ok(None) => panic!("valid catalog was rejected"),
    };
    assert_eq!(fs::read_link(&pointer.path).unwrap(), target);
    assert_eq!(
        pinned.values("model", "").unwrap()[0].value,
        "retained-model"
    );
    drop(pinned);
    let mut writer = resumed.unwrap_or_else(|| {
        pointer
            .resume(&metadata, &catalog, "machine")
            .unwrap()
            .unwrap()
            .0
    });
    writer
        .update(&metadata, |writer| populate(writer, "new-model"))
        .unwrap();
    assert_eq!(fs::read_link(&pointer.path).unwrap(), target);
    assert_eq!(fs::metadata(&pointer.path).unwrap().ino(), identity);
    let current = CatalogReader::open(&pointer.path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(current.values("model", "").unwrap()[0].value, "new-model");
}

#[test]
fn resume_does_not_follow_a_cache_target_into_another_file() {
    let directory = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let metadata = Metadata {
        archive: "synthetic".into(),
        position: polylogue_model::ChangePosition {
            generation: "g".into(),
            index: 1,
            archive: 1,
        },
        written_at: 1,
    };
    let catalog = query_language::catalog::Catalog::new("sessions");
    let elsewhere = CatalogPointer::new(outside.path().join("catalog"));
    let writer = elsewhere
        .build(&metadata, |writer| {
            writer.put_catalog(&catalog)?;
            writer.put_publisher_machine("machine")
        })
        .unwrap();
    let target = writer.path.clone();
    drop(writer);
    let before = fs::read(&target).unwrap();
    std::os::unix::fs::symlink(&target, directory.path().join("catalog.link.db")).unwrap();
    let pointer = CatalogPointer::new(directory.path().join("catalog"));
    std::os::unix::fs::symlink("catalog.link.db", &pointer.path).unwrap();
    assert!(pointer.resume(&metadata, &catalog, "machine").is_err());
    assert_eq!(fs::read(&target).unwrap(), before);
}

#[test]
fn failed_standalone_open_revokes_the_previous_catalog() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let token = archive.scope_token().unwrap();
    let pointer = Arc::new(CatalogPointer::new(directory.path().join("catalog")));
    let writer = build(&pointer, "generation", "cached");
    pointer.activate(&writer, &token).unwrap();
    let conn = crate::archive::open_write(&archive.archive_db(), 1).unwrap();
    conn.execute_batch("PRAGMA user_version=999").unwrap();
    drop(conn);
    drop(writer);
    drop(lock);
    drop(archive);
    let invalidator = Arc::clone(&pointer);
    assert!(
        Archive::open_or_create_with_invalidator(&root, move || invalidator.invalidate()).is_err()
    );
    assert!(!pointer.path.exists());
    assert!(matches!(token.check(), Err(StoreError::ScopeChanged)));
}

#[test]
fn blocked_startup_revokes_the_previous_catalog() {
    {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("archive");
        let (archive, lock) = Archive::open_or_create(&root).unwrap();
        let pointer = Arc::new(CatalogPointer::new(directory.path().join("catalog")));
        let writer = build(&pointer, "generation", "cached");
        pointer
            .activate(&writer, &archive.scope_token().unwrap())
            .unwrap();
        drop(writer);
        drop(lock);
        drop(archive);
        let invalidator = Arc::clone(&pointer);
        let invalidate = move || invalidator.invalidate();
        let opened = Archive::open_blocked_with_invalidator(&root, invalidate);
        assert!(opened.is_ok());
        assert!(!pointer.path.exists());
    }
}
