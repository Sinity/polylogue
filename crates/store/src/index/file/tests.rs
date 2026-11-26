//! The index files as the open and a second process derive them
//! (docs/architecture.md, Durability: the index file).
use super::*;
use crate::archive::{Archive, cache, open_read, open_write};
use crate::{ReadPool, StoreError, Writer};

/// An archive at `root` with an installed index.
fn built(root: &Path) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    Writer::start(&archive).unwrap().build(Vec::new()).unwrap();
}

fn archive_id(root: &Path) -> ArchiveId {
    archive_identity(&open_read(&root.join("archive.db")).unwrap())
        .unwrap()
        .0
}

/// The record position the installed index has taken in, read from its meta.
fn position(root: &Path) -> u64 {
    crate::meta_get(
        &open_read(&root.join("index.db")).unwrap(),
        ARCHIVE_POSITION,
    )
    .unwrap()
    .map_or(0, |kept| kept.parse().unwrap())
}

fn set_index_meta(root: &Path, key: &str, value: &str) {
    let index = open_write(&root.join("index.db"), cache::SETUP).unwrap();
    crate::meta_set(&index, key, value).unwrap();
}

fn served_reason(archive: &Archive) -> RebuildReason {
    match ReadPool::serving(archive, 1, 8 << 20).unwrap().reader() {
        Err(StoreError::IndexBuilding(build)) => build.reason,
        other => panic!("expected index_building, got {:?}", other.err()),
    }
}

#[test]
fn an_installed_index_is_current_beside_its_archive() {
    let dir = tempfile::tempdir().unwrap();
    built(dir.path());
    assert_eq!(
        IndexFiles::read(dir.path()).unwrap(),
        IndexFiles {
            installed: Installed::Current {
                archive: archive_id(dir.path()),
                position: Position(position(dir.path())),
            },
            rebuilding: Rebuilding::No,
        }
    );
}

/// An index is tied to its archive by the identity in both files' meta,
/// never by where the directory is: a copy built elsewhere and moved into
/// place is the installed index.
#[test]
fn an_index_moved_with_its_archive_stays_current() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("built-here");
    let moved = dir.path().join("moved-here");
    built(&first);
    let id = archive_id(&first);
    std::fs::rename(&first, &moved).unwrap();
    let (archive, _lock) = Archive::open_or_create(&moved).unwrap();
    assert_eq!(
        archive.index_files().unwrap(),
        IndexFiles {
            installed: Installed::Current {
                archive: id,
                position: Position(position(&moved)),
            },
            rebuilding: Rebuilding::No,
        }
    );
    assert_eq!(archive.index_build(), None);
    assert!(!moved.join(DIR).exists());
}

#[test]
fn an_index_of_another_archive_is_replaced_by_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    built(dir.path());
    set_index_meta(dir.path(), ARCHIVE_ID, "0123456789abcdef0123456789abcdef");
    assert_eq!(
        IndexFiles::read(dir.path()).unwrap().installed,
        Installed::Unusable {
            why: RebuildReason::OtherArchive
        }
    );
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(
        archive.index_files().unwrap().rebuilding,
        Rebuilding::Writing {
            reason: RebuildReason::OtherArchive
        }
    );
    assert_eq!(served_reason(&archive), RebuildReason::OtherArchive);
    Writer::start(&archive).unwrap().build(Vec::new()).unwrap();
    assert_eq!(
        archive.index_files().unwrap(),
        IndexFiles {
            installed: Installed::Current {
                archive: archive_id(dir.path()),
                position: Position(position(dir.path())),
            },
            rebuilding: Rebuilding::No,
        }
    );
}

/// `archive.db` put back from an older copy is behind the index beside it.
#[test]
fn an_index_ahead_of_its_archive_is_replaced_by_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    built(dir.path());
    let archive_position =
        crate::write_counter(&open_read(&dir.path().join("archive.db")).unwrap()).unwrap() as u64;
    let ahead = archive_position + 7;
    set_index_meta(dir.path(), ARCHIVE_POSITION, &ahead.to_string());
    let why = RebuildReason::AheadOfArchive {
        index: ahead,
        archive: archive_position,
    };
    assert_eq!(
        IndexFiles::read(dir.path()).unwrap().installed,
        Installed::Unusable { why: why.clone() }
    );
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(served_reason(&archive), why);
    assert_eq!(archive.index_build().unwrap().reason, why);
}

/// A process that does not hold the archive derives the same files the
/// daemon's open made: during a rebuild of an index this build reads, it is
/// served from the installed index; with no index, reads answer
/// `index_building` with the rebuild's reason.
#[test]
fn a_second_process_sees_the_files_the_open_made() {
    let dir = tempfile::tempdir().unwrap();
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        assert_eq!(
            IndexFiles::read(dir.path()).unwrap(),
            archive.index_files().unwrap()
        );
        assert_eq!(served_reason(&archive), RebuildReason::NoIndex);
        Writer::start(&archive).unwrap().build(Vec::new()).unwrap();
    }
    // An extra index changes the schema but not its tables: this build
    // reads the file while a rebuild replaces it.
    open_write(&dir.path().join("index.db"), cache::SETUP)
        .unwrap()
        .execute_batch("CREATE INDEX synthetic_extra ON message(pk)")
        .unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let files = IndexFiles::read(dir.path()).unwrap();
    assert_eq!(files, archive.index_files().unwrap());
    assert_eq!(
        files.rebuilding,
        Rebuilding::Writing {
            reason: RebuildReason::SchemaDiffers
        }
    );
    assert!(files.serves());
    let pool = ReadPool::serving(&archive, 1, 8 << 20).unwrap();
    assert_eq!(pool.reader().unwrap().status(&[]).unwrap().messages, 0);
}

/// The reason is on disk from the rebuild's start: a stop before the
/// install leaves it, and the next open rebuilds for it, though the index
/// that gave it is no longer there to read.
#[test]
fn the_reason_of_a_rebuild_survives_a_stop() {
    let dir = tempfile::tempdir().unwrap();
    built(dir.path());
    set_index_meta(dir.path(), ARCHIVE_ID, "0123456789abcdef0123456789abcdef");
    drop(Archive::open_or_create(dir.path()).unwrap());
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(with_suffix(&dir.path().join("index.db"), suffix));
    }
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(
        archive.index_files().unwrap(),
        IndexFiles {
            installed: Installed::None,
            rebuilding: Rebuilding::Writing {
                reason: RebuildReason::OtherArchive
            },
        }
    );
    assert_eq!(served_reason(&archive), RebuildReason::OtherArchive);
}

/// An index of another format is unusable, replaced by a rebuild that
/// records why; the archive is not touched, and the old index stays
/// `index.db` until the swap.
#[test]
fn an_index_of_another_format_is_replaced_and_the_rebuild_records_why() {
    for (change, why) in [
        (
            "PRAGMA user_version = 26",
            RebuildReason::FormatVersion {
                found: 26,
                wanted: schema::INDEX_VERSION,
            },
        ),
        (
            "CREATE TABLE synthetic_older(value)",
            RebuildReason::SchemaDiffers,
        ),
        ("PRAGMA application_id = 0", RebuildReason::OlderFormat),
    ] {
        let dir = tempfile::tempdir().unwrap();
        built(dir.path());
        let index = open_write(&dir.path().join("index.db"), cache::SETUP).unwrap();
        index.execute_batch(change).unwrap();
        let previous = schema::index_identity(&index).unwrap();
        drop(index);
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        assert_eq!(
            archive.index_files().unwrap(),
            IndexFiles {
                installed: Installed::Unusable { why: why.clone() },
                rebuilding: Rebuilding::Writing {
                    reason: why.clone()
                },
            },
            "{change}"
        );
        assert!(dir.path().join("index.db").exists(), "{change}");
        let record = IndexRebuild {
            reason: why.clone(),
            previous_schema: Some(previous),
            schema: schema::build_index_identity().unwrap(),
        };
        assert_eq!(rebuild_record(dir.path()).unwrap(), Some(record.clone()));
        Writer::start(&archive).unwrap().build(Vec::new()).unwrap();
        let status = ReadPool::serving(&archive, 1, 8 << 20)
            .unwrap()
            .reader()
            .unwrap()
            .status(&[])
            .unwrap();
        assert_eq!(status.index_rebuild, Some(record), "{change}");
        assert!(!dir.path().join(DIR).exists(), "{change}");
    }
}

#[test]
fn a_rebuild_directory_with_no_reason_is_removed_at_the_next_open() {
    let dir = tempfile::tempdir().unwrap();
    built(dir.path());
    std::fs::create_dir(dir.path().join(DIR)).unwrap();
    std::fs::write(dir.path().join(DIR).join("reason.json.partial"), b"{").unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!dir.path().join(DIR).exists());
    assert_eq!(archive.index_files().unwrap().rebuilding, Rebuilding::No);
}
