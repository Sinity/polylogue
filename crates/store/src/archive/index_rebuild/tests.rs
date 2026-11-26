use super::*;
use crate::archive::{Archive, cache, open_archive_writer, open_read, open_write};
use crate::schema;
use std::path::Path;

fn meta(conn: &Connection, key: &str) -> Option<String> {
    crate::meta_get(conn, key).unwrap()
}

fn build_files(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = match std::fs::read_dir(root.join(super::super::index_build::DIR))
    {
        Ok(entries) => entries
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => panic!("{error}"),
    };
    names.sort();
    names
}

fn rebuild_status(archive: &Archive) -> Option<IndexRebuild> {
    crate::ReadPool::open(archive)
        .unwrap()
        .reader()
        .unwrap()
        .status()
        .unwrap()
        .index_rebuild
}

#[test]
fn an_index_from_another_schema_is_replaced_by_a_build_without_touching_the_archive() {
    for (change, reason) in [
        (
            "PRAGMA user_version = 26",
            IndexRebuildReason::FormatVersion {
                found: 26,
                wanted: schema::INDEX_VERSION,
            },
        ),
        (
            "CREATE TABLE synthetic_older(value)",
            IndexRebuildReason::SchemaDiffers,
        ),
        ("PRAGMA application_id = 0", IndexRebuildReason::OlderFormat),
    ] {
        set_aside_and_rebuild(change, reason);
    }
}

fn set_aside_and_rebuild(change: &str, reason: IndexRebuildReason) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.execute_batch(change).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('previous-index-marker', 'present')",
            [],
        )
        .unwrap();
    let previous = schema::index_identity(&index).unwrap();
    drop(index);
    let archive_conn = open_archive_writer(&archive).unwrap();
    archive_conn
        .execute(
            "INSERT INTO input(id,format) VALUES ('kept-input','synthetic')",
            [],
        )
        .unwrap();
    drop(archive_conn);
    let embeddings = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
    embeddings.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES (x'0102','paid/synthetic',1,x'0000803f',1)", []).unwrap();
    drop(embeddings);
    drop(lock);

    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let build = schema::build_index_identity().unwrap();
    assert_ne!(previous, build, "{change}");
    // The old index waits beside the build, with the write-ahead log it
    // left; checking whether it can be served opened its shared memory.
    assert_eq!(
        build_files(dir.path()),
        vec![
            "index.db",
            "previous.db",
            "previous.db-shm",
            "previous.db-wal"
        ],
        "{change}"
    );
    let old = open_read(&dir.path().join("index-build/previous.db")).unwrap();
    assert_eq!(
        meta(&old, "previous-index-marker").as_deref(),
        Some("present")
    );
    drop(old);
    assert!(!dir.path().join("index.db").exists());
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(meta(&index, "previous-index-marker"), None);
    assert_eq!(schema::index_identity(&index).unwrap(), build);
    drop(index);
    assert_eq!(
        rebuild_status(&archive),
        Some(IndexRebuild {
            reason: reason.clone(),
            previous_schema: previous.clone(),
            schema: build.clone(),
            building: true,
        }),
        "{change}"
    );
    assert_eq!(
        archive.build_progress().unwrap().reason.as_ref(),
        Some(&reason)
    );

    // The daemon stops before the build ends: the next open discards the
    // build file and starts again, for the same reason.
    drop((archive, lock));
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(
        archive.build_progress().unwrap().reason.as_ref(),
        Some(&reason),
        "{change}"
    );
    assert_eq!(
        rebuild_status(&archive),
        Some(IndexRebuild {
            reason: reason.clone(),
            previous_schema: previous.clone(),
            schema: build.clone(),
            building: true,
        }),
        "{change}"
    );
    let archive_conn = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        archive_conn
            .query_row("SELECT format FROM input WHERE id='kept-input'", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
        "synthetic"
    );
    let embeddings = open_read(&archive.embeddings_db()).unwrap();
    assert_eq!(
        embeddings
            .query_row(
                "SELECT vector FROM embedded WHERE model='paid/synthetic'",
                [],
                |r| r.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
        [0, 0, 128, 63]
    );

    // The build ends: the old index goes, the report stays.
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    assert_eq!(build_files(dir.path()), Vec::<String>::new(), "{change}");
    assert_eq!(
        rebuild_status(&archive),
        Some(IndexRebuild {
            reason,
            previous_schema: previous,
            schema: build,
            building: false,
        })
    );
    drop(lock);
    Archive::open_or_create(dir.path()).unwrap();
}

#[test]
fn files_left_beside_a_built_index_are_removed_at_the_next_open() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    drop(lock);
    std::fs::create_dir(dir.path().join("index-build")).unwrap();
    std::fs::write(dir.path().join("index-build/previous.db"), b"").unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert_eq!(build_files(dir.path()), Vec::<String>::new());
    assert!(!dir.path().join("index-build").exists());
    assert_eq!(rebuild_status(&archive), None);
}

#[test]
fn a_build_file_from_another_schema_is_discarded_and_started_again() {
    let dir = tempfile::tempdir().unwrap();
    {
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        drop(crate::Writer::start(&archive).unwrap());
        let build = Connection::open(archive.index_db()).unwrap();
        build
            .execute_batch(
                "CREATE TABLE synthetic_older(value); INSERT INTO meta VALUES ('older-build','present')",
            )
            .unwrap();
    }
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(meta(&index, "older-build"), None);
    assert_eq!(
        schema::index_identity(&index).unwrap(),
        schema::build_index_identity().unwrap()
    );
}
