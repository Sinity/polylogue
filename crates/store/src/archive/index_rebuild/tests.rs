use super::*;
use crate::archive::{Archive, cache, open_archive_writer, open_read, open_write};

fn meta(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .optional()
        .unwrap()
}

fn set_aside_files(root: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(PREFIX))
        .collect();
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
fn an_index_from_another_format_is_set_aside_and_rebuilt_without_touching_the_archive() {
    for (change, reason) in [
        (
            "PRAGMA user_version = 26",
            IndexRebuildReason::FormatVersion {
                found: 26,
                wanted: schema::INDEX_VERSION,
            },
        ),
        ("PRAGMA application_id = 0", IndexRebuildReason::OlderFormat),
    ] {
        set_aside_and_rebuild(change, reason);
    }
}

#[test]
fn unknown_index1_open_refuses_without_replacing_saved_progress() {
    for building in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
        index
            .execute_batch(
                "CREATE TABLE unexpected(value); INSERT INTO unexpected VALUES('kept');
            INSERT INTO input_ref(pk,id) VALUES(31,'retained');
            INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version,checkpoint)
            VALUES(31,7,'parsing',555,'retained-reading',x'010203');
            INSERT INTO meta VALUES('previous-index-marker','present');",
            )
            .unwrap();
        index
            .execute(
                "UPDATE meta SET value=?1 WHERE key='building'",
                [if building { "1" } else { "0" }],
            )
            .unwrap();
        let identity = schema::index_identity(&index).unwrap();
        let generation = meta(&index, "generation");
        let cursor_key = meta(&index, "cursor_key");
        drop(index);
        drop(lock);
        assert!(matches!(
            Archive::open_or_create(dir.path()),
            Err(StoreError::NoMigration {
                database: "index",
                found: 1
            })
        ));
        assert!(set_aside_files(dir.path()).is_empty());
        let index = open_read(&archive.index_db()).unwrap();
        assert_eq!(schema::index_identity(&index).unwrap(), identity);
        assert_eq!(meta(&index, "generation"), generation);
        assert_eq!(meta(&index, "cursor_key"), cursor_key);
        assert_eq!(
            meta(&index, "previous-index-marker").as_deref(),
            Some("present")
        );
        assert_eq!(
            index
                .query_row(
                    "SELECT input_pk,chain,parsed_upto,hex(checkpoint) FROM parse_state",
                    [],
                    |r| Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, String>(3)?
                    ))
                )
                .unwrap(),
            (31, 7, 555, "010203".into())
        );
    }
}

fn set_aside_and_rebuild(change: &str, reason: IndexRebuildReason) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    crate::Writer::start(&archive).unwrap().derive().unwrap();
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
    let build = schema::build_index_identity(false).unwrap();
    assert_ne!(previous, build, "{change}");
    let name = format!("index-set-aside-{previous}.db");
    assert_eq!(set_aside_files(dir.path()), vec![name.clone()], "{change}");
    let old = open_read(&dir.path().join(&name)).unwrap();
    assert_eq!(
        meta(&old, "previous-index-marker").as_deref(),
        Some("present")
    );
    drop(old);
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(meta(&index, "previous-index-marker"), None);
    assert_eq!(schema::index_identity(&index).unwrap(), build);
    assert!(schema::building(&index).unwrap());
    drop(index);
    assert_eq!(
        rebuild_status(&archive),
        Some(IndexRebuild {
            reason: reason.clone(),
            previous_schema: previous.clone(),
            schema: build.clone(),
            building: true,
            set_aside: Some(name),
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

    // The first build ends: the set-aside file goes, the report stays.
    crate::Writer::start(&archive).unwrap().derive().unwrap();
    assert_eq!(
        set_aside_files(dir.path()),
        Vec::<String>::new(),
        "{change}"
    );
    assert_eq!(
        rebuild_status(&archive),
        Some(IndexRebuild {
            reason,
            previous_schema: previous,
            schema: build,
            building: false,
            set_aside: None,
        })
    );
    drop(lock);
    Archive::open_or_create(dir.path()).unwrap();
}

#[test]
fn a_set_aside_file_left_by_a_stop_is_removed_at_the_next_open_of_a_built_index() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    crate::Writer::start(&archive).unwrap().derive().unwrap();
    drop(lock);
    let left = dir.path().join(file_name("0123456789abcdef"));
    std::fs::write(&left, b"").unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!left.exists());
    assert_eq!(rebuild_status(&archive), None);
}

#[test]
fn an_index_this_build_wrote_has_the_build_identity() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(
        schema::index_identity(&index).unwrap(),
        schema::build_index_identity(false).unwrap()
    );
}
