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

#[test]
fn primary_index1_rebuild_preserves_original_cache_and_paid_and_pending_stores() {
    for boundary in std::iter::once(None)
        .chain(cfg!(feature = "faults").then_some(Some("index.after_set_aside")))
        .chain(cfg!(feature = "faults").then_some(Some("index.after_creation")))
    {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let store = dir.path().join("native");
        let (archive, lock) = Archive::open_primary(&root, &store).unwrap();
        let embeddings = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
        embeddings.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES(x'1234','paid/synthetic',1,x'0000803f',1)",[]).unwrap();
        drop(embeddings);
        drop(lock);
        std::fs::remove_file(archive.index_db()).unwrap();
        crate::fixture::primary_index_before_originals(&archive.index_db()).unwrap();
        let old = open_write(&archive.index_db(), cache::SETUP).unwrap();
        old.execute(
            "INSERT INTO meta VALUES('original-cache-marker','retained')",
            [],
        )
        .unwrap();
        let original_schema = schema::index_identity(&old).unwrap();
        drop(old);
        let original_bytes = std::fs::read(archive.index_db()).unwrap();
        let spool = store
            .join("producer-spool")
            .join("synthetic-producer")
            .join("operations");
        std::fs::create_dir_all(&spool).unwrap();
        let pending = spool.join("authored-pending-body");
        std::fs::write(&pending, b"synthetic immutable pending intent").unwrap();
        #[cfg(feature = "faults")]
        if let Some(boundary) = boundary {
            crate::fault::arm(boundary, "", crate::fault::Action::Fail, 0, 1);
            assert!(Archive::open_primary(&root, &store).is_err());
            crate::fault::clear();
        }
        #[cfg(not(feature = "faults"))]
        let _ = boundary;
        let (current, lock) = Archive::open_primary(&root, &store).unwrap();
        let retained = root.join(file_name(&original_schema));
        assert_eq!(std::fs::read(&retained).unwrap(), original_bytes);
        assert_eq!(
            std::fs::read(&pending).unwrap(),
            b"synthetic immutable pending intent"
        );
        let paid = open_read(&current.embeddings_db()).unwrap();
        assert_eq!(
            paid.query_row(
                "SELECT vector FROM embedded WHERE model='paid/synthetic'",
                [],
                |row| row.get::<_, Vec<u8>>(0)
            )
            .unwrap(),
            [0, 0, 128, 63]
        );
        let index = open_write(&current.index_db(), cache::SETUP).unwrap();
        assert_eq!(
            schema::version(&index).unwrap(),
            schema::PRIMARY_INDEX_VERSION
        );
        assert!(!built(&index, true).unwrap());
        let mut replay = crate::primary::projection::replay_state(&index).unwrap();
        replay.applied_position = Some(5);
        replay.head = Some(5);
        crate::primary::projection::save_replay(&index, &replay).unwrap();
        index
            .execute_batch(
                "INSERT INTO source_input VALUES('original','claude-code');
            INSERT INTO source_revision VALUES('original',1,'first',1,10,1);
            INSERT INTO input_ref(pk,id,format) VALUES(71,'original','claude-code');",
            )
            .unwrap();
        assert!(
            !built(&index, true).unwrap(),
            "head receipt does not replace missing parser progress"
        );
        index.execute_batch("INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto) VALUES(71,1,'reader=1','parsing',5)").unwrap();
        assert!(
            !built(&index, true).unwrap(),
            "an unfinished prefix remains pending"
        );
        index
            .execute_batch("UPDATE parse_state SET status='waiting',at_length=9")
            .unwrap();
        assert!(
            !built(&index, true).unwrap(),
            "a waiting suffix belongs to its exact source length"
        );
        index
            .execute_batch("UPDATE parse_state SET at_length=10")
            .unwrap();
        assert!(
            !built(&index, true).unwrap(),
            "a parser cursor without native admission retains the old cache"
        );
        index
            .execute(
                "UPDATE parse_state SET native_publication=?1",
                [crate::encoding::pack_bytes(b"synthetic admitted receipt").unwrap()],
            )
            .unwrap();
        assert!(
            built(&index, true).unwrap(),
            "waiting at the known source length is an explicit incomplete suffix"
        );
        index.execute_batch("INSERT INTO primary_refused VALUES('unknown',5,'unknown@1','unavailable original evidence',1)").unwrap();
        assert!(
            !built(&index, true).unwrap(),
            "unknown native evidence retains the old cache"
        );
        index.execute_batch("DELETE FROM primary_refused").unwrap();
        remove_when_built(&index, &current.index_db(), true).unwrap();
        assert!(!retained.exists());
        assert_eq!(
            std::fs::read(&pending).unwrap(),
            b"synthetic immutable pending intent"
        );
        drop(index);
        drop(lock);
    }
}

#[test]
fn unknown_primary_shapes_are_refused_without_replacing_the_original_bytes() {
    for version in [1, schema::PRIMARY_INDEX_VERSION] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let store = dir.path().join("native");
        let (archive, lock) = Archive::open_primary(&root, &store).unwrap();
        drop(lock);
        if version == 1 {
            std::fs::remove_file(archive.index_db()).unwrap();
            crate::fixture::primary_index_before_originals(&archive.index_db()).unwrap();
        }
        let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
        index.execute_batch("CREATE TABLE unexplained_primary_data(value BLOB);INSERT INTO unexplained_primary_data VALUES(x'1234')").unwrap();
        drop(index);
        let before = std::fs::read(archive.index_db()).unwrap();
        assert!(
            matches!(Archive::open_primary(&root,&store),Err(StoreError::NoMigration{database:"index",found}) if found==version)
        );
        assert_eq!(std::fs::read(archive.index_db()).unwrap(), before);
        assert!(set_aside_files(&root).is_empty());
    }
}
