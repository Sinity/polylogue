mod boundaries;
mod raw_paths;
use super::*;

#[test]
fn cache_schema_renames_tables_without_changing_input_columns_or_counter_keys() {
    let cache = Connection::open_in_memory().unwrap();
    cache.execute_batch(crate::schema::INDEX).unwrap();
    cache.execute_batch(&schema().unwrap().ddl).unwrap();
    cache
        .execute(
            "INSERT INTO source_input(id,origin) VALUES ('i','test')",
            [],
        )
        .unwrap();
    cache.execute("INSERT INTO source_revision(input,number,produced,chain,length,commit_seq) VALUES ('i',1,'imported',41,9,1)",[]).unwrap();
    let input: String = cache
        .query_row("SELECT input FROM source_revision", [], |row| row.get(0))
        .unwrap();
    assert_eq!(input, "i");
    let total: i64 = cache
        .query_row(
            "SELECT count FROM source_row_count WHERE name='input'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(total, 1);
    let parent: String = cache
        .query_row(
            "SELECT \"table\" FROM pragma_foreign_key_list('source_revision') LIMIT 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(parent, "source_input");
    let index_count: i64 = cache
        .query_row("SELECT n FROM row_count WHERE name='message'", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(index_count, 0);
}

#[test]
fn standalone_logical_source_views_preserve_rowids_and_live_values() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("archive.db");
    let standalone = Connection::open(&path).unwrap();
    standalone.execute_batch(&crate::schema::ARCHIVE).unwrap();
    standalone
        .execute("INSERT INTO input(id,origin) VALUES ('i','test')", [])
        .unwrap();
    let reader = Connection::open_in_memory().unwrap();
    reader
        .execute("ATTACH DATABASE ?1 AS a", [path.to_string_lossy().as_ref()])
        .unwrap();
    views(&reader).unwrap();
    let before: i64 = reader
        .query_row(
            "SELECT count FROM source_row_count WHERE name='input'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(before, 1);
    standalone
        .execute("INSERT INTO input(id,origin) VALUES ('j','test')", [])
        .unwrap();
    let actual: (i64, String) = reader
        .query_row(
            "SELECT rowid,id FROM source_input WHERE id='j'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let expected: (i64, String) = standalone
        .query_row("SELECT rowid,id FROM input WHERE id='j'", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn later_chunks_keep_their_commit_when_a_revision_grows() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        if primary {
            conn.execute_batch(&schema().unwrap().ddl).unwrap();
        } else {
            conn.execute_batch(&crate::schema::ARCHIVE).unwrap();
        }
        let source = Source::of(&conn).unwrap();
        let hash = polylogue_model::BlobHash::of(b"synthetic");
        conn.execute(
            &source
                .sql("INSERT INTO @source.blob(hash,size,stored_size,storage) VALUES (?1,8,8,x'')"),
            [&hash.0[..]],
        )
        .unwrap();
        let mut row = crate::input_rows::InputRows {
            capture_read: None,
            input: "input".into(),
            origin: "synthetic".into(),
            chain: None,
            observation: None,
            revision: Some(crate::input_rows::Revision {
                number: 1,
                produced: crate::input_rows::Produced::First,
                chain: 1,
                length: 4,
                seen_at: Some(1),
                mtime_ns: None,
                ctime_ns: None,
            }),
            extents: vec![crate::input_rows::Extent {
                chain: 1,
                offset: 0,
                length: 4,
                blob: hash,
                blob_offset: 0,
            }],
        };
        row.insert(&conn, 2).unwrap();
        row.revision.as_mut().unwrap().length = 8;
        row.extents[0].offset = 4;
        row.extents[0].blob_offset = 4;
        row.insert(&conn, 9).unwrap();
        assert_eq!(
            Source::committed_through(&conn, "input", 1, 4).unwrap(),
            Some(2)
        );
        assert_eq!(
            Source::committed_through(&conn, "input", 1, 6).unwrap(),
            Some(9)
        );
        assert_eq!(
            Source::committed_through(&conn, "input", 1, 9).unwrap(),
            None
        );
        row.insert(&conn, 15).unwrap();
        assert_eq!(
            Source::committed_through(&conn, "input", 1, 8).unwrap(),
            Some(9)
        );
        // Another extent at a stored place is refused, and the one stored
        // there stays as it was.
        let stored = |conn: &Connection| -> Vec<(i64, i64, i64, i64)> {
            conn.prepare(&source.sql(
                "SELECT offset,length,blob_offset,commit_seq FROM @source.extent ORDER BY offset",
            ))
            .unwrap()
            .query_map([], |r| <(i64, i64, i64, i64)>::try_from(r))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
        };
        let before = stored(&conn);
        assert_eq!(before, [(0, 4, 0, 2), (4, 4, 4, 9)]);
        let mut other = row.clone();
        other.extents[0].length = 2;
        match other.insert(&conn, 21) {
            Err(crate::StoreError::Invalid(reason)) => {
                assert_eq!(reason, "a stored extent cannot be replaced")
            }
            result => panic!("expected the refusal, got {result:?}"),
        }
        assert_eq!(stored(&conn), before);
        conn.execute(
            &source.sql("UPDATE @source.extent SET length=-1 WHERE offset=0"),
            [],
        )
        .unwrap();
        let error = Source::committed_through(&conn, "input", 1, 8).unwrap_err();
        assert_eq!(
            error.to_string(),
            "stored column 1 is unreadable: negative source extent length"
        );
        conn.execute(&source.sql("DELETE FROM @source.extent WHERE offset=0"), [])
            .unwrap();
        assert_eq!(
            Source::committed_through(&conn, "input", 1, 8).unwrap(),
            None
        );
    }
}

#[test]
fn shorter_prefix_has_a_commit_and_keeps_progress() {
    use crate::{Batch, Change, InputKey, Observation, Writer};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let (archive, lock) = crate::Archive::open_or_create(&root).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let (blob, prepared) = archive.prepare_blob(b"abcdefgh").unwrap();
    let results = writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch: Batch {
                input: InputKey {
                    id: "native".into(),
                    format: "synthetic".into(),
                    route: polylogue_model::Route::Live,
                },
                change: Change::First,
                observation: Some(Observation {
                    machine: "machine".into(),
                    host: "host".into(),
                    path: "/home/op/session".into(),
                    path_bytes: b"/home/op/session".to_vec(),
                    facts: None,
                    seen_at_ms: 10,
                    import: None,
                    container: None,
                    database: None,
                }),
                continues: false,
                new_blob: Some(blob),
                blobs: Vec::new(),
                length: 8,
                parse: Vec::new(),
                capture_read: None,
                release: Default::default(),
            },
            files: prepared.into_iter().collect(),
        }])
        .unwrap();
    assert!(results[0].is_ok());
    let (prefix_blob, prepared) = archive.prepare_blob(b"abcd").unwrap();
    writer.store_blob(prefix_blob.clone(), prepared).unwrap();
    drop(writer);
    let mut standalone = Connection::open(archive.archive_db()).unwrap();
    let tx = standalone.transaction().unwrap();
    crate::input_rows::InputRows {
        capture_read: None,
        input: "native".into(),
        origin: "synthetic".into(),
        chain: None,
        revision: None,
        extents: Vec::new(),
        observation: Some(crate::input_rows::ObservationRow {
            id: "prefix-four".into(),
            machine: "machine".into(),
            host: "host".into(),
            path_bytes: b"/home/op/session".to_vec(),
            seen: crate::input_rows::Seen::Prefix {
                chain: 1,
                length: 4,
            },
            device: None,
            inode: None,
            size: Some(4),
            mtime_ns: None,
            ctime_ns: None,
            polylogue_version: "test".into(),
            seen_at: 20,
            details_json: None,
            container: None,
        }),
    }
    .insert(&tx, 100)
    .unwrap();
    tx.commit().unwrap();
    assert_eq!(
        standalone
            .query_row(
                "SELECT count(*) FROM revision WHERE input='native'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        1
    );
    let source_commit = Source::reading_commit(&standalone, "native", 1, 4).unwrap();
    let first_batch: String = standalone
        .query_row(
            "SELECT id FROM observation WHERE input='native' AND id<>'prefix-four'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    // The location that captured a prefix's bytes names it, also when the
    // file was seen back holding that prefix later.
    assert_eq!(source_commit.observation.as_ref(), Some(&first_batch));
    assert_eq!(
        Source::reading_commit(&standalone, "native", 1, 8)
            .unwrap()
            .observation,
        Some(first_batch)
    );
    drop(standalone);
    drop(lock);
    let (archive, _lock) = crate::Archive::open_or_create(&root).unwrap();
    let reads = crate::ReadPool::open(&archive).unwrap();
    let known = reads
        .reader()
        .unwrap()
        .known_watched_input_at("machine", b"/home/op/session")
        .unwrap()
        .unwrap();
    assert_eq!((known.chain, known.stored), (Some(1), 8));
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(
        &mut archive
            .blobs()
            .unwrap()
            .read(&hashed_blobs::Hash(prefix_blob.hash.0))
            .unwrap(),
        &mut bytes,
    )
    .unwrap();
    assert_eq!(bytes, b"abcd");
}

#[test]
fn source_reference_namespace_keeps_archive_metadata_distinct_from_index_metadata() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("archive.db");
    let standalone = Connection::open(&path).unwrap();
    standalone.execute_batch(&crate::schema::ARCHIVE).unwrap();
    standalone
        .execute(
            "INSERT INTO meta(key,value) VALUES('namespace','archive')",
            [],
        )
        .unwrap();
    let reader = Connection::open_in_memory().unwrap();
    reader.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); INSERT INTO meta VALUES('namespace','index');").unwrap();
    reader
        .execute("ATTACH DATABASE ?1 AS a", [path.to_string_lossy().as_ref()])
        .unwrap();
    views(&reader).unwrap();
    let source = Source::reading(&reader).unwrap();
    let value: String = reader
        .query_row(
            &source.sql("SELECT value FROM @source.meta WHERE key='namespace'"),
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(value, "archive");
    assert!(matches!(Source::of(&reader).unwrap(), Source::Standalone));
}

#[test]
fn narrow_schema_uses_the_sqlite_namespace_rewriter() {
    let schema = from_archive(crate::annotations::SCHEMA).unwrap();
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.execute_batch(&schema.ddl).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM sqlite_schema WHERE type='table'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        5
    );
    conn.execute("INSERT INTO source_annotation(id,root,kind,author,via,machine,created_at,commit_seq) VALUES('fact','fact','note','operator','cli','host',1,1)",[]).unwrap();
    conn.execute(
        "INSERT INTO source_annotation_link(annotation,role,ref) VALUES('fact','about','session:codex:s')",
        [],
    )
    .unwrap();
    assert!(
        conn.execute(
            "INSERT INTO source_annotation_link(annotation,role,ref) VALUES('absent','about','session:codex:s')",
            []
        )
        .is_err()
    );
    conn.execute("INSERT INTO source_annotation_import(annotation,input,chain,machine,actor,at,commit_seq) VALUES('fact','input',1,'host','operator','2026-01-01T00:00:00Z',1)", []).unwrap();
    assert!(conn.execute("INSERT INTO source_annotation_import(annotation,input,chain,machine,actor,at,commit_seq) VALUES('absent','input',1,'host','operator','2026-01-01T00:00:00Z',1)", []).is_err());
    assert_eq!(
        conn.query_row("SELECT annotation FROM source_annotation_link", [], |row| {
            row.get::<_, String>(0)
        })
        .unwrap(),
        "fact"
    );
}
