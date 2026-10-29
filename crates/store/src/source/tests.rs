use super::*;

#[test]
fn cache_schema_renames_tables_without_changing_input_columns_or_counter_keys() {
    let cache = Connection::open_in_memory().unwrap();
    cache.execute_batch(crate::schema::INDEX).unwrap();
    cache.execute_batch(&schema().unwrap().ddl).unwrap();
    cache
        .execute(
            "INSERT INTO source_input(id,format) VALUES ('i','test')",
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
            "SELECT n FROM source_row_count WHERE name='input'",
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
    let ledger = Connection::open(&path).unwrap();
    ledger.execute_batch(crate::schema::ARCHIVE).unwrap();
    ledger
        .execute("INSERT INTO input(id,format) VALUES ('i','test')", [])
        .unwrap();
    let reader = Connection::open_in_memory().unwrap();
    reader
        .execute("ATTACH DATABASE ?1 AS a", [path.to_string_lossy().as_ref()])
        .unwrap();
    views(&reader).unwrap();
    let before: i64 = reader
        .query_row(
            "SELECT n FROM source_row_count WHERE name='input'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(before, 1);
    ledger
        .execute("INSERT INTO input(id,format) VALUES ('j','test')", [])
        .unwrap();
    let actual: (i64, String) = reader
        .query_row(
            "SELECT rowid,id FROM source_input WHERE id='j'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    let expected: (i64, String) = ledger
        .query_row("SELECT rowid,id FROM input WHERE id='j'", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn primary_opens_one_source_cache_and_never_creates_a_local_input_ledger() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("polylogue");
    let sinex = directory.path().join("sinex");
    let (archive, lock) = crate::Archive::open_primary(&root, &sinex).unwrap();
    assert!(archive.index_db().exists());
    assert!(archive.embeddings_db().exists());
    for name in ["archive.db", "sinex.db", "segments", "parts"] {
        assert!(!root.join(name).exists(), "unexpected local {name}");
    }
    let reads = crate::ReadPool::open(&archive).unwrap();
    assert_eq!(reads.reader().unwrap().status().unwrap().inputs, 0);
    drop(reads);
    drop(lock);
    let (archive, _lock) = crate::Archive::open_primary(&root, &sinex).unwrap();
    assert_eq!(
        crate::ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .status()
            .unwrap()
            .stored_bytes,
        0
    );
}

#[test]
fn later_chunks_keep_their_actual_admission_when_one_revision_grows() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        if primary {
            conn.execute_batch(&schema().unwrap().ddl).unwrap();
        } else {
            conn.execute_batch(crate::schema::ARCHIVE).unwrap();
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
            input: "input".into(),
            format: "synthetic".into(),
            observation: None,
            revision: Some(crate::input_rows::Revision {
                number: 1,
                produced: "first".into(),
                chain: 1,
                length: 4,
            }),
            extent: Some(crate::input_rows::Extent {
                chain: 1,
                offset: 0,
                length: 4,
                blob: hash,
                blob_offset: 0,
            }),
        };
        row.insert(&conn, 2).unwrap();
        row.revision.as_mut().unwrap().length = 8;
        row.extent.as_mut().unwrap().offset = 4;
        row.extent.as_mut().unwrap().blob_offset = 4;
        row.insert(&conn, 9).unwrap();
        assert_eq!(
            Source::admitted_through(&conn, "input", 1, 4).unwrap(),
            Some(2)
        );
        assert_eq!(
            Source::admitted_through(&conn, "input", 1, 6).unwrap(),
            Some(9)
        );
        assert_eq!(
            Source::admitted_through(&conn, "input", 1, 9).unwrap(),
            None
        );
        row.insert(&conn, 15).unwrap();
        assert_eq!(
            Source::admitted_through(&conn, "input", 1, 8).unwrap(),
            Some(9)
        );
        conn.execute(&source.sql("DELETE FROM @source.extent WHERE offset=0"), [])
            .unwrap();
        assert_eq!(
            Source::admitted_through(&conn, "input", 1, 8).unwrap(),
            None
        );
    }
}
