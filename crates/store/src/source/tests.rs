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
