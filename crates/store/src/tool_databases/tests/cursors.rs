use super::*;

#[test]
fn generated_columns_keep_their_values_and_the_following_columns_stay_aligned() {
    for storage in ["VIRTUAL", "STORED"] {
        for suffix in ["", " WITHOUT ROWID"] {
            let dir = tempfile::tempdir().unwrap();
            let db = dir.path().join("tool.sqlite");
            let out = dir.path().join("tool.jsonl");
            Connection::open(&db).unwrap().execute_batch(&format!("CREATE TABLE items(id INTEGER PRIMARY KEY,seed TEXT,title AS(seed || ' title') {storage},tail INTEGER){suffix}; INSERT INTO items(id,seed,tail) VALUES(7,'synthetic',42);")).unwrap();
            Exporter::default().export(&db, &out, None).unwrap();
            let row = &lines(&out)[1];
            assert_eq!(row["k"], json!([7]));
            assert_eq!(row["r"]["title"], "synthetic title");
            assert_eq!(row["r"]["tail"], 42);
        }
    }
}

fn cursor(_: &Path, table: &str) -> Option<&'static str> {
    (table == "items").then_some("id")
}

#[test]
fn an_empty_table_has_no_cursor_and_accepts_the_smallest_integer_key() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("tool.sqlite");
    let out = dir.path().join("tool.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT);")
        .unwrap();
    let mut exporter = Exporter::with_cursor(cursor);
    exporter.export(&db, &out, None).unwrap();
    source
        .execute("INSERT INTO items VALUES(?1,'first')", [i64::MIN])
        .unwrap();
    exporter.dbs.get_mut(&db).unwrap().cursor_at = Some(Instant::now() - CURSOR_EXPORT);
    let rows = exporter.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(rows.rows, 1);
    assert_eq!(lines(&out)[1]["k"], json!([i64::MIN]));
}

#[test]
fn a_replaced_source_rechecks_rows_below_the_cursor() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("tool.sqlite");
    let out = dir.path().join("tool.jsonl");
    Connection::open(&db).unwrap().execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');").unwrap();
    let mut exporter = Exporter::with_cursor(cursor);
    exporter.export(&db, &out, None).unwrap();
    let replacement = dir.path().join("replacement.sqlite");
    Connection::open(&replacement).unwrap().execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'replacement');").unwrap();
    fs::rename(replacement, &db).unwrap();
    exporter.export(&db, &out, None).unwrap();
    let rows = exporter.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(rows, Exported { rows: 1, gone: 1 });
}

#[test]
fn declared_cursors_read_inserts_and_daily_verification_finds_updates_and_deletions() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("tool.sqlite");
    let out = dir.path().join("tool.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT);
        INSERT INTO items(body) VALUES('first'),('second');",
        )
        .unwrap();
    let mut exporter = Exporter::with_cursor(cursor);
    assert_eq!(exporter.export(&db, &out, None).unwrap().unwrap().rows, 2);
    source.execute_batch("UPDATE items SET body='changed' WHERE id=1; DELETE FROM items WHERE id=2; INSERT INTO items(body) VALUES('third');").unwrap();
    let st = exporter.dbs.get_mut(&db).unwrap();
    st.cursor_at = Some(Instant::now() - CURSOR_EXPORT);
    st.last_seen = None;
    let mut visited = 0;
    let turn = exporter
        .export_with(&db, &out, None, &mut || {
            visited += 1;
            Ok(())
        })
        .unwrap()
        .unwrap();
    assert_eq!(visited, 1, "the stored rows below the cursor were not read");
    assert_eq!(turn, Exported { rows: 1, gone: 0 });
    let st = exporter.dbs.get_mut(&db).unwrap();
    st.catalog
        .as_ref()
        .unwrap()
        .conn
        .execute("UPDATE table_cursor SET verified_at=0", [])
        .unwrap();
    st.cursor_at = Some(Instant::now() - CURSOR_EXPORT);
    st.last_seen = None;
    let verified = exporter.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(verified, Exported { rows: 1, gone: 1 });
    assert!(lines(&out).iter().any(|row| row["r"]["body"] == "changed"));
}

#[test]
fn undeclared_tables_compare_updates_and_reused_keys() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("tool.sqlite");
    let out = dir.path().join("tool.jsonl");
    let source = Connection::open(&db).unwrap();
    source.execute_batch("CREATE TABLE other(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO other VALUES(1,'first'),(2,'second');").unwrap();
    let mut exporter = Exporter::with_cursor(cursor);
    exporter.export(&db, &out, None).unwrap();
    source.execute_batch("DELETE FROM other WHERE id=2; INSERT INTO other VALUES(2,'replacement'); UPDATE other SET body='updated' WHERE id=1;").unwrap();
    exporter.export(&db, &out, None).unwrap();
    assert_eq!(
        exporter.export(&db, &out, None).unwrap().unwrap(),
        Exported { rows: 2, gone: 0 }
    );
}

#[test]
fn restart_uses_the_saved_catalog_and_hash_without_reading_the_export() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("tool.sqlite");
    let out = dir.path().join("tool.jsonl");
    Connection::open(&db).unwrap().execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES(printf('%02000000d',1));").unwrap();
    Exporter::with_cursor(cursor)
        .export(&db, &out, None)
        .unwrap();
    let mut visited = 0;
    let done = Exporter::with_cursor(cursor)
        .export_with(&db, &out, None, &mut || {
            visited += 1;
            Ok(())
        })
        .unwrap()
        .unwrap();
    assert_eq!(done, Exported::default());
    assert_eq!(
        visited, 1,
        "only the source row was visited; neither prefix blocks nor export lines were read"
    );
}
