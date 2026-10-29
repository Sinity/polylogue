use super::*;
use crate::{Batch, Change, ParseStep, ReadPool, Writer, staging::PreparedBatch};

struct Test {
    dir: tempfile::TempDir,
    archive: Archive,
    _lock: crate::ArchiveLock,
    writer: Writer,
    db: PathBuf,
    input: InputKey,
    observation: Observation,
    length: u64,
}
impl Test {
    fn new(sql: &str) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("source.sqlite");
        Connection::open(&db).unwrap().execute_batch(sql).unwrap();
        let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
        let writer = Writer::start(&archive).unwrap();
        Self {
            observation: Observation {
                machine: "test-machine".into(),
                host: "test".into(),
                path: db.display().to_string(),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
            },
            dir,
            archive,
            _lock: lock,
            writer,
            db,
            input: InputKey {
                id: "synthetic-db".into(),
                format: "codex-database".into(),
                arrival_rank: 0,
            },
            length: 0,
        }
    }
    fn capture(
        &mut self,
        reader: &mut DatabaseReader,
    ) -> Result<Option<ReadReport>, DatabaseError> {
        let archive = &self.archive;
        let blobs = std::sync::Arc::new(archive.blobs().unwrap());
        let input = &self.input;
        let observation = &self.observation;
        let writer = &self.writer;
        let mut length = self.length;
        let result = reader.capture(&self.db, archive, input, observation, |bytes| {
            let first = length == 0;
            let from = length;
            length += bytes.len() as u64;
            let mut parse = ParseStep::new(
                polylogue_model::Output::default(),
                polylogue_model::RecordCounts::default(),
                from..length,
                true,
            );
            parse.compare_records(&bytes)?;
            let (new_blob, files) = if bytes.is_empty() {
                (None, vec![])
            } else {
                let (blob, file) = archive.prepare_blob(&bytes)?;
                (Some(blob), file.into_iter().collect())
            };
            let batch = Batch {
                input: input.clone(),
                change: if first {
                    Change::First
                } else if new_blob.is_some() {
                    Change::Appended {
                        from: length - bytes.len() as u64,
                    }
                } else {
                    Change::Unchanged
                },
                observation: Some(observation.clone()),
                continues: false,
                new_blob,
                blobs: vec![],
                length,
                parse: vec![parse],
                release: Default::default(),
            };
            for done in writer.sender()?.store_group(
                vec![PreparedBatch { batch, files }],
                std::sync::Arc::clone(&blobs),
            )? {
                done?;
            }
            Ok(())
        });
        self.length = length;
        result
    }
    fn lines(&self) -> Vec<Value> {
        let pool = ReadPool::open(&self.archive).unwrap();
        let reader = pool.reader().unwrap();
        let p = reader.input_progress(&self.input.id).unwrap().unwrap();
        let bytes = reader
            .stored_chain(&self.input.id, p.chain, p.length)
            .unwrap()
            .read_range(&self.archive.blobs().unwrap(), 0, p.length)
            .unwrap();
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
    fn sql(&self, sql: &str) {
        Connection::open(&self.db)
            .unwrap()
            .execute_batch(sql)
            .unwrap();
    }
}
#[test]
fn changed_deleted_and_dropped_rows_keep_their_records_without_an_export_file() {
    let mut t = Test::new(
        "CREATE TABLE threads(id TEXT PRIMARY KEY,title TEXT,data BLOB,n REAL); CREATE TABLE notes(body TEXT); INSERT INTO threads VALUES('a','first',x'00ff',1.5),('b','second',NULL,NULL); INSERT INTO notes VALUES('kept');",
    );
    let mut reader = DatabaseReader::default();
    assert_eq!(t.capture(&mut reader).unwrap().unwrap().rows, 3);
    let rows = t.lines();
    assert_eq!(rows[0]["source"]["machine"], "test-machine");
    assert!(
        rows.iter()
            .any(|row| row["r"]["data"]["base64"] == "AP8=" && row["r"]["n"] == 1.5)
    );
    assert!(t.capture(&mut reader).unwrap().is_none());
    t.sql("UPDATE threads SET title='renamed' WHERE id='a'; DELETE FROM threads WHERE id='b'; DROP TABLE notes;");
    assert!(t.capture(&mut reader).unwrap().is_none());
    let done = t.capture(&mut reader).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 2));
    assert_eq!(t.lines().iter().filter(|r| r["gone"] == true).count(), 2);
    assert!(!t.archive.root.join("exports").exists());
    assert!(!t.dir.path().join("source.sqlite.jsonl").exists());
}

#[test]
fn comparison_uses_complete_keys_and_generated_columns() {
    for storage in ["VIRTUAL", "STORED"] {
        for suffix in ["", " WITHOUT /* table option */ ROWID"] {
            let mut t = Test::new(&format!(
                "CREATE TABLE pair(a TEXT,b INTEGER,title TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID; INSERT INTO pair VALUES('x',1,'one'),('x',2,'two'); CREATE TABLE generated(id INTEGER PRIMARY KEY,n INTEGER,twice INTEGER GENERATED ALWAYS AS(n*2) {storage}){suffix}; INSERT INTO generated(id,n) VALUES(1,3); CREATE TABLE shadow(ROWID TEXT,value TEXT DEFAULT 'WITHOUT ROWID'); INSERT INTO shadow VALUES('name','body'); CREATE TABLE sqlitex_kept(value TEXT); INSERT INTO sqlitex_kept VALUES('captured');",
            ));
            t.capture(&mut DatabaseReader::default()).unwrap();
            let rows = t.lines();
            assert!(rows.iter().any(|r| r["k"] == json!(["x", 2])));
            assert!(rows.iter().any(|r| r["r"]["twice"] == 6));
            assert!(
                rows.iter()
                    .any(|r| r["t"] == "shadow" && r["k"] == json!([1]))
            );
            assert!(
                rows.iter()
                    .any(|r| r["t"] == "sqlitex_kept" && r["r"]["value"] == "captured")
            );
        }
    }
}

#[test]
fn nullable_keys_and_every_sqlite_storage_class_keep_their_values() {
    let mut t = Test::new(
        "CREATE TABLE t(k TEXT PRIMARY KEY,body,acquired AS(body)); INSERT INTO t VALUES(NULL,CAST(x'ff00' AS TEXT)),(NULL,1e999),(NULL,-1e999),(NULL,x'ff00'),(NULL,1),(NULL,1.0),(NULL,NULL);",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let rows: Vec<_> = t.lines().into_iter().filter(|r| r["t"] == "t").collect();
    assert_eq!(rows.len(), 7);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["k"], json!({"rowid":i + 1}));
        assert_eq!(row["r"]["body"], row["r"]["acquired"]);
        assert!(row["at_ms"].as_i64().is_some());
        assert_eq!(row["encoding"], "sqlite");
    }
    assert_eq!(rows[0]["r"]["body"], json!({"text_base64":"/wA="}));
    assert_eq!(
        rows[1]["r"]["body"],
        json!({"real_bits":"7ff0000000000000"})
    );
    assert_eq!(
        rows[2]["r"]["body"],
        json!({"real_bits":"fff0000000000000"})
    );
    assert_eq!(rows[3]["r"]["body"], json!({"base64":"/wA="}));
    assert!(rows[4]["r"]["body"].is_i64());
    assert!(rows[5]["r"]["body"].is_f64());
    assert!(rows[6]["r"]["body"].is_null());
}

#[test]
fn rows_shadowing_every_rowid_alias_keep_their_observed_positions() {
    let mut t = Test::new(
        "CREATE TABLE t(rowid TEXT,_rowid_ TEXT,oid TEXT); INSERT INTO t VALUES('a','b','c'),('a','b','c');",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let rows: Vec<_> = t.lines().into_iter().filter(|r| r["t"] == "t").collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["k"], json!({"position":1}));
    assert_eq!(rows[1]["k"], json!({"position":2}));
    assert_eq!(rows[0]["r"], rows[1]["r"]);
}

#[test]
fn restart_compares_source_rows_without_reading_stored_history() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES(printf('%02000000d',1));",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let length = t.length;
    t.sql("INSERT INTO items(body) VALUES('new');");
    let done = t.capture(&mut DatabaseReader::default()).unwrap().unwrap();
    assert_eq!((done.rows, done.visited), (1, 2));
    assert!(t.length - length < 200);
    let done = t.capture(&mut DatabaseReader::default()).unwrap().unwrap();
    assert_eq!((done.rows, done.visited), (0, 2));
}

#[test]
fn a_changed_source_finds_updates_deletions_and_reused_keys() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');",
    );
    let mut reader = DatabaseReader::default();
    t.capture(&mut reader).unwrap();
    t.sql("UPDATE items SET body='changed' WHERE id=1; DELETE FROM items WHERE id=2; INSERT INTO items VALUES(3,'third');");
    assert!(t.capture(&mut reader).unwrap().is_none());
    let done = t.capture(&mut reader).unwrap().unwrap();
    assert_eq!((done.rows, done.gone, done.visited), (2, 1, 2));
    t.sql("DELETE FROM items WHERE id=1; INSERT INTO items VALUES(1,'reused');");
    assert!(t.capture(&mut reader).unwrap().is_none());
    let done = t.capture(&mut reader).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 0));
    assert!(t.lines().iter().any(|row| row["r"]["body"] == "reused"));
}

#[test]
fn replacement_schema_change_and_missing_source_compare_every_row() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let replacement = t.dir.path().join("replacement");
    Connection::open(&replacement).unwrap().execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'replacement');").unwrap();
    fs::rename(replacement, &t.db).unwrap();
    let done = t.capture(&mut DatabaseReader::default()).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 1));
    t.sql("ALTER TABLE items ADD COLUMN tail INTEGER DEFAULT 42;");
    assert_eq!(
        t.capture(&mut DatabaseReader::default())
            .unwrap()
            .unwrap()
            .rows,
        1
    );
    assert!(t.lines().iter().any(|r| r["r"]["tail"] == 42));
    let length = t.length;
    fs::remove_file(&t.db).unwrap();
    assert!(t.capture(&mut DatabaseReader::default()).unwrap().is_none());
    assert_eq!(t.length, length);
}

#[test]
fn an_empty_table_accepts_the_smallest_integer_key() {
    let mut t = Test::new("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT);");
    let mut reader = DatabaseReader::default();
    t.capture(&mut reader).unwrap();
    Connection::open(&t.db)
        .unwrap()
        .execute("INSERT INTO items VALUES(?1,'first')", [i64::MIN])
        .unwrap();
    assert!(t.capture(&mut reader).unwrap().is_none());
    assert_eq!(t.capture(&mut reader).unwrap().unwrap().rows, 1);
    assert!(t.lines().iter().any(|r| r["k"] == json!([i64::MIN])));
}

#[test]
fn parsing_earlier_bytes_preserves_newer_comparison_rows_and_schema() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES('first');",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let old: Vec<u8> = t
        .lines()
        .iter()
        .flat_map(|r| format!("{r}\n").into_bytes())
        .collect();
    let earlier = t.length;
    t.sql("UPDATE items SET body='newer'; INSERT INTO items(body) VALUES('second');");
    t.sql("ALTER TABLE items ADD COLUMN n INTEGER DEFAULT 42;");
    t.capture(&mut DatabaseReader::default()).unwrap();
    let conn = Connection::open(t.archive.index_db()).unwrap();
    let before: String = conn
        .query_row("SELECT schema FROM tool_database", [], |r| r.get(0))
        .unwrap();
    let pk: i64 = conn
        .query_row("SELECT pk FROM input_ref WHERE id=?1", [&t.input.id], |r| {
            r.get(0)
        })
        .unwrap();
    let mut update = Comparison::records(&old).unwrap();
    update.length = earlier;
    update.write(&conn, pk).unwrap();
    assert_eq!(
        conn.query_row("SELECT schema FROM tool_database", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        before
    );
    let hash: Vec<u8> = conn
        .query_row("SELECT hash FROM tool_row WHERE key='[1]'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        hash,
        blake3::hash(br#"{"body":"newer","id":1,"n":42}"#).as_bytes()
    );
    update.length = t.length;
    update.write(&conn, pk).unwrap();
    assert_eq!(
        conn.query_row("SELECT schema FROM tool_database", [], |r| r
            .get::<_, String>(0))
            .unwrap(),
        before
    );
}

#[test]
fn a_column_order_change_does_not_repeat_unchanged_named_values() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'same');",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let replacement = t.dir.path().join("replacement");
    Connection::open(&replacement).unwrap().execute_batch("CREATE TABLE items(body TEXT,id INTEGER PRIMARY KEY); INSERT INTO items VALUES('same',1);").unwrap();
    fs::rename(replacement, &t.db).unwrap();
    let done = t.capture(&mut DatabaseReader::default()).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (0, 0));
    assert_eq!(t.lines().iter().filter(|r| r.get("r").is_some()).count(), 1);
}

#[test]
fn a_returned_database_compares_updates_and_deletions() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');",
    );
    let mut reader = DatabaseReader::default();
    t.capture(&mut reader).unwrap();
    let away = t.dir.path().join("away.sqlite");
    fs::rename(&t.db, &away).unwrap();
    assert!(t.capture(&mut reader).unwrap().is_none());
    Connection::open(&away).unwrap().execute_batch("UPDATE items SET body='changed while absent' WHERE id=1; DELETE FROM items WHERE id=2;").unwrap();
    fs::rename(away, &t.db).unwrap();
    let done = t.capture(&mut reader).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 1));
}

#[test]
fn a_stopped_read_needs_no_archive_and_can_be_retried() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'kept');",
    );
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let mut reader = DatabaseReader::default();
    reader.stop_when(std::sync::Arc::clone(&stop));
    let result = reader.capture(
        &t.db,
        &Archive::at(&t.dir.path().join("absent")),
        &t.input,
        &t.observation,
        |_| panic!("stopped read sent bytes"),
    );
    assert!(matches!(result, Err(DatabaseError::Stopped)));
    stop.store(false, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(t.capture(&mut reader).unwrap().unwrap().rows, 1);
}

#[test]
fn unchanged_sources_do_not_need_a_scheduled_read() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'same');",
    );
    let mut reader = DatabaseReader::default();
    t.capture(&mut reader).unwrap();
    reader.dbs.get_mut(&t.db).unwrap().read_at = Some(Instant::now() - BUSY_READ);
    assert!(reader.observe(&t.db).is_none());
}

#[test]
fn an_invalid_comparison_position_does_not_change_row_hashes() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'same');",
    );
    t.capture(&mut DatabaseReader::default()).unwrap();
    let conn = Connection::open(t.archive.index_db()).unwrap();
    let (pk, before): (i64, Vec<u8>) = conn
        .query_row(
            "SELECT input_pk,hash FROM tool_row WHERE tbl='items'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    conn.execute("UPDATE tool_database SET length=-1", [])
        .unwrap();
    let mut update = Comparison::records(
        br#"{"t":"items","k":[1],"r":{"id":1,"body":"changed"}}
"#,
    )
    .unwrap();
    update.from = t.length;
    update.length = t.length + 1;
    assert!(
        matches!(update.write(&conn,pk), Err(crate::StoreError::Invalid(message)) if message.contains("negative"))
    );
    let after: Vec<u8> = conn
        .query_row("SELECT hash FROM tool_row WHERE tbl='items'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(before, after);
}
