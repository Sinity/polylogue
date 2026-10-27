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
    fn export(&mut self, exporter: &mut Exporter) -> Result<Option<Exported>, ExportError> {
        let archive = &self.archive;
        let blobs = std::sync::Arc::new(archive.blobs().unwrap());
        let input = &self.input;
        let observation = &self.observation;
        let writer = &self.writer;
        let mut length = self.length;
        let result = exporter.export(
            &self.db,
            archive,
            input,
            observation,
            |bytes, mut update| {
                let first = length == 0;
                length += bytes.len() as u64;
                update.length = length;
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
                    parse: vec![ParseStep::comparison(update)],
                    release: Default::default(),
                };
                for done in writer.sender()?.store_group(
                    vec![PreparedBatch { batch, files }],
                    std::sync::Arc::clone(&blobs),
                )? {
                    done?;
                }
                Ok(())
            },
        );
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
fn cursor(_: &Path, table: &str) -> Option<&'static str> {
    (table == "items").then_some("id")
}

#[test]
fn changed_deleted_and_dropped_rows_keep_their_records_without_an_export_file() {
    let mut t = Test::new(
        "CREATE TABLE threads(id TEXT PRIMARY KEY,title TEXT,data BLOB,n REAL); CREATE TABLE notes(body TEXT); INSERT INTO threads VALUES('a','first',x'00ff',1.5),('b','second',NULL,NULL); INSERT INTO notes VALUES('kept');",
    );
    let mut exporter = Exporter::default();
    assert_eq!(t.export(&mut exporter).unwrap().unwrap().rows, 3);
    let rows = t.lines();
    assert_eq!(rows[0]["source"]["machine"], "test-machine");
    assert!(
        rows.iter()
            .any(|row| row["r"]["data"]["base64"] == "AP8=" && row["r"]["n"] == 1.5)
    );
    assert!(t.export(&mut exporter).unwrap().is_none());
    t.sql("UPDATE threads SET title='renamed' WHERE id='a'; DELETE FROM threads WHERE id='b'; DROP TABLE notes;");
    assert!(t.export(&mut exporter).unwrap().is_none());
    let done = t.export(&mut exporter).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 2));
    assert_eq!(t.lines().iter().filter(|r| r["gone"] == true).count(), 2);
    assert!(!t.archive.root.join("exports").exists());
    assert!(!t.dir.path().join("source.sqlite.jsonl").exists());
}

#[test]
fn comparison_uses_complete_keys_and_generated_columns() {
    let mut t = Test::new(
        "CREATE TABLE pair(a TEXT,b INTEGER,title TEXT,PRIMARY KEY(a,b)) WITHOUT ROWID; INSERT INTO pair VALUES('x',1,'one'),('x',2,'two'); CREATE TABLE generated(id INTEGER PRIMARY KEY,n INTEGER,twice INTEGER GENERATED ALWAYS AS(n*2)); INSERT INTO generated(id,n) VALUES(1,3); CREATE TABLE shadow(rowid TEXT,value TEXT); INSERT INTO shadow VALUES('name','body');",
    );
    t.export(&mut Exporter::default()).unwrap();
    let rows = t.lines();
    assert!(rows.iter().any(|r| r["k"] == json!(["x", 2])));
    assert!(rows.iter().any(|r| r["r"]["twice"] == 6));
    assert!(
        rows.iter()
            .any(|r| r["t"] == "shadow" && r["k"] == json!([1]))
    );
}

#[test]
fn duplicate_nullable_keys_and_invalid_text_are_refused() {
    for sql in [
        "CREATE TABLE t(k TEXT PRIMARY KEY); INSERT INTO t VALUES(NULL),(NULL);",
        "CREATE TABLE t(body TEXT); INSERT INTO t VALUES(CAST(x'ff' AS TEXT));",
        "CREATE TABLE t(rowid TEXT,_rowid_ TEXT,oid TEXT); INSERT INTO t VALUES('a','b','c');",
    ] {
        let mut t = Test::new(sql);
        assert!(t.export(&mut Exporter::default()).is_err(), "{sql}");
    }
}

#[test]
fn restart_reads_only_new_rows_after_a_declared_cursor() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES(printf('%02000000d',1));",
    );
    t.export(&mut Exporter::with_cursor(cursor)).unwrap();
    let length = t.length;
    t.sql("INSERT INTO items(body) VALUES('new');");
    let done = t
        .export(&mut Exporter::with_cursor(cursor))
        .unwrap()
        .unwrap();
    assert_eq!((done.rows, done.visited), (1, 1));
    assert!(t.length - length < 200);
    let done = t
        .export(&mut Exporter::with_cursor(cursor))
        .unwrap()
        .unwrap();
    assert_eq!((done.rows, done.visited), (0, 0));
}

#[test]
fn daily_verification_finds_updates_deletions_and_reused_keys() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES('first'),('second');",
    );
    let mut exporter = Exporter::with_cursor(cursor);
    t.export(&mut exporter).unwrap();
    t.sql("UPDATE items SET body='changed' WHERE id=1; DELETE FROM items WHERE id=2; INSERT INTO items(body) VALUES('third');");
    exporter.dbs.get_mut(&t.db).unwrap().cursor_at = None;
    let done = t.export(&mut exporter).unwrap().unwrap();
    assert_eq!((done.rows, done.gone, done.visited), (1, 0, 1));
    let conn = Connection::open(t.archive.index_db()).unwrap();
    let mut saved: SavedDatabase = serde_json::from_str(
        &conn
            .query_row("SELECT state FROM tool_database", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
    )
    .unwrap();
    saved.cursors.get_mut("items").unwrap().verified_at = 0;
    conn.execute(
        "UPDATE tool_database SET state=?1",
        [serde_json::to_string(&saved).unwrap()],
    )
    .unwrap();
    exporter.dbs.get_mut(&t.db).unwrap().cursor_at = None;
    let done = t.export(&mut exporter).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (1, 1));
}

#[test]
fn replacement_schema_change_and_missing_source_verify_every_row() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');",
    );
    t.export(&mut Exporter::with_cursor(cursor)).unwrap();
    let replacement = t.dir.path().join("replacement");
    Connection::open(&replacement).unwrap().execute_batch("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'replacement');").unwrap();
    fs::rename(replacement, &t.db).unwrap();
    let done = t
        .export(&mut Exporter::with_cursor(cursor))
        .unwrap()
        .unwrap();
    assert_eq!((done.rows, done.gone), (1, 1));
    t.sql("ALTER TABLE items ADD COLUMN tail INTEGER DEFAULT 42;");
    assert_eq!(
        t.export(&mut Exporter::with_cursor(cursor))
            .unwrap()
            .unwrap()
            .rows,
        1
    );
    assert!(t.lines().iter().any(|r| r["r"]["tail"] == 42));
    let length = t.length;
    fs::remove_file(&t.db).unwrap();
    assert!(t.export(&mut Exporter::default()).unwrap().is_none());
    assert_eq!(t.length, length);
}

#[test]
fn smallest_cursor_and_empty_table_have_no_synthetic_zero() {
    let mut t = Test::new("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT);");
    let mut exporter = Exporter::with_cursor(cursor);
    t.export(&mut exporter).unwrap();
    Connection::open(&t.db)
        .unwrap()
        .execute("INSERT INTO items VALUES(?1,'first')", [i64::MIN])
        .unwrap();
    exporter.dbs.get_mut(&t.db).unwrap().cursor_at = None;
    assert_eq!(t.export(&mut exporter).unwrap().unwrap().rows, 1);
    assert!(t.lines().iter().any(|r| r["k"] == json!([i64::MIN])));
}

#[test]
fn parsing_earlier_bytes_preserves_newer_comparison_rows_and_cursors() {
    let mut t = Test::new(
        "CREATE TABLE items(id INTEGER PRIMARY KEY AUTOINCREMENT,body TEXT); INSERT INTO items(body) VALUES('first');",
    );
    t.export(&mut Exporter::with_cursor(cursor)).unwrap();
    let old: Vec<u8> = t
        .lines()
        .iter()
        .flat_map(|r| format!("{r}\n").into_bytes())
        .collect();
    let earlier = t.length;
    t.sql("UPDATE items SET body='newer'; INSERT INTO items(body) VALUES('second');");
    // A new schema forces verification below the cursor too.
    t.sql("ALTER TABLE items ADD COLUMN n INTEGER DEFAULT 42;");
    t.export(&mut Exporter::with_cursor(cursor)).unwrap();
    let conn = Connection::open(t.archive.index_db()).unwrap();
    let before: String = conn
        .query_row("SELECT state FROM tool_database", [], |r| r.get(0))
        .unwrap();
    let pk: i64 = conn
        .query_row("SELECT pk FROM input_ref WHERE id=?1", [&t.input.id], |r| {
            r.get(0)
        })
        .unwrap();
    let mut update = Comparison::records(&old, None).unwrap();
    update.length = earlier;
    update.write(&conn, pk).unwrap();
    assert_eq!(
        conn.query_row("SELECT state FROM tool_database", [], |r| r
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
        conn.query_row("SELECT state FROM tool_database", [], |r| r
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
    t.export(&mut Exporter::default()).unwrap();
    let replacement = t.dir.path().join("replacement");
    Connection::open(&replacement).unwrap().execute_batch("CREATE TABLE items(body TEXT,id INTEGER PRIMARY KEY); INSERT INTO items VALUES('same',1);").unwrap();
    fs::rename(replacement, &t.db).unwrap();
    let done = t.export(&mut Exporter::default()).unwrap().unwrap();
    assert_eq!((done.rows, done.gone), (0, 0));
    assert_eq!(t.lines().iter().filter(|r| r.get("r").is_some()).count(), 1);
}

#[test]
fn a_returned_database_verifies_rows_below_its_cursor() {
    let mut t=Test::new("CREATE TABLE items(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO items VALUES(1,'first'),(2,'second');");
    let mut exporter=Exporter::with_cursor(cursor);
    t.export(&mut exporter).unwrap();
    let away=t.dir.path().join("away.sqlite");
    fs::rename(&t.db,&away).unwrap();
    assert!(t.export(&mut exporter).unwrap().is_none());
    Connection::open(&away).unwrap().execute_batch("UPDATE items SET body='changed while absent' WHERE id=1; DELETE FROM items WHERE id=2;").unwrap();
    fs::rename(away,&t.db).unwrap();
    let done=t.export(&mut exporter).unwrap().unwrap();
    assert_eq!((done.rows,done.gone),(1,1));
}
