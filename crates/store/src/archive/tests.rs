use super::*;

#[test]
fn open_refuses_mode_mismatch_before_creating_files() {
    fn files(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut result = Vec::new();
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                result.extend(files(&entry.path()));
            } else {
                result.push((entry.path(), fs::read(entry.path()).unwrap()));
            }
        }
        result.sort();
        result
    }
    // An archive recorded as primary: this version opens it in no mode.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("meta.json"), "{\"mode\":\"primary\"}\n").unwrap();
    fs::write(root.join("index.db"), b"synthetic primary index").unwrap();
    let before = files(&root);
    let expected = format!(
        "archive {} is primary, but configuration requests standalone; select its recorded mode or another archive",
        root.display()
    );
    assert_eq!(
        Archive::open_or_create(&root).unwrap_err().to_string(),
        expected
    );
    let Err(blocked) = Archive::open_blocked_with_invalidator(&root, || Ok(())) else {
        panic!("a primary archive opened as blocked");
    };
    assert_eq!(blocked.to_string(), expected);
    assert_eq!(files(&root), before);
}

#[test]
fn a_second_open_is_refused_while_the_lock_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
    assert!(matches!(
        Archive::open_or_create(&root),
        Err(StoreError::Locked { .. })
    ));
    drop(_lock);
    Archive::open_or_create(&root).unwrap();
}

#[test]
fn a_write_connection_limits_the_log_file_it_keeps() {
    let dir = tempfile::tempdir().unwrap();
    let conn = open_write(&dir.path().join("x.db"), cache::SETUP).unwrap();
    let limit: i64 = conn
        .query_row("PRAGMA journal_size_limit", [], |row| row.get(0))
        .unwrap();
    assert_eq!(limit, 64 * 1024 * 1024);
}

#[test]
fn a_refused_open_names_the_pid_of_the_holder() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
    let Err(StoreError::Locked { holder, .. }) = Archive::open_or_create(&root) else {
        panic!("the second open must be refused");
    };
    assert_eq!(holder, Some(std::process::id()));
}

#[test]
fn an_archive_newer_or_with_unknown_tables_is_refused_and_left_as_it_is() {
    let newer = format!("PRAGMA user_version = {}", schema::ARCHIVE_VERSION + 1);
    let older = format!("PRAGMA user_version = {}", schema::ARCHIVE_VERSION - 1);
    // The last as an archive written by a build that declared the current
    // version with other tables.
    for (change, newer) in [
        (newer.as_str(), true),
        (older.as_str(), false),
        ("DROP TABLE revision", false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        drop(lock);
        let conn = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        conn.execute_batch(change).unwrap();
        conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
            .unwrap();
        drop(conn);
        let before = std::fs::read(archive.archive_db()).unwrap();
        let error = Archive::open_or_create(dir.path()).unwrap_err();
        let refused = if newer {
            matches!(
                error,
                StoreError::NewerSchema {
                    database: "archive",
                    ..
                }
            )
        } else {
            matches!(
                error,
                StoreError::OtherFormat {
                    database: "archive",
                    ..
                }
            )
        };
        assert!(refused, "{change}: {error}");
        assert_eq!(std::fs::read(archive.archive_db()).unwrap(), before);
        // The refusal released the lock: a second try gets the same answer.
        assert_eq!(
            Archive::open_or_create(dir.path()).unwrap_err().to_string(),
            error.to_string()
        );
    }
}

/// `embeddings.db` holds paid vectors: the archive's open brings one from an
/// earlier build to this build's schema in place rather than refusing it.
#[test]
fn an_archive_opens_with_an_embeddings_db_from_an_earlier_build() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    drop(lock);
    let path = archive.embeddings_db();
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
    }
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(include_str!(
        "../schema/tests/embeddings_before_revision.sql"
    ))
    .unwrap();
    conn.execute_batch(
        "PRAGMA user_version = 1;
         INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at)
           VALUES (x'01','m-a',0,2,x'0000803f000000c0',100);",
    )
    .unwrap();
    drop(conn);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_read(&archive.embeddings_db()).unwrap();
    let stored: (Vec<u8>, String) = conn
        .query_row(
            "SELECT e.vector, group_concat(m.name) FROM embedded e,
               (SELECT name FROM sqlite_master WHERE name IN ('embedding_revision','query_vector') ORDER BY name) m",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        stored,
        (
            vec![0, 0, 0x80, 0x3f, 0, 0, 0, 0xc0],
            "embedding_revision,query_vector".to_owned()
        )
    );
}

#[test]
fn an_exhausted_counter_refuses_instead_of_reusing_a_position() {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); INSERT INTO meta VALUES ('write_counter','9223372036854775807')").unwrap();
    let tx = conn.transaction().unwrap();
    assert_eq!(
        bump_counter(&tx).unwrap_err().to_string(),
        "write counter exhausted; this is a Polylogue bug; report it with the request ID"
    );
    assert_eq!(
        tx.query_row("SELECT value FROM meta", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "9223372036854775807"
    );
}

mod hash_order;

#[test]
fn a_stopped_process_resumes_its_committed_parse_positions() {
    const CHILD: &str = "POLYLOGUE_TEST_BUILD_RESUME";
    if let Some(path) = std::env::var_os(CHILD) {
        let (archive, _lock) = Archive::open_or_create(Path::new(&path)).unwrap();
        let writer = crate::Writer::start(&archive).unwrap();
        // Committed positions survive a lost process once the index is
        // installed; a build lost with its process starts over
        // (docs/architecture.md, Durability).
        writer.build(Vec::new()).unwrap();
        let output = polylogue_model::Output {
            sessions: vec![polylogue_model::Session::new(
                crate::fixture::writer::sid("resume"),
                polylogue_model::SessionKind::Main,
            )],
            messages: vec![polylogue_model::MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 0,
                end: 1,
                message: crate::fixture::writer::prose("resume", 0, 1, "kept message").remove(0),
            }],
            ..Default::default()
        };
        let mut batch =
            crate::fixture::writer::step("resume-input", "claude-code", "/synthetic", output, 0, 1);
        let mut raw = batch.clone();
        raw.parse.clear();
        let bytes = archive.prepare_bytes(&b"r\n"[..]).unwrap();
        let stored = writer.store_bytes(bytes, raw).unwrap();
        batch.change = crate::Change::Reparse {
            chain: stored.chain,
        };
        batch.observation = None;
        writer.send(batch).unwrap();
        let report = writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
        // Exit without dropping the writer, as a stopped process does.
        std::process::exit(0);
    }
    let dir = tempfile::tempdir().unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "archive::tests::a_stopped_process_resumes_its_committed_parse_positions",
            "--nocapture",
        ])
        .env(CHILD, dir.path())
        .status()
        .unwrap();
    assert!(status.success());
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let pool = crate::ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(reader.status().unwrap().messages, 1);
    let position = reader.input_progress("resume-input").unwrap().unwrap();
    assert_eq!(position.parsed_upto, 2);
    drop(reader);
    let writer = crate::Writer::start(&archive).unwrap();
    writer.derive().unwrap();
    drop(writer);
    let index = open_read(&archive.index_db()).unwrap();
    assert!(schema::same_schema(&index, schema::INDEX_SCHEMA).unwrap());
    assert!(archive.index_build().is_none());
}

#[test]
fn opening_adds_model_completion_index_without_rebuilding_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    // An installed index: a build in progress defers this index to its end.
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.execute_batch("DROP INDEX message_model_at;").unwrap();
    let completion_plan = |index: &Connection| -> Vec<String> {
        index
            .prepare("EXPLAIN QUERY PLAN SELECT DISTINCT model FROM message WHERE model IS NOT NULL ORDER BY model")
            .unwrap()
            .query_map([], |r| r.get(3))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    let before = completion_plan(&index);
    assert!(
        before.iter().any(|step| step.contains("SCAN message")),
        "{before:?}"
    );
    assert!(
        before.iter().any(|step| step.contains("TEMP B-TREE")),
        "{before:?}"
    );
    index
        .execute("INSERT INTO meta VALUES ('example', 'kept')", [])
        .unwrap();
    let generation: String = index
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
        .unwrap();
    drop(index);
    drop(lock);
    for _ in 0..2 {
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let index = open_read(&archive.index_db()).unwrap();
        assert_eq!(schema::version(&index).unwrap(), schema::INDEX_VERSION);
        for (key, expected) in [("example", "kept"), ("generation", generation.as_str())] {
            assert_eq!(
                index
                    .query_row("SELECT value FROM meta WHERE key=?1", [key], |r| r
                        .get::<_, String>(0))
                    .unwrap(),
                expected
            );
        }
        let plan = completion_plan(&index);
        assert!(
            plan.iter()
                .any(|step| step.contains("COVERING INDEX message_model_at")),
            "{plan:?}"
        );
        assert!(
            !plan
                .iter()
                .any(|step| step == "SCAN message" || step.contains("TEMP B-TREE")),
            "{plan:?}"
        );
        drop(index);
        drop(lock);
    }
}

#[test]
fn missing_time_indexes_are_created_without_rebuilding_the_index() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    // An installed index: a build in progress defers these to its end.
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index
        .execute_batch(
            "DROP INDEX message_at; DROP INDEX event_at;
             DROP INDEX message_session_at; DROP INDEX message_model_at; DROP INDEX message_session_model_at;
        INSERT INTO input_ref(id) VALUES ('synthetic');
        INSERT INTO meta VALUES ('index-marker','present');",
        )
        .unwrap();
    drop(index);
    drop(lock);
    for _ in 0..2 {
        let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
        let index = open_read(&archive.index_db()).unwrap();
        assert!(schema::same_schema(&index, schema::INDEX_SCHEMA).unwrap());
        assert_eq!(
            index
                .query_row("SELECT value FROM meta WHERE key='index-marker'", [], |r| r
                    .get::<_, String>(0))
                .unwrap(),
            "present"
        );
        assert_eq!(
            index
                .query_row("SELECT id FROM input_ref", [], |r| r.get::<_, String>(0))
                .unwrap(),
            "synthetic"
        );
        drop(index);
        drop(lock);
    }
}

#[test]
fn opening_clears_prepared_output_left_by_a_stop() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let output = archive.staging_dir().join("unfinished-output");
    std::fs::write(&output, b"synthetic prepared output").unwrap();
    drop(lock);
    let (_archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(!output.exists());
}

/// Readers that take a snapshot after each commit and finish before the
/// next keep the automatic checkpoint from ever starting the log over; the
/// writer's restart between transactions holds the log near its limit.
#[test]
fn the_log_stays_near_its_limit_while_readers_follow_every_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.db");
    let writer = open_write(&path, cache::SETUP).unwrap();
    writer
        .execute_batch("CREATE TABLE row(n INTEGER PRIMARY KEY, body BLOB)")
        .unwrap();
    let reader = open_read(&path).unwrap();
    let page: u64 = writer
        .query_row("PRAGMA page_size", [], |r| r.get::<_, i64>(0))
        .unwrap() as u64;
    let limit = 64;
    let wait = std::time::Duration::from_millis(100);
    let log = dir.path().join("log.db-wal");
    let mut largest = 0;
    let mut restarts = 0;
    for n in 0..400 {
        // A reader that started after the last commit is still reading as
        // the next transaction begins.
        reader.execute_batch("BEGIN").unwrap();
        reader
            .query_row("SELECT count(*) FROM row", [], |r| r.get::<_, i64>(0))
            .unwrap();
        writer
            .execute("INSERT INTO row(n, body) VALUES (?1, zeroblob(8192))", [n])
            .unwrap();
        reader.execute_batch("COMMIT").unwrap();
        if restart_full_log(&writer, "main", &path, limit, wait).unwrap() {
            restarts += 1;
        }
        largest = largest.max(std::fs::metadata(&log).unwrap().len());
    }
    // One commit past the limit at most: the log header, then frames of a
    // page header plus a page each.
    let frames = limit as u64 + 8;
    assert!(
        largest <= 32 + frames * (24 + page),
        "the log reached {largest} bytes"
    );
    assert!(restarts > 0);
    let rows: i64 = writer
        .query_row("SELECT count(*) FROM row", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 400);
}

/// A reader still in the log keeps it: the restart gives up after its wait
/// and the next one, after the reader finished, starts it over.
#[test]
fn a_log_in_use_is_kept_until_its_reader_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("log.db");
    let writer = open_write(&path, cache::SETUP).unwrap();
    writer
        .execute_batch("CREATE TABLE row(n INTEGER PRIMARY KEY, body BLOB)")
        .unwrap();
    writer
        .execute("INSERT INTO row(n, body) VALUES (1, zeroblob(65536))", [])
        .unwrap();
    let reader = open_read(&path).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    reader
        .query_row("SELECT count(*) FROM row", [], |r| r.get::<_, i64>(0))
        .unwrap();
    writer
        .execute("INSERT INTO row(n, body) VALUES (2, zeroblob(65536))", [])
        .unwrap();
    let log = dir.path().join("log.db-wal");
    let before = std::fs::metadata(&log).unwrap().len();
    let wait = std::time::Duration::from_millis(50);
    assert!(!restart_full_log(&writer, "main", &path, 4, wait).unwrap());
    assert_eq!(std::fs::metadata(&log).unwrap().len(), before);
    reader.execute_batch("COMMIT").unwrap();
    assert!(restart_full_log(&writer, "main", &path, 4, wait).unwrap());
    assert_eq!(std::fs::metadata(&log).unwrap().len(), 0);
    let rows: i64 = reader
        .query_row("SELECT count(*) FROM row", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 2);
}

#[test]
fn runtime_connections_leave_committed_wal_for_reopen_while_setup_finishes_it() {
    for reader_last in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wal.db");
        let log = dir.path().join("wal.db-wal");
        let writer = open_runtime_writer(&path, cache::SETUP).unwrap();
        writer.pragma_update(None, "synchronous", "full").unwrap();
        writer
            .execute_batch("CREATE TABLE durable(value INTEGER); INSERT INTO durable VALUES(42)")
            .unwrap();
        let before = fs::read(&log).unwrap();
        assert!(!before.is_empty());
        let reader = reader_last.then(|| open_read(&path).unwrap());
        if let Some(reader) = &reader {
            assert_eq!(
                reader
                    .query_row("SELECT value FROM durable", [], |row| row.get::<_, i64>(0))
                    .unwrap(),
                42
            );
        }
        drop(writer);
        drop(reader);
        assert_eq!(fs::read(&log).unwrap(), before);
        let reader = open_read(&path).unwrap();
        assert_eq!(
            reader
                .query_row("SELECT value FROM durable", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            42
        );
        drop(reader);
        assert_eq!(fs::read(&log).unwrap(), before);
        let setup = open_write(&path, cache::SETUP).unwrap();
        assert_eq!(
            setup
                .query_row("SELECT value FROM durable", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            42
        );
        drop(setup);
        match fs::metadata(&log) {
            Ok(metadata) => assert_eq!(metadata.len(), 0),
            Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::NotFound),
        }
        assert_eq!(
            open_read(&path)
                .unwrap()
                .query_row("SELECT value FROM durable", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            42
        );
    }
}

/// A real unclosed transaction leaves a hot journal. Native recovery may
/// restore the owned private index, while application readers stay read-only.
#[test]
fn private_index_readers_recover_journals_without_accepting_writes() {
    const CHILD: &str = "POLYLOGUE_TEST_HOT_JOURNAL";
    if let Some(path) = std::env::var_os(CHILD) {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=PERSIST; PRAGMA synchronous=FULL;
            PRAGMA cache_size=5; BEGIN IMMEDIATE;
            UPDATE recovery_probe SET value=1, payload=zeroblob(8192)",
        )
        .unwrap();
        // Deliberately skip SQLite's destructor and rollback in this subprocess.
        std::process::exit(0);
    }
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = due_a_build(&dir.path().join("archive"));
    let path = archive.index_db();
    {
        let conn = index_build::open_writer(&archive).unwrap();
        conn.execute_batch(
            "CREATE TABLE recovery_probe(value INTEGER, payload BLOB);
            WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<100)
            INSERT INTO recovery_probe SELECT 0,zeroblob(8192) FROM n",
        )
        .unwrap();
    }
    let provider = dir.path().join("provider.sqlite");
    Connection::open(&provider)
        .unwrap()
        .execute_batch("CREATE TABLE source(value TEXT); INSERT INTO source VALUES('sentinel')")
        .unwrap();
    let original_provider = fs::read(&provider).unwrap();
    for reader_kind in ["direct", "pool", "tool database"] {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "archive::tests::private_index_readers_recover_journals_without_accepting_writes",
                "--nocapture",
            ])
            .env(CHILD, &path)
            .status()
            .unwrap();
        assert!(status.success());
        // The normal opener's schema-dependent cache pragma can itself meet
        // the hot journal. A bare READONLY open isolates its SELECT refusal.
        let original =
            Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let error = original
            .query_row("SELECT sum(value) FROM recovery_probe", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap_err();
        assert_eq!(error.sqlite_error().unwrap().extended_code, 776);
        drop(original);
        if reader_kind == "pool" {
            let pool = crate::ReadPool::open(&archive).unwrap();
            let reader = pool.reader().unwrap();
            assert_eq!(
                reader
                    .query("SELECT sum(value) AS value FROM recovery_probe", &[])
                    .unwrap()[0]["value"],
                0
            );
            assert!(
                reader
                    .query("UPDATE recovery_probe SET value=2 RETURNING value", &[])
                    .is_err()
            );
            assert!(
                reader
                    .query_each(
                        "UPDATE recovery_probe SET value=2 RETURNING value",
                        &[],
                        |_| Ok::<_, StoreError>(())
                    )
                    .is_err()
            );
            assert!(
                reader
                    .query_counts(
                        "UPDATE recovery_probe SET value=2 RETURNING value,value",
                        &[]
                    )
                    .is_err()
            );
            assert_eq!(
                reader
                    .query("SELECT sum(value) AS value FROM recovery_probe", &[])
                    .unwrap()[0]["value"],
                0
            );
        } else if reader_kind == "tool database" {
            let input = crate::InputKey {
                id: "synthetic-db".into(),
                format: "codex-database".into(),
                route: polylogue_model::Route::Live,
            };
            let observation = crate::Observation {
                machine: "test-machine".into(),
                host: "test".into(),
                path: provider.display().to_string(),
                path_bytes: provider.as_os_str().as_encoded_bytes().to_vec(),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
                database: None,
            };
            let mut output = Vec::new();
            let report = crate::tool_databases::DatabaseReader::default()
                .capture(&provider, &archive, &input, &observation, |bytes, _| {
                    output.extend(bytes);
                    Ok(())
                })
                .unwrap()
                .unwrap();
            assert_eq!(report.changed, 1);
            assert!(String::from_utf8(output).unwrap().contains("sentinel"));
            assert_eq!(fs::read(&provider).unwrap(), original_provider);
        } else {
            let reader = archive.open_index_read().unwrap();
            assert_eq!(
                reader
                    .query_row("SELECT sum(value) FROM recovery_probe", [], |row| row
                        .get::<_, i64>(0))
                    .unwrap(),
                0
            );
            assert!(
                reader
                    .execute("UPDATE recovery_probe SET value=2", [])
                    .is_err()
            );
        }
    }
    crate::Writer::start(&archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
    assert!(archive.index_build().is_none());
    assert!(
        archive
            .open_index_read()
            .unwrap()
            .is_readonly(rusqlite::MAIN_DB)
            .unwrap()
    );
    assert!(
        crate::ReadPool::serving(&archive, 1, 8 << 20)
            .unwrap()
            .reader()
            .unwrap()
            .conn()
            .unwrap()
            .is_readonly(rusqlite::MAIN_DB)
            .unwrap()
    );
}

/// An archive whose register records an input and whose index file is
/// gone, opened so that a build is due.
fn due_a_build(root: &std::path::Path) -> (Archive, crate::ArchiveLock) {
    {
        let (archive, _lock) = Archive::open_or_create(root).unwrap();
        open_write(&archive.archive_db(), cache::SETUP)
            .unwrap()
            .execute(
                "INSERT INTO input(id,origin) VALUES ('synthetic','claude-code')",
                [],
            )
            .unwrap();
    }
    for suffix in ["", "-wal", "-shm"] {
        let _ = fs::remove_file(root.join(format!("index.db{suffix}")));
    }
    let (archive, lock) = Archive::open_or_create(root).unwrap();
    assert!(archive.index_build().is_some());
    (archive, lock)
}
