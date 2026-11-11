use super::*;

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
fn a_same_boot_build_keeps_its_rows_and_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    writer.flush().unwrap();
    drop(writer);
    // A stop in the middle of a first build leaves the mark set.
    open_write(&archive.index_db(), cache::SETUP)
        .unwrap()
        .execute("UPDATE meta SET value = '1' WHERE key = 'building'", [])
        .unwrap();
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    assert!(schema::building(&open_read(&archive.index_db()).unwrap()).unwrap());
    let writer = crate::Writer::start(&archive).unwrap();
    writer.derive().unwrap();
    assert!(!schema::building(&open_read(&archive.index_db()).unwrap()).unwrap());
}

/// Reopening an index whose first build is in progress leaves the indexes
/// that build defers to its end; building them at open would read whole
/// tables at every restart.
#[test]
fn reopening_a_first_build_does_not_build_its_deferred_indexes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("a");
    let (archive, lock) = Archive::open_or_create(&root).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    assert!(schema::building(&index).unwrap());
    // As the writer leaves them while the build runs.
    for name in crate::writer::deferred_index_names() {
        index
            .execute_batch(&format!("DROP INDEX IF EXISTS {name}"))
            .unwrap();
    }
    // A read index the build does not defer, missing, is completed at open.
    index
        .execute_batch("DROP INDEX block_original_ordinal")
        .unwrap();
    drop(index);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let index = open_read(&archive.index_db()).unwrap();
    let present: Vec<String> = index
        .prepare("SELECT name FROM sqlite_master WHERE type='index'")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let rebuilt: Vec<&str> = crate::writer::deferred_index_names()
        .filter(|name| present.iter().any(|p| p == name))
        .collect();
    assert_eq!(rebuilt, Vec::<&str>::new());
    assert!(present.iter().any(|p| p == "block_original_ordinal"));
}

#[test]
fn an_archive_newer_or_with_unknown_tables_is_refused_and_left_as_it_is() {
    let newer = format!("PRAGMA user_version = {}", schema::ARCHIVE_VERSION + 1);
    // The second as an archive written by a build that declared the current
    // version with other tables.
    for (change, newer) in [(newer.as_str(), true), ("DROP TABLE revision", false)] {
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
                StoreError::NoMigration {
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
    assert!(
        bump_counter(&tx)
            .unwrap_err()
            .to_string()
            .contains("counter exhausted")
    );
    assert_eq!(
        tx.query_row("SELECT value FROM meta", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "9223372036854775807"
    );
}

mod hash_order;

#[test]
fn a_build_from_another_boot_resumes_rows_with_a_new_generation() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = crate::Writer::start(&archive).unwrap();
    drop(writer);
    let conn = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let before: String = conn
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
        .unwrap();
    conn.execute_batch("UPDATE meta SET value='1' WHERE key='building'; UPDATE meta SET value='another-boot' WHERE key='boot_id'; INSERT INTO meta VALUES ('kept-marker','present')").unwrap();
    drop(conn);
    drop(lock);
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_read(&archive.index_db()).unwrap();
    assert_eq!(
        conn.query_row("SELECT value FROM meta WHERE key='kept-marker'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap(),
        "present"
    );
    let after: String = conn
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_ne!(after, before);
    assert!(schema::building(&conn).unwrap());
    drop(conn);
    let writer = crate::Writer::start(&archive).unwrap();
    writer.derive().unwrap();
    drop(writer);
    drop(lock);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let conn = open_read(&archive.index_db()).unwrap();
    assert_eq!(
        conn.query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap(),
        after
    );
    assert!(!schema::building(&conn).unwrap());
}

#[test]
fn a_stopped_process_resumes_its_committed_parse_positions() {
    const CHILD: &str = "POLYLOGUE_TEST_BUILD_RESUME";
    if let Some(path) = std::env::var_os(CHILD) {
        let (archive, _lock) = Archive::open_or_create(Path::new(&path)).unwrap();
        let writer = crate::Writer::start(&archive).unwrap();
        let output = polylogue_model::Output {
            sessions: vec![polylogue_model::Session::new(
                crate::fixture::writer::sid("resume"),
                polylogue_model::SessionKind::Main,
            )],
            messages: vec![polylogue_model::MessageRow {
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
    assert!(
        schema::same_schema(
            &index,
            &[
                schema::INDEX,
                crate::coverage::SCHEMA,
                crate::variants::SCHEMA,
                crate::parser_context::SCHEMA,
                schema::READ_INDEXES
            ]
        )
        .unwrap()
    );
    assert!(!schema::building(&index).unwrap());
}

#[test]
fn opening_adds_model_completion_index_without_rebuilding_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    // A finished index: a first build in progress defers this index to its end.
    index
        .execute_batch(
            "UPDATE meta SET value='0' WHERE key='building'; DROP INDEX message_model_at;",
        )
        .unwrap();
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
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    // A finished index: a first build in progress defers these to its end.
    index
        .execute_batch(
            "UPDATE meta SET value='0' WHERE key='building';
             DROP INDEX message_at; DROP INDEX event_at;
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
        assert!(
            schema::same_schema(
                &index,
                &[
                    schema::INDEX,
                    crate::coverage::SCHEMA,
                    crate::variants::SCHEMA,
                    crate::parser_context::SCHEMA,
                    schema::READ_INDEXES
                ]
            )
            .unwrap()
        );
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
        if restart_full_log(&writer, &path, limit, wait).unwrap() {
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
    assert!(!restart_full_log(&writer, &path, 4, wait).unwrap());
    assert_eq!(std::fs::metadata(&log).unwrap().len(), before);
    reader.execute_batch("COMMIT").unwrap();
    assert!(restart_full_log(&writer, &path, 4, wait).unwrap());
    assert_eq!(std::fs::metadata(&log).unwrap().len(), 0);
    let rows: i64 = reader
        .query_row("SELECT count(*) FROM row", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 2);
}
