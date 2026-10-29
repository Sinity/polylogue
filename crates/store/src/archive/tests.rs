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

#[test]
fn an_incompatible_index_is_refused_without_removing_its_rows() {
    for change in [
        "PRAGMA user_version = 26".to_owned(),
        "CREATE TABLE synthetic_older(value)".to_owned(),
        "UPDATE meta SET value='1' WHERE key='building'; CREATE TABLE synthetic_older(value)"
            .to_owned(),
    ] {
        refused_index_keeps_the_archive(&change);
    }
}

fn refused_index_keeps_the_archive(change: &str) {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index.execute_batch(change).unwrap();
    index
        .execute(
            "INSERT INTO meta VALUES ('previous-index-marker', 'present')",
            [],
        )
        .unwrap();
    drop(index);
    let archive_conn = open_archive_writer(&archive).unwrap();
    let value = "archive note value independent of row dictionary";
    archive_conn.execute("INSERT INTO note(id, first_note,kind,value,author,machine,created_at,commit_seq) VALUES ('note','note','note',?1,'operator','synthetic',1,1)", [&value]).unwrap();
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
    let error = Archive::open_or_create(dir.path()).unwrap_err();
    assert!(
        matches!(
            error,
            StoreError::IndexVersion { .. }
                | StoreError::NoMigration {
                    database: "index",
                    ..
                }
        ),
        "{error}"
    );
    let index = open_read(&archive.index_db()).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT value FROM meta WHERE key='previous-index-marker'",
                [],
                |r| r.get::<_, String>(0)
            )
            .unwrap(),
        "present"
    );
    let archive_conn = open_read(&archive.archive_db()).unwrap();
    assert_eq!(
        archive_conn
            .query_row("SELECT value FROM note WHERE id='note'", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        value
    );
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
                offset: 0,
                end: 1,
                message: crate::fixture::writer::prose("resume", 0, 1, "kept message").remove(0),
            }],
            ..Default::default()
        };
        writer
            .send(crate::fixture::writer::step(
                "resume-input",
                "claude-code",
                "/synthetic",
                output,
                0,
                1,
            ))
            .unwrap();
        assert!(writer.flush().unwrap().errors.is_empty());
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
    assert!(schema::same_schema(&index, &[schema::INDEX, schema::READ_INDEXES]).unwrap());
    assert!(!schema::building(&index).unwrap());
}

#[test]
fn opening_adds_model_completion_index_without_rebuilding_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let index = open_write(&archive.index_db(), cache::SETUP).unwrap();
    index
        .execute_batch("DROP INDEX message_model; DROP INDEX message_model_at;")
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
                .any(|step| step.contains("COVERING INDEX message_model")),
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
        assert!(schema::same_schema(&index, &[schema::INDEX, schema::READ_INDEXES]).unwrap());
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
