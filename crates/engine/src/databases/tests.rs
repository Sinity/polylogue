use super::*;

#[test]
fn completed_records_wake_the_loop_and_unchanged_sources_need_no_archive_read() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state_5.sqlite");
    rusqlite::Connection::open(&db).unwrap().execute_batch("CREATE TABLE kept(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO kept VALUES(1,'synthetic');").unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let source = || DatabaseInput {
        path: db.clone(),
        input: InputKey {
            id: "synthetic-input".into(),
            format: "codex-database".into(),
            arrival_rank: 0,
        },
        observation: Observation {
            machine: "test-machine".into(),
            host: "test".into(),
            path: db.display().to_string(),
            facts: None,
            seen_at_ms: 1,
            import: None,
            container: None,
        },
    };
    let mut databases = Databases::default();
    let (wake, changes) = channel();
    databases
        .start(
            vec![source()],
            archive.clone(),
            writer.sender().unwrap(),
            Some(wake),
        )
        .unwrap();
    assert!(matches!(changes.recv().unwrap(), LoopMsg::Changed));
    let done = databases.completed(true);
    assert!(done.changed);
    assert_eq!(done.inputs_seen, 1);
    assert!(done.bytes_stored > 0);
    assert!(done.errors.is_empty(), "{:?}", done.errors);
    let reads = ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM tool_row", &[])
            .unwrap()[0]["n"],
        0
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM tool_database", &[])
            .unwrap()[0]["n"],
        0
    );
    drop(reader);
    drop(reads);

    // The archive path cannot be opened. Unchanged file facts should finish
    // the interval before trying to open any archive reader.
    databases
        .start(
            vec![source()],
            Archive::at(&dir.path().join("absent")),
            writer.sender().unwrap(),
            None,
        )
        .unwrap();
    let done = databases.completed(true);
    assert!(!done.changed);
    assert_eq!(done.inputs_seen, 0);
    assert_eq!(done.bytes_stored, 0);
    assert!(done.errors.is_empty(), "{:?}", done.errors);
}

#[test]
fn native_capture_continues_the_input_named_by_its_source_header() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let db = home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let source = rusqlite::Connection::open(&db).unwrap();
    source.execute_batch("CREATE TABLE threads(id TEXT PRIMARY KEY,name TEXT); INSERT INTO threads VALUES('00000000-0000-4000-8000-000000000042','Synthetic title');").unwrap();
    let sql: String = source
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name='threads'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let header = serde_json::json!({"db":"state_5.sqlite","at_ms":1,"schema":{"threads":sql},"source":{"machine":"test-machine","path":db}});
    let row = serde_json::json!({"t":"threads","k":["00000000-0000-4000-8000-000000000042"],"r":{"id":"00000000-0000-4000-8000-000000000042","name":"Synthetic title"},"h":"unused"});
    let bytes = format!("{header}\n{row}\n").into_bytes();
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    {
        let writer = Writer::start(&archive).unwrap();
        writer
            .store_bytes(
                archive.prepare_bytes(bytes.as_slice()).unwrap(),
                Batch {
                    input: InputKey {
                        id: "stored-database".into(),
                        format: "codex-database".into(),
                        arrival_rank: 0,
                    },
                    change: Change::First,
                    observation: Some(Observation {
                        machine: "test-machine".into(),
                        host: "test".into(),
                        path: dir
                            .path()
                            .join("observed-records.jsonl")
                            .display()
                            .to_string(),
                        facts: None,
                        seen_at_ms: 1,
                        import: None,
                        container: None,
                    }),
                    continues: false,
                    new_blob: None,
                    blobs: vec![],
                    length: bytes.len() as u64,
                    parse: vec![],
                    release: Default::default(),
                },
            )
            .unwrap();
        writer.derive().unwrap();
    }
    let mut config = crate::test_config();
    config.home = home;
    config.tool_dirs.clear();
    config.machine = "test-machine".into();
    config.host = "test".into();
    config.workers = 1;
    let mut engine = Engine::new(config, archive).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(engine.inputs[&db].input_id, "stored-database");
    let reads = ReadPool::open(&engine.archive).unwrap();
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM a.input", &[])
            .unwrap()[0]["n"],
        1
    );
    assert_eq!(
        reader
            .input_progress("stored-database")
            .unwrap()
            .unwrap()
            .length,
        bytes.len() as u64
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM session", &[])
            .unwrap()[0]["n"],
        1
    );
}
