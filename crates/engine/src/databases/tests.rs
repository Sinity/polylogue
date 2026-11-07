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
        acquisition: crate::acquisition::AcquisitionContext {
            source: polylogue_store::primary::SourceSelector {
                origin: "codex-database".into(),
                machine: "test-machine".into(),
                path_bytes: db.as_os_str().as_encoded_bytes().to_vec(),
            },
            route: polylogue_store::primary::AcquisitionRoute::NativeCapture,
            basis: None,
            facts: None,
            raw_predecessor: None,
            required_disposition: None,
            clock: polylogue_model::WallClock::fixed(jiff::Timestamp::from_millisecond(1).unwrap()),
        },
        observation: Observation {
            machine: "test-machine".into(),
            host: "test".into(),
            path: db.display().to_string(),
            path_bytes: db.as_os_str().as_encoded_bytes().to_vec(),
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
fn native_capture_continues_the_input_at_the_observed_byte_path() {
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
    let header = serde_json::json!({"db":"state_5.sqlite","at_ms":1,"schema":{"threads":sql},"source":{"machine":"test-machine","path":"/synthetic/metadata-only/state_5.sqlite"}});
    let row = serde_json::json!({"t":"threads","k":["00000000-0000-4000-8000-000000000042"],"r":{"id":"00000000-0000-4000-8000-000000000042","name":"Synthetic title"},"h":"unused"});
    let bytes = format!("{header}\n{row}\n").into_bytes();
    assert_eq!(bytes.len(), 343);
    let (archive, _lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let initial_chain = {
        let writer = Writer::start(&archive).unwrap();
        let stored = writer
            .store_bytes(
                archive.prepare_bytes(bytes.as_slice()).unwrap(),
                Batch {
                    required_disposition: None,
                    input: InputKey {
                        id: "stored-database".into(),
                        format: "codex-database".into(),
                        arrival_rank: 0,
                    },
                    change: Change::First,
                    observation: Some(Observation {
                        machine: "test-machine".into(),
                        host: "test".into(),
                        path: db.display().to_string(),
                        path_bytes: db.as_os_str().as_encoded_bytes().to_vec(),
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
                    acquisition: None,
                    basis: None,
                    release: Default::default(),
                },
            )
            .unwrap();
        writer.derive().unwrap();
        stored.chain
    };
    let mut config = crate::test_config();
    config.home = home;
    config.tool_dirs.clear();
    config.machine = "test-machine".into();
    config.host = "test".into();
    config.workers = 1;
    let expected_at = config.clock.now().as_millisecond();
    let expected_header = serde_json::json!({
        "db":"state_5.sqlite", "at_ms":expected_at,
        "schema":{"threads":sql},
        "table_metadata":{"threads":{
            "columns":[
                {"cid":0,"name":"id","declared_type":"TEXT","not_null":false,"primary_key":1,"hidden":0},
                {"cid":1,"name":"name","declared_type":"TEXT","not_null":false,"primary_key":0,"hidden":0}
            ], "without_rowid":false
        }},
        "source":{"machine":"test-machine","path":db.to_string_lossy()}
    });
    let expected_suffix_length = serde_json::to_vec(&expected_header).unwrap().len() + 1;
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
    let progress = reader.input_progress("stored-database").unwrap().unwrap();
    assert_eq!(progress.chain, initial_chain);
    assert_eq!(
        progress.length,
        (bytes.len() + expected_suffix_length) as u64
    );
    let captured = reader
        .stored_range(
            "stored-database",
            progress.chain,
            0,
            progress.length,
            &engine.blobs,
        )
        .unwrap();
    // Historical SQL-only metadata receives one current provider header.
    // Its original source-path string remains data, not an input selector.
    assert_eq!(&captured[..bytes.len()], bytes.as_slice());
    let suffix = &captured[bytes.len()..];
    assert_eq!(suffix.len(), expected_suffix_length);
    assert_eq!(suffix.last(), Some(&b'\n'));
    assert_eq!(suffix.iter().filter(|&&byte| byte == b'\n').count(), 1);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(suffix).unwrap(),
        expected_header
    );
    let known = reader
        .known_watched_input_at("test-machine", db.as_os_str().as_encoded_bytes())
        .unwrap()
        .unwrap();
    assert_eq!(known.input_id, "stored-database");
    assert_eq!(
        known.path_bytes.as_deref(),
        Some(db.as_os_str().as_encoded_bytes())
    );
    assert_eq!(known.chain, Some(initial_chain));
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM session", &[])
            .unwrap()[0]["n"],
        1
    );
    let admission = reader.query("SELECT s.id,r.chain,r.admitted FROM session_record r JOIN session s ON s.pk=r.session_pk JOIN input_ref i ON i.pk=r.input_pk WHERE i.id='stored-database'", &[]).unwrap();
    assert_eq!(admission.len(), 1);
    assert_eq!(
        admission[0]["id"],
        "codex:00000000-0000-4000-8000-000000000042"
    );
    assert_eq!(admission[0]["chain"], initial_chain);
    assert!(admission[0]["admitted"].as_u64().unwrap() > 0);
    drop(reader);
    let quiet = engine.run_pass().unwrap();
    assert!(quiet.errors.is_empty(), "{quiet:?}");
    assert_eq!(quiet.bytes_stored, 0);
    let reader = reads.reader().unwrap();
    assert_eq!(
        reader
            .input_progress("stored-database")
            .unwrap()
            .unwrap()
            .length,
        progress.length
    );
    assert_eq!(reader.query("SELECT s.id,r.chain,r.admitted FROM session_record r JOIN session s ON s.pk=r.session_pk JOIN input_ref i ON i.pk=r.input_pk WHERE i.id='stored-database'", &[]).unwrap(), admission);
}

#[test]
fn a_database_pass_yields_when_its_waiting_parser_has_no_local_work() {
    let home = polylogue_testkit::ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE kept(id INTEGER PRIMARY KEY, body TEXT);")
        .unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    let bytes = b"{\"t\":\"kept\"";
    let input = engine.location_id(&db);
    engine
        .writer
        .store_bytes(
            engine.archive.prepare_bytes(bytes.as_slice()).unwrap(),
            Batch {
                required_disposition: None,
                input: InputKey {
                    id: input.clone(),
                    format: "codex-database".into(),
                    arrival_rank: 0,
                },
                change: Change::First,
                observation: Some(engine.observation(&db, None)),
                continues: false,
                new_blob: None,
                blobs: vec![],
                length: bytes.len() as u64,
                parse: vec![],
                acquisition: None,
                basis: None,
                release: Default::default(),
            },
        )
        .unwrap();
    engine.load().unwrap();
    let origin = polylogue_sources::origin::by_name("codex-database").unwrap();
    let mut discovering = Pass::default();
    engine
        .capture_databases(
            &[collect::Found {
                path: db.clone(),
                origin,
            }],
            false,
            &mut discovering,
        )
        .unwrap();
    assert!(discovering.capture_pending);
    assert!(
        !engine.captures.pending(),
        "confirmed bytes need their first reading"
    );
    let plan = engine.stored_plan(&db).unwrap().unwrap();
    let mut parsed = Pass::default();
    engine.execute(vec![plan], &mut parsed).unwrap();
    assert!(parsed.errors.is_empty(), "{parsed:?}");
    let before = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&input)
        .unwrap()
        .unwrap();
    assert!(before.parsed && before.waiting);
    assert_eq!(before.parsed_upto, 0);
    assert_eq!(before.length, bytes.len() as u64);
    std::fs::remove_file(&db).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(!pass.capture_pending);
    assert_eq!(pass.bytes_read, 0);
    assert!(!engine.captures.pending());
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .input_progress(&input)
            .unwrap()
            .unwrap(),
        before
    );
}
