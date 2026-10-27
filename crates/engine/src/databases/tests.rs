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
        },
    };
    let mut databases = Databases::default();
    let (wake, changes) = channel();
    databases
        .start(
            vec![source()],
            archive,
            writer.sender().unwrap(),
            Some(wake),
        )
        .unwrap();
    assert!(matches!(changes.recv().unwrap(), LoopMsg::Changed));
    let done = databases.completed(true);
    assert!(done.changed);
    assert!(done.errors.is_empty(), "{:?}", done.errors);

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
    assert!(done.errors.is_empty(), "{:?}", done.errors);
}
