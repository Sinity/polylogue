use super::*;

#[test]
fn an_origin_without_a_parser_keeps_bytes_and_commits_capture_progress() {
    let home = ClaudeHome::new();
    let path = home.home.join(".pi/agent/sessions/project/session.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = b"{\"native\":\"synthetic retained record\"}\n";
    std::fs::write(&path, bytes).unwrap();
    let (mut engine, _lock) = open_engine(&home);
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    let reader = engine.reads.reader().unwrap();
    let known = reader
        .known_input_at("test-machine", &path.to_string_lossy())
        .unwrap()
        .unwrap();
    assert_eq!(known.parsed_upto, bytes.len() as u64);
    let state = reader
        .query(
            "SELECT status,reading_version,records FROM parse_state",
            &[],
        )
        .unwrap();
    assert_eq!(state[0]["status"], "no_parser_yet");
    assert_eq!(state[0]["reading_version"], "");
    assert_eq!(state[0]["records"], 0);
    let chain = reader
        .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
        .unwrap();
    let mut actual = Vec::new();
    chain
        .reader(Arc::clone(&engine.blobs))
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, bytes);
    drop(reader);
    let unchanged = engine.run_pass().unwrap();
    assert!(unchanged.errors.is_empty(), "{unchanged:?}");
    assert_eq!(unchanged.bytes_stored, 0);
}

#[test]
fn an_unparsed_database_commits_native_row_comparison_and_capture_progress() {
    let home = ClaudeHome::new();
    let path = home.home.join(".config/io.datasette.llm/logs.db");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let database = rusqlite::Connection::open(&path).unwrap();
    database
        .execute_batch("CREATE TABLE native_record(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO native_record VALUES(1,'synthetic retained body');")
        .unwrap();
    let (mut engine, _lock) = open_engine(&home);
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    engine.captures.completed(true);
    let parsed = engine.run_pass().unwrap();
    assert!(parsed.errors.is_empty(), "{parsed:?}");
    let reader = engine.reads.reader().unwrap();
    let progress = reader
        .input_progress(&engine.location_id(&path))
        .unwrap()
        .unwrap();
    assert!(progress.length > 0);
    assert_eq!(progress.parsed_upto, progress.length);
    let state = reader
        .query(
            "SELECT status,reading_version,records FROM parse_state",
            &[],
        )
        .unwrap();
    assert_eq!(state[0]["status"], "no_parser_yet");
    assert_ne!(state[0]["reading_version"], "");
    assert_eq!(state[0]["records"], 2);
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM tool_database", &[])
            .unwrap()[0]["n"],
        1
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM tool_row", &[])
            .unwrap()[0]["n"],
        1
    );
}
