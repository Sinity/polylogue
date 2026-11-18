use super::*;
use serde_json::json;

#[test]
fn antigravity_summary_database_updates_repeats_and_rebuilds_without_the_source() {
    let w = ClaudeHome::new();
    let path = w
        .home
        .join(".gemini/antigravity-cli/conversation_summaries.db");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE conversation_summaries(conversation_id TEXT PRIMARY KEY,title TEXT NOT NULL,workspace_uris TEXT NOT NULL,parent_conversation_id TEXT NOT NULL,last_modified_time TEXT NOT NULL,preview TEXT);
            INSERT INTO conversation_summaries VALUES('child','First title','[\"file:///work/project\"]','parent','2026-03-05 04:21:34+00:00','not a message');").unwrap();
    }
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let quiet = engine.run_pass().unwrap();
        assert!(quiet.errors.is_empty(), "{:?}", quiet.errors);
        assert_eq!(quiet.bytes_read, 0);
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader
                .display_title(&SessionId::parse("antigravity:child").unwrap())
                .unwrap()
                .as_deref(),
            Some("First title")
        );
        let inputs = reader.query("SELECT format FROM a.input WHERE format IN ('antigravity-database','tool-database')", &[]).unwrap();
        assert_eq!(inputs.len(), 1, "one location has one capture owner");
        assert_eq!(inputs[0]["format"], "antigravity-database");
    }
    {
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute("UPDATE conversation_summaries SET title='Updated title',last_modified_time='2026-03-06 05:21:34+01:00'", []).unwrap();
    }
    let before;
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        let id = SessionId::parse("antigravity:child").unwrap();
        assert_eq!(
            reader.display_title(&id).unwrap().as_deref(),
            Some("Updated title")
        );
        assert!(reader.messages(&id, false).unwrap().is_empty());
        before = reader
            .events(&id)
            .unwrap()
            .into_iter()
            .map(|event| (event.kind, event.at, event.data))
            .collect::<Vec<_>>();
        assert!(
            before
                .iter()
                .any(|(_, _, row)| row["title"] == "Updated title")
        );
        assert_eq!(
            reader
                .query(
                    "SELECT parent_id,kind FROM session_link WHERE parent_id='antigravity:parent'",
                    &[]
                )
                .unwrap()
                .len(),
            1
        );
    }
    fs::remove_file(&path).unwrap();
    delete_index(&w.root);
    let (mut engine, _lock) = w.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let id = SessionId::parse("antigravity:child").unwrap();
    assert_eq!(
        reader.display_title(&id).unwrap().as_deref(),
        Some("Updated title")
    );
    let after = reader
        .events(&id)
        .unwrap()
        .into_iter()
        .map(|event| (event.kind, event.at, event.data))
        .collect::<Vec<_>>();
    assert_eq!(after, before, "the stored row dump owns reconstruction");
}

#[test]
fn antigravity_exports_are_read_by_the_real_loop_and_rebuild() {
    let w = ClaudeHome::new();
    let dir = w.config(64 << 20).spool_dir("antigravity-exports");
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("export.json"),serde_json::to_vec(&json!({"source":"antigravity_language_server","cascadeId":"exported", "markdown":"### User Input\nMarkdown prompt\n### Planner Response\nMarkdown answer"})).unwrap()).unwrap();
    let home_dir = w.home.join(".gemini/antigravity/conversations");
    fs::create_dir_all(&home_dir).unwrap();
    let home_only = serde_json::to_vec(&json!({"source":"antigravity_language_server","cascadeId":"home-only","markdown":"### User Input\nIgnore this file"})).unwrap();
    fs::write(home_dir.join("export.json"), &home_only).unwrap();
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        assert!(
            reader
                .query(
                    "SELECT id FROM session WHERE id='antigravity:home-only'",
                    &[]
                )
                .unwrap()
                .is_empty()
        );
        let exported = reader
            .messages(&SessionId::parse("antigravity:exported").unwrap(), false)
            .unwrap();
        assert_eq!(
            exported.iter().map(Message::text).collect::<Vec<_>>(),
            ["Markdown prompt", "Markdown answer"]
        );
    }
    fs::remove_file(dir.join("export.json")).unwrap();
    delete_index(&w.root);
    let (mut engine, _lock) = w.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .messages(&SessionId::parse("antigravity:exported").unwrap(), false)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn opaque_conversation_files_and_conversion_events_share_a_session_and_rebuild() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".gemini/antigravity/conversations");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("not-converted.pb");
    fs::write(&path, b"synthetic opaque conversation file").unwrap();
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let spool = engine.config.spool_dir("antigravity-exports");
        // Wait for this fixture's worker before shutdown asks it to stop.
        while !fs::read_dir(&spool).is_ok_and(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| e.path().extension().is_some_and(|e| e == "json"))
        }) {
            std::thread::yield_now();
        }
    }
    let spool;
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        spool = engine.config.spool_dir("antigravity-exports");
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let quiet = engine.run_pass().unwrap();
        assert!(quiet.errors.is_empty(), "{:?}", quiet.errors);
        assert_eq!(quiet.bytes_read, 0);
    }
    fs::remove_file(path).unwrap();
    for entry in fs::read_dir(spool).unwrap() {
        fs::remove_file(entry.unwrap().path()).unwrap();
    }
    delete_index(&w.root);
    let (mut engine, _lock) = w.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let events = reader
        .events(&SessionId::parse("antigravity:not-converted").unwrap())
        .unwrap();
    assert_eq!(events.len(), 2, "{events:?}");
    let conversion = events
        .iter()
        .find(|event| event.kind == "antigravity_conversion")
        .unwrap();
    assert_eq!(conversion.data["status"], "server_unavailable");
    assert!(
        events
            .iter()
            .any(|event| event.kind == "antigravity_conversation_bytes")
    );
    let sessions = reader
        .query(
            "SELECT id FROM session WHERE id='antigravity:not-converted'",
            &[],
        )
        .unwrap();
    assert_eq!(sessions.len(), 1);
}
