use super::*;
use serde_json::json;

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
