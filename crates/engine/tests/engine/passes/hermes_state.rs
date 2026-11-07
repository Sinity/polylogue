use super::*;
use polylogue_model::{Block, ErrorReason, PromptKind, Role, ToolOutcome};

fn tool_database(path: &Path, id: &str, question: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let connection = rusqlite::Connection::open(path).unwrap();
    connection.execute_batch("CREATE TABLE schema_version(version INTEGER); INSERT INTO schema_version VALUES(16);
        CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model TEXT,model_config TEXT,parent_session_id TEXT,started_at REAL,ended_at REAL,title TEXT,end_reason TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,tool_call_id TEXT,tool_name TEXT,tool_calls TEXT,reasoning_content TEXT,timestamp REAL,active INTEGER,compacted INTEGER,observed INTEGER);").unwrap();
    connection.execute("INSERT INTO sessions VALUES(?1,'cli','synthetic-model','{}',NULL,1712000000,NULL,'Native Hermes',NULL)", [id]).unwrap();
    connection
        .execute(
            "INSERT INTO messages VALUES(1,?1,'user',?2,NULL,NULL,NULL,NULL,1712000001,1,0,0)",
            rusqlite::params![id, question],
        )
        .unwrap();
    connection.execute("INSERT INTO messages VALUES(2,?1,'assistant','answer λ🙂',NULL,NULL,?2,'native reasoning',1712000002,1,0,0)", rusqlite::params![id,r#"[{"id":"ok-call","function":{"name":"terminal","arguments":"{\"command\":\"printf x\"}"}},{"id":"bad-call","function":{"name":"terminal","arguments":"{}"}}]"#]).unwrap();
    connection.execute("INSERT INTO messages VALUES(3,?1,'tool',?2,'ok-call','terminal',NULL,NULL,1712000003,1,0,0)", rusqlite::params![id,r#"{"output":"exact successful output","exit_code":0}"#]).unwrap();
    connection.execute("INSERT INTO messages VALUES(4,?1,'tool',?2,'bad-call','terminal',NULL,NULL,1712000004,1,0,0)", rusqlite::params![id,r#"{"output":"exact failure output","exit_code":7}"#]).unwrap();
}

fn hermes_ids(w: &ClaudeHome) -> Vec<SessionId> {
    w.reads()
        .reader()
        .unwrap()
        .query(
            "SELECT id FROM session WHERE id GLOB 'hermes:*' ORDER BY id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|row| SessionId::parse(row["id"].as_str().unwrap()).unwrap())
        .collect()
}

#[test]
fn hermes_native_capture_reads_roles_reasoning_and_zero_nonzero_tool_outcomes() {
    let w = ClaudeHome::new();
    tool_database(
        &w.home.join(".hermes/state.db"),
        "same-native-id",
        "Question Ω",
    );
    let (mut engine, _lock) = w.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let ids = hermes_ids(&w);
    assert_eq!(
        ids.len(),
        1,
        "stored native Hermes rows must become a conversation"
    );
    assert!(
        ids[0]
            .to_string()
            .starts_with("hermes:same-native-id@profile-")
    );
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let messages = reader.messages(&ids[0], false).unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[0].prompt_kind, Some(PromptKind::Prose));
    assert_eq!(messages[0].text(), "Question Ω");
    assert_eq!(messages[1].role, Role::Assistant);
    assert_eq!(messages[1].model.as_deref(), Some("synthetic-model"));
    assert!(messages[1].blocks.iter().any(|block| matches!(block, Block::Thinking { text:Some(text), .. } if text == "native reasoning")));
    assert_eq!(messages[2].role, Role::System);
    assert_eq!(messages[2].prompt_kind, Some(PromptKind::Harness));
    assert!(messages[2].blocks.iter().any(|block| matches!(block, Block::ToolResult { outcome:ToolOutcome::Ok, output:Some(text), .. } if text == "exact successful output")));
    assert!(messages[3].blocks.iter().any(|block| matches!(block, Block::ToolResult { outcome:ToolOutcome::Error {reason:Some(ErrorReason::ExitCode(7))}, output:Some(text), .. } if text == "exact failure output")));
}

#[test]
fn two_same_named_tool_databases_keep_profile_identity_after_provider_removal_and_rebuild() {
    let w = ClaudeHome::new();
    let second_home = w.home.parent().unwrap().join("second-home");
    let first = w.home.join(".hermes/state.db");
    let second = second_home.join(".hermes/state.db");
    tool_database(&first, "same-native-id", "first profile text");
    tool_database(&second, "same-native-id", "second profile text");
    let before;
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        engine.config.home = second_home;
        assert!(engine.run_pass().unwrap().errors.is_empty());
        before = hermes_ids(&w);
        assert_eq!(before.len(), 2);
        assert_ne!(before[0], before[1]);
        let bodies = before
            .iter()
            .map(|id| w.reads().reader().unwrap().messages(id, false).unwrap()[0].text())
            .collect::<Vec<_>>();
        assert!(bodies.contains(&"first profile text".into()));
        assert!(bodies.contains(&"second profile text".into()));
    }
    fs::remove_file(first).unwrap();
    fs::remove_file(second).unwrap();
    delete_index(&w.root);
    let (mut restored, _lock) = w.engine(64 << 20);
    let pass = restored.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(hermes_ids(&w), before);
    for id in before {
        assert_eq!(
            w.reads()
                .reader()
                .unwrap()
                .messages(&id, false)
                .unwrap()
                .len(),
            4
        );
    }
}

#[test]
fn hermes_many_messages_in_records_keep_supplied_subagent_metadata() {
    let w = ClaudeHome::new();
    let path = w.home.join(".hermes/state.db");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,title TEXT,model TEXT,cwd TEXT); INSERT INTO sessions VALUES('child','subagent','Supplied child title','supplied-model','/synthetic/work'); CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,active INTEGER,compacted INTEGER);").unwrap();
    let tx = db.transaction().unwrap();
    for n in 0..128 {
        tx.execute(
            "INSERT INTO messages VALUES(?1,'child','assistant',?2,0,0)",
            rusqlite::params![n, format!("stored body {n}")],
        )
        .unwrap();
    }
    tx.commit().unwrap();
    drop(db);
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let pool = w.reads();
    let r = pool.reader().unwrap();
    let ids = hermes_ids(&w);
    assert_eq!(ids.len(), 1);
    let session = r.session(&ids[0]).unwrap().unwrap();
    assert_eq!(session.kind, polylogue_model::SessionKind::Subagent);
    assert_eq!(session.title.unwrap().text, "Supplied child title");
    assert_eq!(session.cwd, vec!["/synthetic/work"]);
    let messages = r.messages(&ids[0], false).unwrap();
    assert_eq!(messages.len(), 128);
    assert!(
        messages
            .iter()
            .all(|m| m.model.as_deref() == Some("supplied-model"))
    );
    assert_eq!(
        r.query(
            "SELECT count(*) AS n FROM message WHERE on_active_path=0",
            &[]
        )
        .unwrap()[0]["n"],
        128
    );
    let counts = r.query("SELECT records FROM parse_state", &[]).unwrap();
    println!("Hermes128 messages: counts={counts:?}");
    assert_eq!(counts[0]["records"], 130);
}
