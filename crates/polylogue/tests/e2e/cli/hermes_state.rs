use super::*;
use serde_json::json;

#[test]
fn native_hermes_query_read_export_preserve_tool_data_and_supplied_compression_ancestry() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let native = home.join(".hermes/state.db");
    std::fs::create_dir_all(native.parent().unwrap()).unwrap();
    let db = rusqlite::Connection::open(&native).unwrap();
    db.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model TEXT,parent_session_id TEXT,end_reason TEXT,started_at REAL,title TEXT);
        INSERT INTO sessions VALUES('parent','cli','supplied-model',NULL,'compression',0,'Native parent');
        INSERT INTO sessions VALUES('child','cli','supplied-model','parent',NULL,1,'Native child');
        CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,reasoning_content TEXT,tool_calls TEXT,tool_call_id TEXT,timestamp REAL,active INTEGER,compacted INTEGER);
        INSERT INTO messages VALUES(1,'parent','user','retained ancestor Ω',NULL,NULL,NULL,0,0,1);").unwrap();
    let calls = json!([{"id":"native-call","function":{"name":"terminal","arguments":"{\"command\":\"printf synthetic\",\"custom\":7}"}}]).to_string();
    db.execute("INSERT INTO messages VALUES(2,'child','assistant','native reply λ🙂','native reasoning',?1,NULL,1,1,0)", [&calls]).unwrap();
    let result = json!({"output":"exact native output","exit_code":7,"custom":{"supplied":false}});
    db.execute(
        "INSERT INTO messages VALUES(3,'child','tool',?1,NULL,NULL,'native-call',2,1,0)",
        [result.to_string()],
    )
    .unwrap();
    drop(db);
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"].as_u64() == Some(3)
    });
    let query = cli(&env, &["--json", "origin:hermes | sort by id"]);
    assert!(query.status.success(), "{query:?}");
    let query = json(&query);
    assert_eq!(query["count"], 2);
    let child = query["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["title"] == "Native child")
        .unwrap()["id"]
        .as_str()
        .unwrap();
    assert!(child.starts_with("hermes:child@profile-"));
    let read = cli(&env, &["--json", "read", child]);
    assert!(read.status.success(), "{read:?}");
    let read = json(&read);
    let messages = read["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["model"], "supplied-model");
    assert_eq!(messages[0]["at"], "1970-01-01T00:00:01Z");
    assert_eq!(messages[0]["blocks"][0]["text"], "native reasoning");
    let call = messages[0]["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|block| block["type"] == "tool_call")
        .unwrap();
    assert_eq!(
        call["input"],
        json!({"command":"printf synthetic","custom":7})
    );
    assert_eq!(call["call_id"], "native-call");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(messages[1]["prompt_kind"], "harness");
    assert_eq!(messages[1]["blocks"][0]["data"], result);
    let logical = cli(&env, &["--json", "read", child, "--logical"]);
    assert!(logical.status.success(), "{logical:?}");
    let logical = json(&logical);
    assert_eq!(logical["messages"].as_array().unwrap().len(), 2);
    let exported = cli(&env, &["export", child, "--format", "json"]);
    assert!(exported.status.success(), "{exported:?}");
    let document =
        polylogue_model::export::Document::from_json(&String::from_utf8(exported.stdout).unwrap())
            .unwrap();
    assert_eq!(document.sessions[0].messages.len(), 2);
    let link = &document.sessions[0].links[0];
    assert_eq!(link.kind, polylogue_model::LinkKind::Continuation);
    assert!(link.parent.native().as_str().starts_with("parent@profile-"));
    assert!(
        link.at
            .as_ref()
            .unwrap()
            .native()
            .as_str()
            .starts_with("parent@profile-")
    );
    let parent = cli(&env, &["--json", "read", &link.parent.to_string()]);
    assert!(parent.status.success(), "{parent:?}");
    let parent = json(&parent);
    assert_eq!(
        parent["messages"][0]["blocks"][0]["text"],
        "retained ancestor Ω"
    );
    assert_eq!(
        parent["messages"][0]["id"],
        link.at.as_ref().unwrap().to_string()
    );
    assert!(matches!(
        &document.sessions[0].messages[1].blocks[0],
        polylogue_model::Block::ToolResult {
            outcome: polylogue_model::ToolOutcome::Error {
                reason: Some(polylogue_model::ErrorReason::ExitCode(7))
            },
            ..
        }
    ));
}
