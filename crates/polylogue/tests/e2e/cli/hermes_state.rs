use super::claude_account_text::{messages, reading, records};
use super::*;
use polylogue_model::bundle::Record;
use serde_json::json;

#[test]
fn native_hermes_query_read_export_preserve_tool_data_and_supplied_compression_parents() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let native = home.join(".hermes/state.db");
    std::fs::create_dir_all(native.parent().unwrap()).unwrap();
    let db = rusqlite::Connection::open(&native).unwrap();
    db.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model TEXT,parent_session_id TEXT,end_reason TEXT,started_at REAL,title TEXT);
        INSERT INTO sessions VALUES('parent','cli','supplied-model',NULL,'compression',0,'Native parent');
        INSERT INTO sessions VALUES('child','cli','supplied-model','parent',NULL,1,'Native child');
        CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,reasoning_content TEXT,tool_calls TEXT,tool_call_id TEXT,timestamp REAL,active INTEGER,compacted INTEGER);
        INSERT INTO messages VALUES(1,'parent','user','stored ancestor Ω',NULL,NULL,NULL,0,0,1);").unwrap();
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
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| {
        status["store"]["messages"].as_u64() == Some(3)
    });
    let query = cli(
        &env,
        &["--json", "sessions where origin:hermes | sort by id"],
    );
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
    let child_messages = messages(&env, child);
    let messages = &child_messages;
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
    let inherited = reading(&env, child, &["--show", "inherited"]);
    // A supplied compression predecessor is retained as a link, rather than
    // a fork whose messages are prepended to the child's active reading.
    let own = reading(&env, child, &[]);
    // Only the session card differs: reading the parent prefix reports whether
    // the lineage was complete (docs/reading.md, lineage).
    let mut inherited_items = inherited.document.items.clone();
    let mut own_items = own.document.items.clone();
    for (items, expected) in [
        (
            &mut inherited_items,
            polylogue_model::read::ValueState::known(true),
        ),
        (
            &mut own_items,
            polylogue_model::read::ValueState::NotApplicable,
        ),
    ] {
        match &mut items[0].kind {
            polylogue_model::read::ItemKind::Card { card } => match card.as_mut() {
                polylogue_model::read::Card::Session {
                    lineage_complete, ..
                } => {
                    assert_eq!(*lineage_complete, expected);
                    *lineage_complete = polylogue_model::read::ValueState::NotApplicable;
                }
                other => panic!("first item is a session card, not {other:?}"),
            },
            other => panic!("first item is a card, not {other:?}"),
        }
    }
    assert_eq!(inherited_items, own_items);
    let exported = records(&env, child);
    let mut exported_messages: Vec<_> = exported
        .iter()
        .filter_map(|record| match record {
            Record::Message(record) => Some(record),
            _ => None,
        })
        .collect();
    exported_messages.sort_by_key(|record| record.order);
    assert_eq!(exported_messages.len(), 2);
    let links: Vec<_> = exported
        .iter()
        .filter_map(|record| match record {
            Record::Link(record) => Some(&record.link),
            _ => None,
        })
        .collect();
    assert_eq!(links.len(), 1);
    let link = links[0];
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
    let parent = super::claude_account_text::messages(&env, &link.parent.to_string());
    assert_eq!(parent[0]["blocks"][0]["text"], "stored ancestor Ω");
    assert_eq!(parent[0]["id"], link.at.as_ref().unwrap().to_string());
    assert!(matches!(
        &exported_messages[1].message.blocks[0],
        polylogue_model::Block::ToolResult {
            outcome: polylogue_model::ToolOutcome::Error {
                reason: Some(polylogue_model::ErrorReason::ExitCode(7))
            },
            ..
        }
    ));
}

#[test]
fn native_hermes_empty_active_path_is_distinct_from_missing_activity_flags() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let native = home.join(".hermes/state.db");
    std::fs::create_dir_all(native.parent().unwrap()).unwrap();
    let db = rusqlite::Connection::open(&native).unwrap();
    db.execute_batch("CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,title TEXT);
        INSERT INTO sessions VALUES('empty','cli','All inactive'),('mixed','cli','Mixed activity'),('unknown','cli','Unknown activity');
        CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,active INTEGER,compacted INTEGER);
        INSERT INTO messages VALUES
          (1,'empty','user','inactive first',0,0),(2,'empty','assistant','inactive second',0,0),
          (3,'mixed','user','abandoned first',0,0),(4,'mixed','assistant','active reply',1,0),(5,'mixed','user','abandoned last',0,0),
          (6,'unknown','user','unknown first',NULL,NULL),(7,'unknown','assistant','unknown second',NULL,NULL);").unwrap();
    drop(db);
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(7));
    let listed = cli(&env, &["--json", "sessions where origin:hermes"]);
    assert!(listed.status.success(), "{listed:?}");
    let listed = json(&listed);
    let mut states = Vec::new();
    for (title, stored, active) in [
        ("All inactive", 2, 0),
        ("Mixed activity", 3, 1),
        ("Unknown activity", 2, 2),
    ] {
        let sid = listed["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["title"] == title)
            .unwrap()["id"]
            .as_str()
            .unwrap();
        assert_eq!(messages(&env, sid).len(), stored);
        let query = format!("messages where session.title:\"{title}\"");
        let visible = cli(&env, &["--json", &query]);
        assert_eq!(
            query_rows(&visible)["count"],
            active,
            "{title}: default active-path query"
        );
        let all = format!("messages where session:{sid} active:*");
        let all = cli(&env, &["--json", &all]);
        assert!(all.status.success(), "{all:?}");
        assert_eq!(json(&all)["count"], stored);
        let markers = cli(
            &env,
            &[
                "--json",
                &format!("events where session:{sid} kind:active_path_empty"),
            ],
        );
        assert_eq!(
            query_rows(&markers)["count"],
            u64::from(title == "All inactive")
        );
        let export = records(&env, sid);
        let events: Vec<_> = export
            .iter()
            .filter_map(|record| match record {
                Record::Event(record) => Some(&record.event),
                _ => None,
            })
            .collect();
        let message_states: Vec<_> = events
            .iter()
            .filter(|event| event.kind == "hermes_message_state")
            .collect();
        assert_eq!(message_states.len(), stored);
        for event in message_states {
            if title == "All inactive" {
                assert_eq!(
                    serde_json::to_value(&event.data).unwrap(),
                    json!({"active":0,"compacted":0})
                );
            }
            if title == "Unknown activity" {
                assert_eq!(
                    serde_json::to_value(&event.data).unwrap(),
                    json!({"active":null,"compacted":null})
                );
            }
        }
        if title == "All inactive" {
            let marker = events
                .iter()
                .find(|e| e.kind == "active_path_empty")
                .unwrap();
            assert!(marker.after.is_none());
        }
        states.push((sid.to_owned(), json(&all), active));
    }
    let unknown = states
        .iter()
        .find(|(id, _, _)| id.starts_with("hermes:unknown@profile-"))
        .unwrap()
        .0
        .clone();
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    assert!(pool.reader().unwrap().query("SELECT m.on_active_path FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id=?1",&[polylogue_store::SqlValue::Text(unknown)]).unwrap().iter().all(|r|r["on_active_path"].is_null()));
    drop(pool);
    drop(daemon);
    let empty = states
        .iter()
        .find(|(id, _, _)| id.starts_with("hermes:empty@profile-"))
        .unwrap()
        .0
        .clone();
    // An unknown flag leaves the session's path unknown: `active` is
    // three-valued, so those messages answer `active:unknown`.
    for (flag, condition, expected) in [
        (Some(1), "active:true", 1),
        (Some(0), "active:true", 0),
        (None, "active:unknown", 2),
        (Some(0), "active:true", 0),
    ] {
        let db = rusqlite::Connection::open(&native).unwrap();
        db.execute("UPDATE messages SET active=?1 WHERE id=2", [flag])
            .unwrap();
        drop(db);
        let daemon = start_daemon(&env);
        wait_rows(
            &env,
            &format!("messages where session:{empty} {condition}"),
            |rows| rows["count"] == expected,
        );
        assert_eq!(messages(&env, &empty).len(), 2);
        drop(daemon);
    }
    std::fs::remove_file(&native).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _rebuilt = start_daemon(&env);
    wait_status(&env, |s| s["store"]["messages"].as_u64() == Some(7));
    for (sid, expected, active) in states {
        let query = format!("messages where session:{sid} active:*");
        let out = cli(&env, &["--json", &query]);
        assert!(out.status.success(), "{out:?}");
        let out = json(&out);
        assert_eq!(out["rows"], expected["rows"]);
        assert_eq!(out["columns"], expected["columns"]);
        assert_eq!(out["count"], expected["count"]);
        // `active` is three-valued: a session whose path is unknown has no
        // message on or off it, and `active:unknown` names them.
        let (on_path, unknown_path) = if sid.starts_with("hermes:unknown@") {
            (0, active)
        } else {
            (active, 0)
        };
        for (condition, expected) in [("active:true", on_path), ("active:unknown", unknown_path)] {
            let visible = cli(
                &env,
                &[
                    "--json",
                    &format!("messages where session:{sid} {condition}"),
                ],
            );
            assert_eq!(query_rows(&visible)["count"], expected, "{sid} {condition}");
        }
    }
}
