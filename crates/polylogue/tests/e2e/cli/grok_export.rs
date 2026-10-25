use super::*;
use serde_json::json;
use std::collections::BTreeSet;

#[test]
fn grok_account_import_reads_exact_turns_without_filename_or_export_order_identity() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let a = json!({"conversation":{"title":"Synthetic account A", "create_time":{"$date":{"$numberLong":"1712000000000"}}}, "responses":[
        {"response":{"sender":"human", "message":"# AGENTS.md\nExplain this synthetic document.", "create_time":{"$date":{"$numberLong":"1712000001000"}}}},
        {"response":{"sender":"assistant", "message":"answer A λ🙂", "create_time":1712000002}},
        {"response":{"sender":"assistant", "message":"answer A λ🙂", "create_time":1712000002}}
    ]});
    let mut b = a.clone();
    b["conversation"]["title"] = json!("Synthetic account B");
    b["responses"] = json!([a["responses"][0].clone(), {"response":{"sender":"assistant", "message":"answer B", "create_time":1712000002}}]);
    let first = json!({"conversations":[a.clone(), b.clone()]});
    let source = dir.path().join("prod-grok-backend.json");
    let original = serde_json::to_vec_pretty(&first).unwrap();
    std::fs::write(&source, &original).unwrap();
    let import = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(import.status.success(), "{import:?}");
    let query = cli(&env, &["--json", "origin:grok | sort by id"]);
    assert!(query.status.success(), "{query:?}");
    let rows = json(&query)["rows"].as_array().unwrap().clone();
    assert_eq!(
        rows.len(),
        2,
        "same opening text and time do not identify one conversation"
    );
    let sessions: BTreeSet<String> = rows
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    let mut original_ids = BTreeSet::new();
    for row in &rows {
        let session = row["id"].as_str().unwrap();
        assert!(session.starts_with("grok:~"));
        let read = cli(&env, &["--json", "read", session]);
        assert!(read.status.success(), "{read:?}");
        let read = json(&read);
        let messages = read["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["prompt_kind"], "prose");
        assert_eq!(
            messages[0]["blocks"][0]["text"],
            "# AGENTS.md\nExplain this synthetic document."
        );
        assert_eq!(messages[0]["at"], "2024-04-01T19:33:21Z");
        assert_eq!(messages[1]["role"], "assistant");
        assert!(messages[1].get("model").is_none());
        if row["title"] == "Synthetic account A" {
            assert_eq!(messages.len(), 3);
            assert_eq!(messages[1]["blocks"][0]["text"], "answer A λ🙂");
            assert_eq!(messages[1]["blocks"], messages[2]["blocks"]);
            assert_ne!(messages[1]["id"], messages[2]["id"]);
        } else {
            assert_eq!(messages.len(), 2);
        }
        original_ids.extend(
            messages
                .iter()
                .map(|message| message["id"].as_str().unwrap().to_owned()),
        );
    }
    let mut reordered = first.clone();
    reordered["conversations"].as_array_mut().unwrap().reverse();
    for conversation in reordered["conversations"].as_array_mut().unwrap() {
        conversation["conversation"]["create_time"] = json!("2024-04-01T21:33:20+02:00");
        conversation["responses"].as_array_mut().unwrap().reverse();
        for entry in conversation["responses"].as_array_mut().unwrap() {
            *entry = entry["response"].take();
        }
    }
    let renamed = dir.path().join("another-name.json");
    std::fs::write(&renamed, serde_json::to_vec(&reordered).unwrap()).unwrap();
    let import = cli(&env, &["import", renamed.to_str().unwrap()]);
    assert!(import.status.success(), "{import:?}");
    let query = cli(&env, &["--json", "origin:grok | sort by id"]);
    let rows = json(&query)["rows"].as_array().unwrap().clone();
    assert_eq!(
        rows.iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>(),
        sessions
    );
    let messages = cli(&env, &["--json", "messages where origin:grok"]);
    assert!(messages.status.success(), "{messages:?}");
    let messages = json(&messages);
    assert_eq!(messages["count"], 5);
    assert_eq!(
        messages["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap().to_owned())
            .collect::<BTreeSet<_>>(),
        original_ids
    );
    assert_eq!(std::fs::read(&source).unwrap(), original);

    let mut changed = a;
    changed["responses"]
        .as_array_mut()
        .unwrap()
        .push(json!({"response":{"sender":"assistant", "message":"new appended evidence"}}));
    let appended = dir.path().join("appended.json");
    std::fs::write(
        &appended,
        serde_json::to_vec(&json!({"conversations":[changed]})).unwrap(),
    )
    .unwrap();
    assert!(
        cli(&env, &["import", appended.to_str().unwrap()])
            .status
            .success()
    );
    let query = cli(&env, &["--json", "origin:grok"]);
    assert_eq!(
        json(&query)["count"],
        3,
        "ID-less append continuity is not guessed"
    );
}

#[test]
fn grok_native_ids_blank_content_and_unknown_shapes_are_visible() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let value = json!({"conversations":[{"conversation":{"conversationId":"native-account", "title":"", "model":"not-inherited"}, "responses":[
        {"sender":"human", "responseId":"native-user", "message":"", "create_time":null},
        {"sender":"assistant", "responseId":"native-answer", "parentResponseId":"native-user", "message":"native supplied λ🙂\nanswer", "model":"supplied-model", "createTime":{"$date":{"$numberLong":"1712000000123"}}},
        {"sender":"assistant", "responseId":"native-no-model", "message":"no inherited model"},
            {"sender":"assistant", "responseId":"native-empty-model", "message":"explicit empty model", "model":"", "create_time":0},
        {"sender":"unrecognized", "message":"unknown authorship"}, {"sender":"human", "message":null}, null
    ]}]});
    let source = dir.path().join("grok-native.json");
    std::fs::write(&source, serde_json::to_vec(&value).unwrap()).unwrap();
    let imported = cli(&env, &["import", source.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    let read = cli(&env, &["--json", "read", "grok:native-account"]);
    assert!(read.status.success(), "{read:?}");
    let read = json(&read);
    assert_eq!(read["messages"].as_array().unwrap().len(), 4);
    assert_eq!(read["messages"][0]["id"], "grok:native-user");
    assert_eq!(read["messages"][0]["blocks"][0]["text"], "");
    assert!(read["messages"][0].get("at").is_none());
    assert_eq!(read["messages"][1]["at"], "2024-04-01T19:33:20.123Z");
    assert_eq!(read["messages"][1]["parent"], "grok:native-user");
    assert_eq!(read["messages"][1]["model"], "supplied-model");
    assert!(read["messages"][2].get("model").is_none());
    assert!(read["messages"][2].get("at").is_none());
    assert_eq!(read["messages"][3]["model"], "");
    assert_eq!(read["messages"][3]["at"], "1970-01-01T00:00:00Z");
    let exported = cli(&env, &["export", "grok:native-account", "--format", "json"]);
    assert!(exported.status.success(), "{exported:?}");
    let exported: serde_json::Value = serde_json::from_slice(&exported.stdout).unwrap();
    assert_eq!(exported["sessions"][0]["messages"], read["messages"]);
    assert!(
        exported["sessions"][0]["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["kind"] == "grok_import_diagnostic"
                && event["data"]
                    .to_string()
                    .contains("unrecognized response sender"))
    );
}

#[test]
fn grok_metadata_before_the_shared_recognition_prefix_requires_the_standard_filename() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    let conversations = json!([{"conversation":{"id":"metadata-prefix"}, "responses":[
        {"responseId":"prefix-turn", "sender":"human", "message":"synthetic prefix turn"}
    ]}]);
    let bytes = format!(
        "{{\"account_metadata\":\"{}\",\"conversations\":{conversations}}}",
        "synthetic ".repeat(12)
    );
    let renamed = dir.path().join("renamed-metadata-first.json");
    std::fs::write(&renamed, &bytes).unwrap();
    let unsupported = cli(&env, &["--json", "import", renamed.to_str().unwrap()]);
    assert!(unsupported.status.success(), "{unsupported:?}");
    let report = json(&unsupported);
    assert_eq!(report["unrecognized"].as_array().unwrap().len(), 1);
    assert_eq!(report["imported"].as_array().unwrap().len(), 1);
    let standard = dir.path().join("prod-grok-backend.json");
    std::fs::write(&standard, &bytes).unwrap();
    let imported = cli(&env, &["--json", "import", standard.to_str().unwrap()]);
    assert!(imported.status.success(), "{imported:?}");
    assert_eq!(
        json(&imported)["already_stored"].as_array().unwrap().len(),
        1
    );
    let read = cli(&env, &["--json", "read", "grok:metadata-prefix"]);
    assert!(read.status.success(), "{read:?}");
    assert_eq!(
        json(&read)["messages"][0]["blocks"][0]["text"],
        "synthetic prefix turn"
    );
    drop(_daemon);
    std::fs::remove_file(renamed).unwrap();
    std::fs::remove_file(standard).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    let rebuilt = cli(&env, &["--json", "read", "grok:metadata-prefix"]);
    assert!(rebuilt.status.success(), "{rebuilt:?}");
    assert_eq!(
        json(&rebuilt)["messages"][0]["blocks"][0]["text"],
        "synthetic prefix turn"
    );
}
