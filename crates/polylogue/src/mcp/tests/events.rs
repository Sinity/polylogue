use super::*;

#[test]
fn mcp_reads_input_events_through_the_same_typed_reader_and_budget_cursor() {
    let w = test_daemon::start("independent ordinary conversation");
    let source = w.home.join("synthetic.atof.jsonl");
    std::fs::write(&source,format!("{}\n",json!({"atof_version":"0.1","kind":"mark","uuid":"synthetic","name":"neutral","data":{"text":" exact MCP λ🙂 evidence ","credential":"--secret=\"unfinished --token=\"synthetic-later-value\" tail"}}))).unwrap();
    let result = crate::daemon::service::handle(
        crate::daemon::request::Request::Import {
            request_id: "mcp-event-fixture".into(),
            paths: vec![source],
        },
        &w.daemon,
    );
    assert!(
        matches!(result, crate::daemon::Response::Ok { .. }),
        "{result:?}"
    );
    let archive = Archive::at(&w.daemon.archive.root);
    let listed: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "query",
            args(json!({"query":"events where source:hermes"})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(listed["count"], 1);
    let reference = format!("event:{}", listed["rows"][0]["id"].as_str().unwrap());
    let ordinary = call(
        &archive,
        "read",
        args(json!({"session":reference,"format":"json"})),
    )
    .unwrap();
    assert!(!ordinary.contains("synthetic-later-value"));
    let ordinary: serde_json::Value = serde_json::from_str(&ordinary).unwrap();
    assert!(ordinary["event"].get("session").is_none());
    assert_eq!(
        ordinary["event"]["data"]["data"]["text"],
        " exact MCP λ🙂 evidence "
    );
    let first: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "read",
            args(json!({"session":reference,"budget":2})),
        )
        .unwrap(),
    )
    .unwrap();
    let cursor = first["next"].as_str().unwrap();
    assert!(!cursor.contains("evidence"));
    let next: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "read",
            args(json!({"session":reference,"budget":3,"cursor":cursor})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_ne!(next["text"], first["text"]);
    let wrong = call(
        &archive,
        "read",
        args(json!({"session":reference,"view":"dialogue","budget":2,"cursor":cursor})),
    )
    .unwrap_err();
    assert!(wrong.starts_with("error (bad_request)"), "{wrong}");
}
