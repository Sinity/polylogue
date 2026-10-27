use super::*;

#[test]
fn mcp_read_pages_use_bound_continuations() {
    let daemon = test_daemon::start("synthetic MCP prompt λ🙂 long enough for pages");
    let archive = Archive::at(&daemon.daemon.archive.root);
    let first = call(
        &archive,
        "read",
        args(json!({"session":"claude-code:s1", "budget":1})),
    )
    .unwrap();
    let first: serde_json::Value = serde_json::from_str(&first).unwrap();
    let token = first["next"].as_str().unwrap();
    let second = call(
        &archive,
        "read",
        args(json!({"session":"claude-code:s1", "budget":2, "cursor":token})),
    )
    .unwrap();
    let second: serde_json::Value = serde_json::from_str(&second).unwrap();
    assert_ne!(first["text"], second["text"]);
    let error = call(
        &archive,
        "read",
        args(json!({"session":"claude-code:s1", "view":"dialogue", "budget":1, "cursor":token})),
    )
    .unwrap_err();
    assert!(error.starts_with("error (bad_request)"), "{error}");
}

#[test]
fn budgeted_mcp_output_includes_content_once() {
    let daemon = test_daemon::start(&"synthetic word ".repeat(10000));
    let archive = Archive::at(&daemon.daemon.archive.root);
    let budget = 2048;
    let output = call(
        &archive,
        "read",
        args(json!({"session":"claude-code:s1","budget":budget})),
    )
    .unwrap();
    let data: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert!(data.get("slices").is_none());
    assert_eq!(data["text"].as_str().unwrap().chars().count(), budget * 4);
    // Metadata and sealed cursors add a fixed cost, independent of page size.
    assert!(output.len() < budget * 4 + 5000, "{}", output.len());
    let sliced = call(
        &archive,
        "read",
        args(json!({"session":"claude-code:s1","budget":budget,"slices":true})),
    )
    .unwrap();
    let sliced: serde_json::Value = serde_json::from_str(&sliced).unwrap();
    assert!(sliced["slices"].is_array());
}
