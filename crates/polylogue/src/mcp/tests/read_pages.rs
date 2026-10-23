use super::*;

#[test]
fn mcp_read_pages_use_bound_continuations() {
    let world = test_daemon::start("synthetic MCP prompt λ🙂 long enough for pages");
    let archive = Archive::at(&world.daemon.archive.root);
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
