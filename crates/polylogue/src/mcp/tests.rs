//! MCP tool calls against a real daemon on a temporary archive.

use super::*;
use crate::test_daemon;
use serde_json::json;

fn args(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn tools_answer_with_the_daemon_and_refuse_bad_calls() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);

    let status: serde_json::Value =
        serde_json::from_str(&call(&archive, "status", args(json!({}))).unwrap()).unwrap();
    assert_eq!(status["store"]["messages"], 1);

    // Every option of `read` reaches the daemon.
    let text = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:s1", "format": "markdown"})),
    )
    .unwrap();
    let read: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(
        read["content"].as_str().unwrap().contains("walrus"),
        "{text}"
    );

    let text = call(&archive, "query", args(json!({"query": "walrus"}))).unwrap();
    assert!(text.contains("claude-code:s1"), "{text}");

    let e = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:s1", "pages": 1})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (bad_request)"), "{e}");
    let e = call(
        &archive,
        "excise",
        args(json!({"target": "session:claude-code:s1"})),
    )
    .unwrap_err();
    assert!(e.contains("no tool excise"), "{e}");
    let e = call(
        &archive,
        "read",
        args(json!({"session": "claude-code:nope"})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (not_found)"), "{e}");
}

mod read_pages;

mod events;

mod attachments;

#[test]
fn explain_is_a_table_tool_with_paged_catalogs_and_query_errors() {
    let w = test_daemon::start("grammar discovery");
    let archive = Archive::at(&w.daemon.archive.root);
    assert!(tools().iter().any(|t| t.name == "explain"));
    let page: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "fields", "limit": 1})),
        )
        .unwrap(),
    )
    .unwrap();
    let next = page["next"].as_str().unwrap();
    let second: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "fields", "limit": 1, "continuation": next})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(second["subject"], "fields");
    assert_ne!(page["entries"], second["entries"]);
    let query: serde_json::Value = serde_json::from_str(
        &call(
            &archive,
            "explain",
            args(json!({"subject": "query", "expression": "tool_calls where outcome:error"})),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(query["explanation"]["unit"], "tool_calls");
    for arguments in [
        json!({"subject": "query"}),
        json!({"subject": "query", "expression": "messages where invalid:x"}),
    ] {
        assert!(call(&archive, "explain", args(arguments)).is_err());
    }
}

#[test]
fn explain_preserves_query_error_kinds_and_correction_hints() {
    let w = test_daemon::start("grammar correction");
    let archive = Archive::at(&w.daemon.archive.root);
    let error = call(
        &archive,
        "explain",
        args(json!({"subject": "query", "expression": "\"unclosed phrase"})),
    )
    .unwrap_err();
    assert!(error.starts_with("error (invalid_query):"), "{error}");
    assert!(error.contains("try: \"unclosed phrase\""), "{error}");
}
