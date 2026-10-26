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

/// The example queries the instructions give agents run as given, and each
/// tool they name exists.
#[test]
fn the_instructions_name_real_tools_and_working_queries() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);
    let examples: Vec<&str> = INSTRUCTIONS
        .split("Examples:")
        .nth(1)
        .unwrap()
        .split("An empty answer")
        .next()
        .unwrap()
        .split('`')
        .skip(1)
        .step_by(2)
        .collect();
    assert_eq!(examples.len(), 5, "{examples:?}");
    for query in examples {
        let answer = call(&archive, "query", args(json!({ "query": query })));
        assert!(answer.is_ok(), "{query}: {answer:?}");
    }
    let names: Vec<String> = tools().iter().map(|t| t.name.to_string()).collect();
    for tool in ["brief", "recall", "takeover", "query", "read", "note_add", "status"] {
        assert!(names.contains(&tool.to_owned()), "no tool {tool}");
        assert!(INSTRUCTIONS.contains(&format!("`{tool}`")), "{tool} is not described");
    }
}

mod read_pages;

mod events;

mod attachments;
