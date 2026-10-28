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

mod explain;

#[test]
fn server_instructions_include_guidance_and_current_command_names() {
    let server = Polylogue {
        archive: std::path::PathBuf::from("/synthetic/archive"),
    };
    let info = server.get_info();
    let text = info.instructions.unwrap();
    assert!(text.contains("Continue pages"));
    for command in commands::COMMANDS.iter().filter(|c| c.mcp) {
        assert!(text.contains(&format!("`{}`", command.mcp_name())));
    }
    assert_eq!(text, crate::manual::text());
}

#[test]
fn manual_query_examples_compile_through_the_public_command() {
    let w = test_daemon::start("manual query examples");
    let archive = Archive::at(&w.daemon.archive.root);
    for query in crate::manual::text()
        .split('`')
        .filter(|text| text.starts_with("sessions where ") || text.starts_with("messages where "))
    {
        call(
            &archive,
            "explain",
            args(json!({"subject":"query", "expression":query})),
        )
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    }
}

#[test]
fn errors_show_archived_controls_as_text() {
    let hostile = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";
    let text = super::error_text(
        hostile,
        hostile,
        Some(&serde_json::json!({"suggestion":hostile})),
    );
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}')
    );
    assert!(text.contains("\\x1b]52;c;YQ==\\x07"));
}
