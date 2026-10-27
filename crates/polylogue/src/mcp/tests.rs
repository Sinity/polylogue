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
fn added_read_formats_work_through_the_public_command() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);
    for format in ["csv", "org", "yaml"] {
        let text = call(
            &archive,
            "read",
            args(json!({"session":"claude-code:s1","format":format})),
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["format"], format);
        assert!(value["content"].as_str().unwrap().contains("walrus"));
    }
}

#[test]
fn added_formats_keep_budgeted_read_cursors_usable() {
    let w = test_daemon::start("the walrus migration has a long synthetic prompt to page through");
    let archive = Archive::at(&w.daemon.archive.root);
    for format in ["csv", "org", "yaml"] {
        let first = call(
            &archive,
            "read",
            args(json!({"session":"claude-code:s1","budget":2,"format":format})),
        )
        .unwrap();
        let first: serde_json::Value = serde_json::from_str(&first).unwrap();
        let cursor = first["next"].as_str().unwrap();
        assert_eq!(first["format"], format);
        assert!(first["content"].as_str().unwrap().contains(cursor));
        let second = call(
            &archive,
            "read",
            args(json!({"session":"claude-code:s1","budget":2,"format":format,"cursor":cursor})),
        )
        .unwrap();
        let second: serde_json::Value = serde_json::from_str(&second).unwrap();
        assert_ne!(first["text"], second["text"]);
    }
}
