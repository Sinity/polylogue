use super::*;
use serde_json::json;

#[test]
fn exit_codes_in_each_output_style() {
    let ok = |t: &str, o: &str| output_outcome(t, o).0;
    assert_eq!(
        ok(
            "exec_command",
            "Chunk ID: 1\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\nx"
        ),
        ToolOutcome::Ok
    );
    assert_eq!(
        ok(
            "shell_command",
            "Exit code: 2\nWall time: 0.1 seconds\nOutput:\n"
        ),
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(2))
        }
    );
    assert_eq!(
        ok(
            "shell",
            r#"{"output":"a\n","metadata":{"exit_code":1,"duration_seconds":0.0}}"#
        ),
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(1))
        }
    );
    assert_eq!(
        ok("exec", "Script completed\nWall time 3.6 seconds"),
        ToolOutcome::Ok
    );
    assert_eq!(
        ok("exec", "Script failed\nWall time 0.0 seconds"),
        ToolOutcome::Error { reason: None }
    );
    assert_eq!(
        ok(
            "apply_patch",
            "apply_patch verification failed: Failed to find expected lines"
        ),
        ToolOutcome::Error { reason: None }
    );
    assert_eq!(ok("update_plan", "Plan updated"), ToolOutcome::Ok);
    assert_eq!(
        ok("exec_command", "something without a status"),
        ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable
        }
    );
}

#[test]
fn running_commands_name_their_handle() {
    let (o, r) = output_outcome(
        "exec_command",
        "Chunk ID: c\nWall time: 1.0 seconds\nProcess running with session ID 56246\nOutput:\n",
    );
    assert_eq!(
        o,
        ToolOutcome::Unknown {
            reason: UnknownReason::RunningInBackground
        }
    );
    assert_eq!(r, Some(Running::Process("56246".into())));
    let (_, r) = output_outcome(
        "exec",
        "Script running with cell ID 4\nWall time 11.0 seconds",
    );
    assert_eq!(r, Some(Running::Cell("4".into())));
}

#[test]
fn mcp_namespaces_join_the_name() {
    let t = |ns: &str, n: &str| tool_name(&json!({"name": n, "namespace": ns})).unwrap();
    assert_eq!(t("mcp__serena", "find_symbol"), "mcp__serena__find_symbol");
    assert_eq!(
        t("mcp__context7__", "query_docs"),
        "mcp__context7__query_docs"
    );
    assert_eq!(
        t("collaboration", "send_message"),
        "collaboration.send_message"
    );
    assert_eq!(
        tool_name(&json!({"name": "exec_command"})).unwrap(),
        "exec_command"
    );
}

#[test]
fn commands_read_from_each_call_shape() {
    assert_eq!(
        call_command("exec_command", &json!({"cmd": "git status"})).as_deref(),
        Some("git status")
    );
    assert_eq!(
        call_command("shell", &json!({"command": ["bash", "-lc", "ls -la"]})).as_deref(),
        Some("ls -la")
    );
    let patch = "*** Begin Patch\n*** Update File: src/a.rs\n@@\n-x\n+y\n*** End Patch";
    assert_eq!(
        call_path("apply_patch", &json!(patch)).as_deref(),
        Some("src/a.rs")
    );
}

#[test]
fn data_url_images_become_blobs() {
    let mut blobs = Vec::new();
    let b = message_blocks(
        &json!([{"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}]),
        &mut blobs,
    );
    let Block::Image(a) = &b[0] else { panic!() };
    assert_eq!(a.media_type.as_deref(), Some("image/png"));
    assert_eq!(a.size, Some(5));
    assert_eq!(blobs.len(), 1);
}

#[test]
fn harness_text_is_marked() {
    let k = |t: &str| prompt_kind(&json!([{"type": "input_text", "text": t}]));
    assert_eq!(
        k("<environment_context>\n  <cwd>/x</cwd>"),
        PromptKind::Harness
    );
    assert_eq!(k("# AGENTS.md instructions for /x"), PromptKind::Harness);
    assert_eq!(
        k("<user_shell_command>ls</user_shell_command>"),
        PromptKind::Command
    );
    assert_eq!(k("fix the tests"), PromptKind::Prose);
}

#[test]
fn completed_items_give_exit_codes_and_durations() {
    let c = completed(&json!({"status": "completed", "exit_code": 3, "duration": {"secs": 1, "nanos": 500_000_000}})).unwrap();
    assert_eq!(
        c.outcome,
        ToolOutcome::Error {
            reason: Some(ErrorReason::ExitCode(3))
        }
    );
    assert_eq!(c.duration_ms, Some(1500));
    assert!(completed(&json!({"type": "Reasoning"})).is_none());
}
