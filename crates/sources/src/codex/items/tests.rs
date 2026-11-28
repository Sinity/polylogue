use super::*;
use polylogue_model::{AttachmentFailure, BlobHash};
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
fn plain_data_urls_stay_references_and_malformed_base64_records_a_failure() {
    let mut blobs = Vec::new();
    let b = message_blocks(
        &json!([
            {"type": "input_image", "image_url": "data:image/png,plain"},
            {"type": "input_image", "image_url": "data:image/png;base64,!!"}
        ]),
        &mut blobs,
    );
    let [Block::Image(reference), Block::Image(malformed)] = &b[..] else {
        panic!("expected two images")
    };
    assert_eq!(reference.url.as_deref(), Some("data:image/png,plain"));
    assert_eq!(reference.failure, None);
    assert_eq!(malformed.media_type.as_deref(), Some("image/png"));
    assert_eq!(malformed.failure, Some(AttachmentFailure::InvalidBase64));
    assert_eq!(malformed.blob, None);
    assert!(blobs.is_empty());
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

#[test]
fn multi_file_change_paths_ignore_json_object_order() {
    for changes in [r#"{"z.rs":{},"a.rs":{}}"#, r#"{"a.rs":{},"z.rs":{}}"#] {
        let changes: Json = serde_json::from_str(changes).unwrap();
        let blocks = recovered_call(
            &json!({"type":"FileChange","id":"synthetic-change","changes":changes}),
            &mut vec![],
        )
        .unwrap();
        let Block::ToolCall { path, input, .. } = &blocks[0] else {
            panic!()
        };
        assert_eq!(path.as_deref(), Some("a.rs"));
        assert_eq!(input, &changes);
    }
}

#[test]
fn recovered_mcp_result_keeps_media_and_structured_parts() {
    let raw = serde_json::json!({"type":"McpToolCall","id":"call","server":"files","tool":"read","status":"completed","durationMs":25,"arguments":{},"result":[{"type":"text","text":"two bytes"},{"type":"image","image_url":"data:image/png;base64,AQI="},{"type":"tool_reference","name":"next"}]});
    let mut blobs = Vec::new();
    let blocks = recovered_call(&raw, &mut blobs).unwrap();
    let Block::ToolResult {
        data,
        attachments,
        reported_duration_ms,
        output,
        ..
    } = &blocks[1]
    else {
        panic!("result")
    };
    assert_eq!(data.as_ref(), raw.get("result"));
    assert_eq!(*reported_duration_ms, Some(25));
    assert_eq!(output.as_deref(), Some("two bytes"));
    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0].blob, Some(BlobHash::of(&[1, 2])));
    assert_eq!(blobs[0].bytes, [1, 2]);
}

#[test]
fn patch_body_kept_for_built_in_and_namespaced_tools() {
    let patch = "*** Begin Patch\n*** Add File: synthetic.txt\n+λ\n*** End Patch";
    for name in ["apply_patch", "functions.apply_patch"] {
        assert_eq!(written_text(name, &json!(patch)).as_deref(), Some(patch));
        assert_eq!(action(name), ToolAction::FileEdit);
        assert_eq!(
            call_path(name, &json!(patch)).as_deref(),
            Some("synthetic.txt")
        );
    }
    assert_eq!(
        written_text("exec_command", &json!({"command":patch})),
        None
    );
}

#[test]
fn output_text_that_states_a_failure_names_it() {
    let outcome = |t: &str, o: &str| output_outcome(t, o).0;
    assert_eq!(
        outcome("exec", "Script error: boom\nWall time 0.0 seconds"),
        ToolOutcome::Error { reason: None }
    );
    let interrupted = ToolOutcome::Error {
        reason: Some(ErrorReason::Interrupted),
    };
    assert_eq!(outcome("shell", "aborted by user"), interrupted);
    assert_eq!(
        outcome("shell", "Command interrupted by the user"),
        interrupted
    );
    assert_eq!(
        outcome("custom_tool", r#"{"success":false,"output":"nope"}"#),
        ToolOutcome::Error { reason: None }
    );
}

#[test]
fn completed_status_without_an_exit_code_gives_the_outcome() {
    let outcome = |status: &str| completed(&json!({ "status": status })).map(|c| c.outcome);
    for status in ["failed", "error"] {
        assert_eq!(outcome(status), Some(ToolOutcome::Error { reason: None }));
    }
    for status in ["declined", "rejected"] {
        assert_eq!(
            outcome(status),
            Some(ToolOutcome::Error {
                reason: Some(ErrorReason::Denied(status.into()))
            })
        );
    }
    for status in ["interrupted", "cancelled", "aborted"] {
        assert_eq!(
            outcome(status),
            Some(ToolOutcome::Error {
                reason: Some(ErrorReason::Interrupted)
            })
        );
    }
    for status in ["inProgress", "in_progress", "running"] {
        assert_eq!(
            outcome(status),
            Some(ToolOutcome::Unknown {
                reason: UnknownReason::RunningInBackground
            })
        );
    }
    assert_eq!(outcome("completed"), Some(ToolOutcome::Ok));
    assert_eq!(outcome("queued"), None);
}

#[test]
fn a_duration_too_long_for_milliseconds_is_unknown() {
    let huge =
        json!({"status": "completed", "duration": {"secs": u64::MAX / 1000 + 1, "nanos": 0}});
    assert_eq!(completed(&huge).unwrap().duration_ms, None);
}
