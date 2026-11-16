use super::*;
use command_surface::{Via, author_of};
use serde_json::json;
use std::io::Write;

fn caller_details(tool: Option<&str>) -> CallerEvidence {
    CallerEvidence {
        harness: Harness::new("claude-code"),
        session: AgentSession::parse("claude-code:stale-startup"),
        harness_pid: Some(17),
        tool_call_id: tool.map(str::to_owned),
        ..Default::default()
    }
}
fn append(path: &std::path::Path, event: serde_json::Value) {
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(&mut output, &event).unwrap();
    output.write_all(b"\n").unwrap();
}
fn event(kind: &str, session: &str, tool: Option<&str>, agent: Option<&str>) -> serde_json::Value {
    json!({"harness_pid":17,"provider":"claude-code","event":kind,"session_id":session,"payload":{"tool_use_id":tool,"agent_id":agent,"tool_name":"Bash","tool_input":{"command":"polylogue status"}}})
}

#[test]
fn caller_tool_call_selects_the_worker_session() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = OpenCalls::new(directory.path().into());
    append(&path, event("SessionStart", "resumed", None, None));
    append(
        &path,
        event(
            "PreToolUse",
            "resumed",
            Some("call-one"),
            Some("worker-one"),
        ),
    );
    append(
        &path,
        event(
            "PreToolUse",
            "resumed",
            Some("call-two"),
            Some("worker-two"),
        ),
    );
    let caller = caller_details(Some("call-two"));
    let selected = registry.identity(&caller).unwrap();
    let accepted = author_of(&caller, Via::Mcp, Some(&selected));
    assert_eq!(
        accepted.author.to_string(),
        "agent:claude-code:agent-worker-two"
    );
    let caller = caller_details(None);
    assert!(registry.identity(&caller).is_none());
    append(
        &path,
        event("PostToolUse", "resumed", Some("call-two"), None),
    );
    assert!(
        registry.identity(&caller).is_none(),
        "a unique remaining call does not identify the caller"
    );
    assert!(
        registry
            .identity(&caller_details(Some("call-two")))
            .is_none()
    );
    let session_only = CallerEvidence {
        session: AgentSession::parse("claude-code:explicit-session"),
        ..caller
    };
    assert_eq!(
        author_of(&session_only, Via::Cli, None).author.to_string(),
        "agent:claude-code:explicit-session"
    );
}

#[test]
fn partial_hook_record_is_observed_only_after_its_newline_and_file_replacement_resets_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = OpenCalls::new(directory.path().into());
    let caller = caller_details(Some("call-one"));
    let next = serde_json::to_vec(&event("PreToolUse", "first", Some("call-one"), None)).unwrap();
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap();
    output.write_all(&next).unwrap();
    assert!(registry.identity(&caller).is_none());
    output.write_all(b"\n").unwrap();
    assert_eq!(
        registry.identity(&caller).unwrap().session.to_string(),
        "claude-code:first"
    );
    let replacement = directory.path().join("replacement");
    append(
        &replacement,
        event("PreToolUse", "replacement", Some("other-call"), None),
    );
    std::fs::rename(&replacement, &path).unwrap();
    assert!(registry.identity(&caller).is_none());
    assert_eq!(
        registry
            .identity(&caller_details(Some("other-call")))
            .unwrap()
            .session
            .to_string(),
        "claude-code:replacement"
    );
}

#[test]
fn a_reused_pid_never_applies_another_harnesses_hook_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = OpenCalls::new(directory.path().into());
    append(&path, event("PreToolUse", "old", Some("call-one"), None));
    assert!(
        registry
            .identity(&caller_details(Some("call-one")))
            .is_some()
    );
    let caller = CallerEvidence {
        harness: Harness::new("codex"),
        harness_pid: Some(17),
        tool_call_id: Some("call-one".into()),
        ..Default::default()
    };
    assert!(registry.identity(&caller).is_none());
}
