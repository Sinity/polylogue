use super::*;
use polylogue_model::{
    Origin,
    authorship::{Channel, compute_authorship},
};
use serde_json::json;
use std::io::Write;

fn evidence(tool: Option<&str>) -> CallerEvidence {
    CallerEvidence {
        harness: Some(Origin::new("claude-code").unwrap()),
        session: Some(SessionId::parse("claude-code:stale-startup").unwrap()),
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
fn latest_session_and_exact_tool_call_replace_stale_startup_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = Registry::new(directory.path().into());
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
    let caller = evidence(Some("call-two"));
    let selected = registry.identity(&caller).unwrap();
    let accepted = compute_authorship(Channel::Mcp, &caller, Some(&selected));
    assert_eq!(
        accepted.author.to_string(),
        "agent:claude-code:agent-worker-two"
    );
    let caller = evidence(None);
    let selected = registry.identity(&caller).unwrap();
    assert_eq!(
        compute_authorship(Channel::Cli, &caller, Some(&selected))
            .author
            .to_string(),
        "agent:claude-code:resumed"
    );
    append(
        &path,
        event("PostToolUse", "resumed", Some("call-two"), None),
    );
    let selected = registry.identity(&caller).unwrap();
    assert_eq!(
        compute_authorship(Channel::Cli, &caller, Some(&selected))
            .author
            .to_string(),
        "agent:claude-code:agent-worker-one"
    );
}

#[test]
fn partial_hook_record_is_observed_only_after_its_newline_and_file_replacement_resets_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = Registry::new(directory.path().into());
    append(&path, event("SessionStart", "first", None, None));
    assert_eq!(
        registry
            .identity(&evidence(None))
            .unwrap()
            .session
            .to_string(),
        "claude-code:first"
    );
    let next = serde_json::to_vec(&event("SessionStart", "second", None, None)).unwrap();
    let mut output = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    output.write_all(&next).unwrap();
    assert_eq!(
        registry
            .identity(&evidence(None))
            .unwrap()
            .session
            .to_string(),
        "claude-code:first"
    );
    output.write_all(b"\n").unwrap();
    assert_eq!(
        registry
            .identity(&evidence(None))
            .unwrap()
            .session
            .to_string(),
        "claude-code:second"
    );
    let replacement = directory.path().join("replacement");
    append(
        &replacement,
        event("SessionStart", "replacement", None, None),
    );
    std::fs::rename(&replacement, &path).unwrap();
    assert_eq!(
        registry
            .identity(&evidence(None))
            .unwrap()
            .session
            .to_string(),
        "claude-code:replacement"
    );
}

#[test]
fn a_quoted_reference_to_the_program_does_not_claim_a_cli_tool_call() {
    assert!(!runs_polylogue("echo 'polylogue status'"));
    assert!(runs_polylogue("env LANG=C polylogue status"));
    assert!(runs_polylogue("/tools/polylogue status"));
}

#[test]
fn a_reused_pid_never_applies_another_harnesses_hook_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("17.ndjson");
    let registry = Registry::new(directory.path().into());
    append(&path, event("SessionStart", "old", None, None));
    assert!(registry.identity(&evidence(None)).is_some());
    let caller = CallerEvidence {
        harness: Some(Origin::new("codex").unwrap()),
        harness_pid: Some(17),
        ..Default::default()
    };
    assert!(registry.identity(&caller).is_none());
}
