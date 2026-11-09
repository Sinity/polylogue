//! The hook spool is the one Polylogue format that a deployment's previous
//! build reads after a rollback: hooks keep appending while the daemon is
//! replaced, and the build rolled back to reads what the new one wrote.
//! `rollback-lines.ndjson` holds one line of each shape this build writes,
//! with the per-invocation fields (`event_id`, `harness_pid`) fixed. The
//! deployed build's reader is checked against that file; a change to what
//! the hook writes changes the file, and needs that check again.

use super::*;

const LINES: &str = include_str!("rollback-lines.ndjson");

/// The keys a spool reader recognizes a hook spool line by.
const RECOGNIZED_FIRST_KEYS: &[&str] = &["event_type", "event_id", "at_ns"];

fn first_key(line: &str) -> &str {
    line.strip_prefix("{\"")
        .and_then(|rest| rest.split_once('"'))
        .map_or("", |(key, _)| key)
}

#[test]
fn the_hook_writes_exactly_the_line_shapes_the_rollback_fixture_holds() {
    let dir = tempfile::tempdir().unwrap();
    let startup = startup();
    let native = json!({
        "boot_id": "synthetic-boot",
        "git_identity": {
            "branch": null,
            "common_dir": "/home/op/project/.git",
            "head": "1111111111111111111111111111111111111111",
            "root_commits": ["2222222222222222222222222222222222222222"],
        },
        "native_unavailable": ["branch: empty native value"],
        "repository": "https://example.invalid/project.git",
        "terminal_identity": {
            "adapter": "kitty", "boot_id": "synthetic-boot", "host": "synthetic-host",
            "parent_process_id": 4241, "process_id": 4242, "window_id": "7",
        },
    });
    let start = json!({
        "cwd": "/home/op/project", "hook_event_name": "SessionStart",
        "session_id": "rollback-claude", "source": "startup",
    });
    let invocation = append_to_spool(
        dir.path(),
        Harness::ClaudeCode,
        "SessionStart",
        &start,
        &native,
        &startup,
    )
    .unwrap();
    for (harness, event, payload, native) in [
        (
            Harness::Codex,
            "PreToolUse",
            json!({"session_id": "rollback-codex", "tool_name": "shell", "tool_input": {"command": ["ls"]}}),
            json!({"boot_id": "synthetic-boot"}),
        ),
        (
            Harness::GeminiCli,
            "AfterTool",
            json!({"session_id": "rollback-gemini", "hook_event_name": "AfterTool"}),
            json!({}),
        ),
        (
            Harness::Hermes,
            "pre_llm_call",
            json!({"session_id": "rollback-hermes", "extra": {
                "is_first_turn": false, "conversation_history": [{"role": "user"}],
            }}),
            json!({}),
        ),
        (
            Harness::ClaudeCode,
            "Notification",
            json!({"message": "waiting for input"}),
            json!({}),
        ),
    ] {
        append_to_spool(dir.path(), harness, event, &payload, &native, &startup).unwrap();
    }
    let recorder = EmissionRecorder {
        dir: dir.path(),
        harness: Harness::ClaudeCode,
        event: "SessionStart",
        invocation: Some(&invocation),
        session: session_of(Harness::ClaudeCode, &start, &startup),
        native: &native,
        startup: &startup,
    };
    assert!(recorder.emit(&["rollback-delivery".into()], "context", |_, _, _| true));

    let mut written = Vec::new();
    for file in std::fs::read_dir(dir.path()).unwrap() {
        let text = std::fs::read_to_string(file.unwrap().path()).unwrap();
        written.extend(text.lines().map(str::to_owned));
    }
    let expected: Vec<Value> = LINES
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(written.len(), expected.len(), "{written:#?}");
    for line in &written {
        assert!(
            RECOGNIZED_FIRST_KEYS.contains(&first_key(line)),
            "a reader recognizes a spool file by its first key: {line}"
        );
        let mut value: Value = serde_json::from_str(line).unwrap();
        let fixture = expected
            .iter()
            .find(|fixture| fixture["event"] == value["event"])
            .unwrap_or_else(|| panic!("no fixture line for {line}"));
        assert!(value["event_id"].as_str().is_some_and(|id| !id.is_empty()));
        assert!(value["harness_pid"].is_u64());
        value["event_id"] = fixture["event_id"].clone();
        value["harness_pid"] = fixture["harness_pid"].clone();
        if value.get("harness_start").is_some() {
            assert!(value["harness_start"].is_null() || value["harness_start"].is_u64());
            value["harness_start"] = fixture["harness_start"].clone();
        }
        if let Some(receipt) = value["payload"].get_mut("invocation") {
            assert_eq!(receipt.as_str(), Some(invocation.as_str()));
            *receipt = fixture["payload"]["invocation"].clone();
        }
        assert_eq!(&value, fixture);
    }
    for line in LINES.lines() {
        assert!(RECOGNIZED_FIRST_KEYS.contains(&first_key(line)), "{line}");
    }
}

#[test]
fn this_builds_reader_takes_every_rollback_fixture_line() {
    use polylogue_sources::hook_spool::{HookSpool, ParserState, recognize};
    assert!(recognize(
        Path::new("/home/op/.local/state/polylogue/hooks/4242.ndjson"),
        LINES.as_bytes()
    ));
    let step =
        polylogue_sources::parse::run(&HookSpool, &mut ParserState::default(), LINES.as_bytes(), 0)
            .unwrap();
    assert_eq!(step.counts.records, 6);
    assert_eq!(step.counts.unreadable, Default::default());
    let sessions: Vec<String> = step
        .output
        .events
        .iter()
        .map(|event| event.session.as_ref().unwrap().to_string())
        .collect();
    assert_eq!(
        sessions,
        [
            "claude-code:rollback-claude",
            "codex:rollback-codex",
            "gemini-cli:rollback-gemini",
            "hermes:rollback-hermes@profile-default",
            "claude-code:rollback-claude",
        ]
    );
    assert_eq!(step.counts.skipped.len(), 1, "{:?}", step.counts.skipped);
}
