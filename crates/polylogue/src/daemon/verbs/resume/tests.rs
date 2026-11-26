use crate::test_daemon::{self, Call};
use serde_json::json;

#[test]
fn resume_in_hands_off() {
    let d = test_daemon::start("continue the clock work");
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1", "dry_run": true,
                "steps": [{"verb": "resume", "in": "codex"}],
            }),
        )
        .id("resume-dry")
        .answer(&d.daemon),
    );
    let outcome = &answer["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "would_do", "{answer}");
    assert_eq!(outcome["refusal"], serde_json::Value::Null, "{answer}");
    let would = outcome["would"].as_str().unwrap();
    let tokens = would
        .strip_prefix("would continue claude-code:s1 in codex with a brief of ")
        .unwrap()
        .strip_suffix(" tokens")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let invocation =
        d.daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "resume-test",
            ));
    let reader = d.daemon.reads.reader().unwrap();
    let context = d.daemon.resolve_operation_context(invocation).unwrap();
    let source = polylogue_model::SessionId::parse("claude-code:s1").unwrap();
    let prepared = crate::brief::prepare(
        crate::brief::BriefContext {
            reader: &reader,
            pending_deliveries: &[],
            delivery_id: "resume-dry-1",
            context: &context,
            ranking_identity: None,
            near: None,
        },
        &d.daemon.scope,
        &[],
        &crate::brief::Ask::Session {
            session: source,
            for_session: None,
        },
        command_surface::TokenBudget::Default,
    )
    .unwrap();
    assert_eq!(
        tokens,
        polylogue_model::read::estimate_tokens(&format!(
            "Hand-off from claude-code:s1:\n{}",
            prepared.text
        ))
    );
    drop(reader);
    assert_eq!(d.daemon.acts.hosted_sessions().len(), 0);
    let deliveries = crate::test_answers::empty(
        Call::new("rows", json!({"selection": "deliveries"})).answer(&d.daemon),
    );
    assert_eq!(deliveries["reply"]["count"], 0);
}

#[test]
fn resume_over_several_is_ambiguous_with_candidates() {
    let d = test_daemon::start_with_sources_and_workers(
        "first session",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/s2.jsonl"),
                polylogue_testkit::claude_user("u2", "s2", "second session"),
            )
            .unwrap();
        },
        1,
        2,
    );
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "references": ["session:claude-code:s1", "session:claude-code:s2"],
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    let outcome = &answer["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "failed");
    assert_eq!(outcome["error"]["kind"], "ambiguous");
    assert_eq!(
        outcome["error"]["candidates"],
        json!(["claude-code:s1", "claude-code:s2"])
    );
}

/// The synthetic session records no directory, so its harness resume is
/// refused rather than shown as a line to run anywhere
/// (docs/session-control.md, the resume directory).
#[test]
fn resume_dry_run_without_a_directory_is_refused() {
    let d = test_daemon::start("resume this work");
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1", "dry_run": true,
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    assert_eq!(
        answer["reply"]["steps"][0]["outcome"]["refusal"]["message"],
        "directory_unknown: the session recorded no directory to resume in; continue with context instead: polylogue session:claude-code:s1 brief"
    );
}

/// An option of `--in` without it is refused, not dropped.
#[test]
fn resume_options_for_a_hand_off_need_in() {
    let d = test_daemon::start("continue the clock work");
    for step in [
        json!({"verb": "resume", "at": "message:claude-code:u1"}),
        json!({"verb": "resume", "cwd": "/home/op"}),
        json!({"verb": "resume", "stop_source": true}),
    ] {
        let answer = Call::new(
            "rows",
            json!({"selection": "session:claude-code:s1", "dry_run": true, "steps": [step]}),
        )
        .id("resume-without-in")
        .answer(&d.daemon);
        let value = serde_json::to_value(&answer).unwrap();
        let text = value.to_string();
        assert!(text.contains("apply only with --in HARNESS"), "{text}");
    }
}

/// A session with live events showing a tool call in progress is running:
/// `resume` refuses it with the lines that reach it, while an ended one gets
/// its resume command.
#[test]
fn resume_refuses_a_running_session() {
    let d = test_daemon::start_with_sources_and_workers(
        "running session",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/s2.jsonl"),
                polylogue_testkit::claude_user("u2", "s2", "ended session"),
            )
            .unwrap();
            let spool = test_daemon::engine_config(home.to_owned())
                .spool_dir(polylogue_sources::hook_spool::ORIGIN);
            std::fs::create_dir_all(&spool).unwrap();
            let now = jiff::Timestamp::now().as_nanosecond().to_string();
            let events = [
                json!({"at_ns": now, "event": "PreToolUse", "provider": "claude-code", "session_id": "s1", "payload": {"tool_name": "Read", "tool_input": {"file_path": "/home/op/project/lib.rs"}}}),
                json!({"at_ns": now, "event": "SessionEnd", "provider": "claude-code", "session_id": "s2", "payload": {}}),
            ];
            std::fs::write(
                spool.join("session-events.ndjson"),
                events
                    .iter()
                    .map(|event| format!("{event}\n"))
                    .collect::<String>(),
            )
            .unwrap();
        },
        1,
        2,
    );
    let running = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1",
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    let outcome = &running["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "failed", "{running}");
    assert_eq!(outcome["error"]["kind"], "refused", "{running}");
    let message = outcome["error"]["message"].as_str().unwrap();
    assert!(message.starts_with("session_running:"), "{message}");
    assert!(
        message.contains("polylogue claude-code:s1 read --follow"),
        "{message}"
    );
    assert!(
        message.contains("polylogue claude-code:s1 send 'TEXT'"),
        "{message}"
    );
    let ended = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s2", "dry_run": true,
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    // An ended session is not refused as running; this one recorded no
    // directory, so its harness resume is refused for that
    // (docs/session-control.md, the resume directory).
    assert_eq!(
        ended["reply"]["steps"][0]["outcome"]["refusal"]["message"],
        "directory_unknown: the session recorded no directory to resume in; continue with context instead: polylogue session:claude-code:s2 brief"
    );
}
