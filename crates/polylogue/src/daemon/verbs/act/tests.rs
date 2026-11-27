use crate::test_daemon::{self, Call};
use serde_json::{Value, json};

#[test]
fn send_many_reaches_running_only() {
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
    let answer = Call::new(
        "rows",
        json!({
            "references": ["session:claude-code:s1", "session:claude-code:s2"],
            "steps": [{"verb": "send", "text": "read the 'clock' file"}],
        }),
    )
    .id("group-send")
    .answer(&d.daemon);
    assert_eq!(answer.exit_code(), 5);
    let (report, missing) = crate::test_answers::degraded(answer);
    assert_eq!(missing, ["message to claude-code:s2"]);
    let sent = &report["reply"]["steps"][0]["outcome"];
    assert_eq!(sent["status"], "done", "{report}");
    assert_eq!(
        sent["reply"]["reply"]["reached"].as_array().unwrap().len(),
        1
    );
    assert_eq!(
        sent["reply"]["reply"]["reached"][0]["deliveries"][0]["session"],
        "claude-code:s1"
    );
    assert_eq!(
        sent["reply"]["reply"]["reached"][0]["deliveries"][0]["channel"],
        "hook"
    );
    assert_eq!(
        sent["reply"]["reply"]["skipped"],
        json!([{
            "session": "claude-code:s2",
            "reason": {"kind": "session_not_running", "state": "ended"},
            "line": format!("polylogue claude-code:s2 send {}", polylogue_render::shell_quote("read the 'clock' file")),
        }])
    );
    let listed = crate::test_answers::empty(
        Call::new(
            "rows",
            json!({"selection": "deliveries where recipient:claude-code:s2"}),
        )
        .answer(&d.daemon),
    );
    assert_eq!(
        listed["reply"]["count"], 0,
        "an ended recipient has no delivery to resume"
    );
    assert_eq!(d.daemon.acts.hosted_sessions().len(), 0);
}

#[test]
fn acts_dry_run_reports_channels_and_permissions() {
    let d = test_daemon::start_with_clock(
        "ended session",
        polylogue_model::WallClock::fixed("2026-03-02T10:00:00Z".parse().unwrap()),
    );
    let report = crate::test_answers::ok(Call::new("rows", json!({
        "selection": "session:claude-code:s1", "dry_run": true,
        "steps": [{"verb": "send", "text": "continue"}, {"verb": "answer", "value": "allow"}, {"verb": "interrupt"}],
    })).id("acts-dry").answer(&d.daemon));
    let steps = report["reply"]["steps"].as_array().unwrap();
    assert_eq!(
        steps[0]["outcome"]["would"],
        "would send \"continue\" to claude-code:s1: resumed headlessly"
    );
    assert_eq!(
        steps[1]["outcome"]["would"],
        "would answer allow for claude-code:s1: refused: session_not_live: the session is ended; nothing waits for an answer"
    );
    assert_eq!(
        steps[2]["outcome"]["would"],
        "would interrupt claude-code:s1: refused: session_not_live: the session is ended; no turn runs"
    );
    let deliveries = crate::test_answers::empty(
        Call::new("rows", json!({"selection": "deliveries"})).answer(&d.daemon),
    );
    assert_eq!(deliveries["reply"]["count"], Value::from(0));
}
