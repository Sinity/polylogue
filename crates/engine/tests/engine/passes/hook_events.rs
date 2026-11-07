//! Hook events through the loop: they attach to sessions that transcripts
//! and exports provide, and each hook invocation is one event however often
//! its file is stored.

use super::*;

/// The hook command's line for one invocation.
fn hook_line(at_ns: u64, event: &str, session: &str, payload: serde_json::Value) -> String {
    let mut payload = payload;
    payload["session_id"] = session.into();
    let line = serde_json::json!({
        "event_id": uuid::Uuid::now_v7().to_string(),
        "at_ns": at_ns.to_string(), "event": event, "harness_pid": 4242,
        "provider": "claude-code", "session_id": session, "payload": payload,
    });
    format!("{line}\n")
}

fn hook_spool(w: &ClaudeHome) -> std::path::PathBuf {
    let spool = w
        .config(64 << 20)
        .spool_dir(polylogue_sources::hook_spool::ORIGIN);
    fs::create_dir_all(&spool).unwrap();
    spool
}

#[test]
fn a_hook_event_never_creates_a_session() {
    let w = ClaudeHome::new();
    let spool = hook_spool(&w);
    let nested = user("n1", "nested", "a transcript inside a payload");
    let lines = hook_line(
        1_790_856_000_000_000_000,
        "SessionStart",
        "h1",
        serde_json::json!({"cwd": "/proj"}),
    ) + &hook_line(
        1_790_856_001_000_000_000,
        "PreCompact",
        "h1",
        serde_json::json!({"transcript": [serde_json::from_str::<serde_json::Value>(nested.trim()).unwrap()]}),
    );
    fs::write(spool.join("4242.ndjson"), lines).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE stub = 0"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:nested'"),
        0,
        "a transcript inside a payload was read as a session"
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 0);
    let on_h1 = "SELECT count(*) AS n FROM event e JOIN session s ON s.pk = e.session_pk \
                 WHERE s.id = 'claude-code:h1' AND e.kind LIKE 'hook_%'";
    assert_eq!(w.count(on_h1), 2);

    w.append(
        "h1.jsonl",
        &(user("u1", "h1", "hello") + &assistant("a1", "h1", "hi")),
    );
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("h1"), ["hello", "hi"]);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE stub = 0"),
        1,
        "the transcript made the session"
    );
    assert_eq!(w.count(on_h1), 2, "the events attached to it");
}

#[test]
fn a_stored_hook_line_is_one_event_however_often_it_is_stored() {
    let w = ClaudeHome::new();
    let spool = hook_spool(&w);
    w.append("h2.jsonl", &user("u1", "h2", "run it"));
    // Two invocations with exactly the same payload and timestamp.
    let payload = serde_json::json!({"tool_name": "Bash", "tool_input": {"command": "ls"}});
    let lines = hook_line(
        1_790_856_000_000_000_001,
        "PreToolUse",
        "h2",
        payload.clone(),
    ) + &hook_line(1_790_856_000_000_000_001, "PreToolUse", "h2", payload);
    fs::write(spool.join("4242.ndjson"), &lines).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    e.run_pass().unwrap();
    let events = "SELECT count(*) AS n FROM event WHERE kind = 'hook_pre_tool_use'";
    assert_eq!(w.count(events), 2, "two invocations are two events");

    // The same file stored again, under another name and as the same file
    // rewritten with identical bytes.
    fs::write(spool.join("4243.ndjson"), &lines).unwrap();
    fs::write(spool.join("4242.ndjson"), &lines).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.count(events), 2, "storing the lines again added events");
}
