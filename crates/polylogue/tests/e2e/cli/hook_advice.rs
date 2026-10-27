//! Advice preparation and delivery through a real daemon.

use super::*;
use serde_json::json;

/// A Claude Code session in `cwd` that ran `cargo test` twice; both failed.
fn failing_session(cwd: &str, runs: i64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let at = |seconds: i64| {
        polylogue_model::Timestamp::from_unix_ms(now - 600_000 + seconds * 1000)
            .unwrap()
            .to_string()
    };
    let mut lines = vec![
        json!({"type": "user", "uuid": "u0", "sessionId": "advised", "cwd": cwd,
        "timestamp": at(0), "message": {"role": "user", "content": "run the tests"}}),
    ];
    for n in 1..=runs {
        lines.push(
            json!({"type": "assistant", "uuid": format!("a{n}"), "sessionId": "advised", "cwd": cwd,
            "timestamp": at(n * 10),
            "message": {"id": format!("msg_a{n}"), "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": format!("call{n}"), "name": "Bash",
                             "input": {"command": "cargo test"}}]}}),
        );
        lines.push(json!({"type": "user", "uuid": format!("r{n}"), "sessionId": "advised", "cwd": cwd,
            "timestamp": at(n * 10 + 5),
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": format!("call{n}"),
                "is_error": true, "content": "test result: FAILED. 1 failed"}]}}));
    }
    lines.iter().map(|l| format!("{l}\n")).collect()
}

fn pre_tool_use(env: &[(&str, &Path)], cwd: &str, command: &str) -> String {
    let mut hook = Command::new(crate::BIN);
    hook.args(["hook", "PreToolUse"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped());
    for (key, value) in env {
        hook.env(key, value);
    }
    let mut child = hook.spawn().unwrap();
    let payload = json!({"session_id": "next", "cwd": cwd, "hook_event_name": "PreToolUse",
        "tool_name": "Bash", "tool_input": {"command": command}});
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.to_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{out:?}");
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn a_check_that_keeps_failing_here_is_named_before_it_runs_again() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let project = home.join(".claude/projects/-work-proj");
    std::fs::create_dir_all(&project).unwrap();
    let cwd = "/work/proj";
    std::fs::write(project.join("advised.jsonl"), failing_session(cwd, 2)).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let _daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["tool_calls"].as_u64() >= Some(2));
    let socket = archive.join("daemon.sock");
    let ask = || {
        json!({"op":"context", "request_id":polylogue_store::user_writes::new_request_id(),
        "ask":{"ask":"tool_use", "cwd":cwd, "session":"claude-code:next", "command":"cargo test"}})
    };
    let send = |request: serde_json::Value| {
        let mut stream = std::os::unix::net::UnixStream::connect(&socket).unwrap();
        writeln!(stream, "{request}").unwrap();
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        let response: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(response["result"], "ok", "{response}");
        response["data"].clone()
    };
    let prepared = send(ask());
    let text = prepared["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("`cargo test` failed 2 times here"), "{text}");
    assert!(text.contains("claude-code:advised"), "{text}");
    let count = || {
        json(&cli(&env, &["--json", "query", "deliveries"]))["count"]
            .as_u64()
            .unwrap()
    };
    assert_eq!(count(), 0);
    let record = || {
        json!({"op":"advice_delivered", "request_id":prepared["delivery"],
        "session":"claude-code:next", "cwd":cwd, "text":text})
    };
    send(record());
    send(record());
    assert_eq!(count(), 1);
    assert_eq!(send(ask())["prepared"]["empty"], true);
    let outcome = json(&cli(&env, &["--json", "query", "deliveries"]));
    assert_eq!(outcome["rows"][0]["outcome"], "delivered");
    std::fs::write(project.join("advised.jsonl"), failing_session(cwd, 3)).unwrap();
    wait_status(&env, |s| s["store"]["tool_calls"].as_u64() >= Some(3));
    assert_eq!(send(ask())["prepared"]["empty"], false);
    // A command that runs no verification gets nothing.
    assert_eq!(pre_tool_use(&env, cwd, "ls -la"), "");
}
