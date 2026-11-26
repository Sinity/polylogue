use super::{cli, json, line, query_rows, start_daemon_with, wait_status};
use serde_json::{Value, json};
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;

fn fixture(root: &Path, harness: &str, script: &str) -> (crate::Fixture, super::Daemon) {
    let home = root.join("home");
    let sessions = home.join(".claude/projects/-morrow");
    std::fs::create_dir_all(&sessions).unwrap();
    let mut message: Value = serde_json::from_str(&line("u1", "s1", "Plan the release")).unwrap();
    message["cwd"] = json!(root.join("missing-checkout"));
    std::fs::write(sessions.join("s1.jsonl"), format!("{message}\n")).unwrap();

    let bin = root.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let program = bin.join(harness);
    std::fs::write(&program, script).unwrap();
    std::fs::set_permissions(program, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = std::env::join_paths(
        std::iter::once(bin).chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
    )
    .unwrap();
    let env = crate::Fixture::new(&root.join("archive"), &home, &root.join("config"));
    let mut command = env.command();
    command.env("PATH", path);
    let daemon = start_daemon_with(&env, command);
    wait_status(&env, |status| status["store"]["messages"] == 1);
    (env, daemon)
}

fn step(value: &Value) -> &Value {
    let outcome = &value["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "done", "{value}");
    &outcome["reply"]["reply"]
}

#[test]
fn hosted_start_reports_the_harness_exit_and_records_why_it_ended() {
    let root = tempfile::tempdir().unwrap();
    let (env, _daemon) = fixture(
        root.path(),
        "claude",
        "#!/bin/sh\nprintf '%s\\n' 'sign-in required' >&2\nexit 3\n",
    );
    let out = cli(
        &env,
        &["--json", "start", "Plan the release", "--in", "claude-code"],
    );
    assert_eq!(out.status.code(), Some(4), "{out:?}");
    let failure = json(&out);
    assert_eq!(failure["error"]["kind"], "io");
    assert_eq!(
        failure["error"]["message"],
        "claude_stream_json: exit status: 3: sign-in required"
    );

    let ended = || {
        query_rows(&cli(
            &env,
            &[
                "--json",
                "--fields",
                "session,kind,data",
                "events where kind:registry_hosted_ended",
            ],
        ))
    };
    wait_status(&env, |_| ended()["count"] == 1);
    let events = ended();
    let event = &events["rows"][0];
    assert_eq!(event["kind"], "registry_hosted_ended");
    assert_eq!(event["data"]["detail"], "exit status: 3: sign-in required");
    let session = format!("session:{}", event["session"].as_str().unwrap());
    let out = cli(&env, &[&session, "read", "--view", "live"]);
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    assert!(
        String::from_utf8(out.stdout)
            .unwrap()
            .contains("exit status: 3: sign-in required")
    );
}

#[test]
fn send_to_a_gone_directory_reports_not_sent_and_exits_five() {
    let root = tempfile::tempdir().unwrap();
    let (env, _daemon) = fixture(root.path(), "claude", "#!/bin/sh\nexit 99\n");
    let out = cli(
        &env,
        &["--json", "session:claude-code:s1", "send", "continue"],
    );
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    let value = json(&out);
    let row = &step(&value)["reached"][0]["deliveries"][0];
    assert_eq!(row["session"], "claude-code:s1");
    assert_eq!(row["outcome"], "not_sent");
    assert_eq!(row["channel"], "resume");
    assert_eq!(
        row["reason"],
        json!({"kind": "directory_missing", "path": root.path().join("missing-checkout")})
    );
}

#[test]
fn send_to_two_sessions_reports_the_skipped_state_and_exits_five() {
    let root = tempfile::tempdir().unwrap();
    let (env, _daemon) = fixture(
        root.path(),
        "claude",
        r#"#!/bin/sh
IFS= read -r line || exit 1
printf '%s\n' '{"type":"system","subtype":"init"}'
IFS= read -r line || exit 1
printf '%s\n' '{"type":"result"}'
"#,
    );
    let started = cli(&env, &["--json", "start", "Begin", "--in", "claude-code"]);
    assert_eq!(started.status.code(), Some(0), "{started:?}");
    let started = json(&started);
    let hosted = started["session"].as_str().unwrap();
    let host_reference = format!("session:{hosted}");
    wait_status(&env, |status| status["store"]["sessions"] == 2);
    let out = cli(
        &env,
        &[
            "--json",
            "--all",
            "session:claude-code:s1",
            &host_reference,
            "send",
            "continue",
        ],
    );
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    let value = json(&out);
    let report = step(&value);
    assert_eq!(
        report["skipped"],
        json!([{
            "session": "claude-code:s1",
            "reason": {"kind": "session_not_running", "state": "ended"},
            "line": "polylogue claude-code:s1 send continue"
        }])
    );
    assert_eq!(report["reached"].as_array().unwrap().len(), 1);
    assert_eq!(report["reached"][0]["deliveries"][0]["session"], hosted);
    assert_eq!(
        report["reached"][0]["deliveries"][0]["outcome"],
        "delivered"
    );
}

#[test]
fn an_answer_without_a_matching_option_reports_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let (env, _daemon) = fixture(
        root.path(),
        "gemini",
        r#"#!/bin/sh
while IFS= read -r line; do
    id=$(printf '%s' "$line" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
    case "$line" in
        *'"initialize"'*) printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":1}}\n' "$id" ;;
        *'"session/new"'*) printf '{"jsonrpc":"2.0","id":"%s","result":{"sessionId":"morrow"}}\n' "$id" ;;
        *'"session/prompt"'*)
            printf '%s\n' '{"jsonrpc":"2.0","id":7,"method":"session/request_permission","params":{"sessionId":"morrow","options":[{"optionId":"no","kind":"reject_once"}]}}' ;;
        *'"outcome":"cancelled"'*)
            printf '%s\n' "$line" > cancelled.json
            exit 0 ;;
    esac
done
"#,
    );
    let started = cli(&env, &["--json", "start", "Begin", "--in", "gemini-cli"]);
    assert_eq!(started.status.code(), Some(0), "{started:?}");
    assert_eq!(json(&started)["session"], "gemini-cli:morrow");
    wait_status(&env, |_| {
        query_rows(&cli(&env, &["--json", "sessions where live:waiting"]))["count"] == 1
    });
    let answer = cli(
        &env,
        &["--json", "session:gemini-cli:morrow", "answer", "allow"],
    );
    assert_eq!(answer.status.code(), Some(0), "{answer:?}");
    let value = json(&answer);
    let row = &step(&value)[0];
    assert_eq!(row["request"], "request:gemini-cli:morrow#7");
    assert_eq!(row["channel"], "protocol");
    assert_eq!(row["outcome"], "cancelled");
    assert_eq!(row["cancellation"], json!({"options": ["reject_once"]}));
    wait_status(&env, |_| root.path().join("cancelled.json").exists());
    let sent: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("cancelled.json")).unwrap())
            .unwrap();
    assert_eq!(
        sent,
        json!({"jsonrpc":"2.0","id":7,"result":{"outcome":{"outcome":"cancelled"}}})
    );
}

#[test]
fn retrying_a_continuation_with_a_failed_interrupt_starts_one_session() {
    let root = tempfile::tempdir().unwrap();
    let (env, _daemon) = fixture(
        root.path(),
        "claude",
        r#"#!/bin/sh
IFS= read -r line || exit 1
printf '%s\n' "$line" >> first-turns.jsonl
printf '%s\n' '{"type":"result"}'
"#,
    );
    let args = [
        "--json",
        "--request-id",
        "morrow-continue",
        "session:claude-code:s1",
        "resume",
        "--in",
        "claude-code",
        "--cwd",
        root.path().to_str().unwrap(),
        "--stop-source",
    ];
    let mut replies = Vec::new();
    for _ in 0..2 {
        let out = cli(&env, &args);
        assert_eq!(out.status.code(), Some(0), "{out:?}");
        let value = json(&out);
        let reply = &step(&value)[0];
        assert_eq!(reply["briefed_from"], "claude-code:s1");
        assert_eq!(reply["stop_source"]["error"]["kind"], "refused");
        replies.push(reply["session"].clone());
    }
    assert_eq!(replies[0], replies[1]);
    assert_eq!(
        std::fs::read_to_string(root.path().join("first-turns.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}
