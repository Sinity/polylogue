//! `polylogue internal hook` never blocks the harness: a hook that cannot start
//! writes nothing to stdout, one line to stderr and exits 0.

use base64::Engine as _;
use std::process::{Output, Stdio};

fn run(command: std::process::Command) -> Output {
    run_with_payload(command, br#"{"session_id": "s1"}"#)
}

fn run_with_payload(mut command: std::process::Command, payload: &[u8]) -> Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), payload).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn hook_with_lone_surrogate_payload_keeps_its_bytes() {
    for payload in [
        br#"{"session_id":"s1","prompt":"\uD800"}"#.as_slice(),
        b"{\"session_id\":\"s1\",\"prompt\":\"\xff\"}\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut command = crate::command(dir.path());
        let state = dir.path().join("state");
        command.env("XDG_STATE_HOME", &state);
        command.args(["internal", "hook", "UserPromptSubmit"]);
        let output = run_with_payload(command, payload);
        assert_quiet_success(&output);
        let paths: Vec<_> = std::fs::read_dir(state.join("polylogue/hooks"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        let [path] = &paths[..] else {
            panic!("unexpected spool files: {paths:?}")
        };
        let record: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(record["event"], "UserPromptSubmit");
        assert_eq!(record["provider"], "claude-code");
        let unreadable = &record["unreadable_payload"];
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(unreadable["bytes_base64"].as_str().unwrap())
            .unwrap();
        assert_eq!(bytes, payload);
        assert!(
            unreadable["reason"]
                .as_str()
                .unwrap()
                .starts_with("hook payload is not JSON:")
        );
        assert!(record.get("payload").is_none());
        assert!(record.get("session_id").is_none());
    }
}

fn assert_quiet_success(output: &Output) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "stderr: {stderr}");
    assert!(output.stdout.is_empty(), "stdout: {:?}", output.stdout);
    assert_eq!(stderr.lines().count(), 1, "stderr: {stderr}");
    assert!(stderr.starts_with("polylogue hook: "), "stderr: {stderr}");
}

#[test]
fn malformed_config_exits_zero_with_empty_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let deployment = dir.path().join("config/polylogue/polylogue.toml");
    std::fs::create_dir_all(deployment.parent().unwrap()).unwrap();
    std::fs::write(&deployment, "archive = [unterminated\n").unwrap();
    let mut command = crate::command(dir.path());
    command.args(["internal", "hook", "SessionStart"]);
    assert_quiet_success(&run(command));
}

#[test]
fn deleted_cwd_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let gone = dir.path().join("gone");
    std::fs::create_dir(&gone).unwrap();
    let hook = crate::command(dir.path());
    let mut command = std::process::Command::new("sh");
    command
        .env_clear()
        .envs(hook.get_envs().filter_map(|(k, v)| Some((k, v?))))
        .arg("-c")
        .arg(r#"cd "$1" && rmdir "$1" && shift && exec "$@""#)
        .arg("sh")
        .arg(&gone)
        .arg(hook.get_program())
        .args(hook.get_args())
        .args(["internal", "hook", "Stop"]);
    assert_quiet_success(&run(command));
}

#[test]
fn unknown_harness_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let mut command = crate::command(dir.path());
    command.args(["internal", "hook", "Stop", "--harness", "no-such-harness"]);
    assert_quiet_success(&run(command));
}
