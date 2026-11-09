//! `polylogue hook` never blocks the harness: a hook that cannot start
//! writes nothing to stdout, one line to stderr and exits 0.

use std::process::{Output, Stdio};

fn run(mut command: std::process::Command) -> Output {
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    std::io::Write::write_all(child.stdin.as_mut().unwrap(), br#"{"session_id": "s1"}"#).unwrap();
    child.wait_with_output().unwrap()
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
    command.args(["hook", "SessionStart"]);
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
        .args(["hook", "Stop"]);
    assert_quiet_success(&run(command));
}

#[test]
fn unknown_harness_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let mut command = crate::command(dir.path());
    command.args(["hook", "Stop", "--harness", "no-such-harness"]);
    assert_quiet_success(&run(command));
}
