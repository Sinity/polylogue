use super::*;
use serde_json::{Value, json};
use std::io::Write;
use std::os::unix::net::UnixListener;

#[test]
fn gemini_hooks_install_and_uninstall_through_the_public_command() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("settings.json");
    let original = json!({"model":"synthetic", "hooks":{"BeforeAgent":[{"hooks":[{"type":"command","command":"other"}]}]}});
    std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut installed = None;
    for action in ["install", "install", "uninstall"] {
        let out = crate::command(dir.path())
            .args([
                "hooks",
                action,
                "--harness",
                "gemini-cli",
                "--settings",
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let value: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        if action == "install" {
            assert_eq!(value["model"], "synthetic");
            assert_eq!(
                value["hooks"]["SessionStart"][0]["hooks"][0]["command"],
                "polylogue hook SessionStart --harness gemini-cli"
            );
            assert_eq!(
                value["hooks"]["SessionStart"][0]["hooks"][0]["timeout"],
                10000
            );
            if let Some(previous) = installed.replace(value.clone()) {
                assert_eq!(value, previous);
            }
        } else {
            assert_eq!(value, original);
        }
    }
}

#[test]
fn manual_runs_offline_as_text_and_json() {
    let dir = tempfile::tempdir().unwrap();
    let archive = dir.path().join("absent-archive");
    let plain = cli(
        &crate::Fixture::new(
            &archive,
            &dir.path().join("home"),
            &dir.path().join("config"),
        ),
        &["manual"],
    );
    assert!(plain.status.success(), "{plain:?}");
    let text = String::from_utf8(plain.stdout).unwrap();
    assert!(text.contains("polylogue"));
    assert!(text.contains("sessions where"));
    let structured = cli(
        &crate::Fixture::new(
            &archive,
            &dir.path().join("home"),
            &dir.path().join("config"),
        ),
        &["--json", "manual"],
    );
    assert!(structured.status.success(), "{structured:?}");
    assert_eq!(json(&structured)["manual"], text);
    assert!(!archive.exists());
}

#[test]
fn gemini_start_and_prompt_events_request_context_through_respond() {
    for (event, payload, expected) in [
        (
            "SessionStart",
            json!({"session_id":"current","cwd":"/synthetic/project","source":"resume"}),
            json!({"ask":"session_start","cwd":"/synthetic/project","session":"gemini-cli:current","source":"resume"}),
        ),
        (
            "BeforeAgent",
            json!({"session_id":"current","prompt":"Continue https://chatgpt.com/c/previous"}),
            json!({"ask":"takeover","session":"chatgpt:previous","for_session":"gemini-cli:current"}),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        let home = dir.path().join("home");
        std::fs::create_dir(&archive).unwrap();
        let listener = UnixListener::bind(polylogue_store::Archive::at(&archive).socket()).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["op"], "context");
            assert_eq!(request["sender"], "hook");
            assert_eq!(request["ask"], expected);
            writeln!(stream, "{}", json!({"result":"ok","data":{"delivery":"synthetic-delivery","prepared":{"empty":false,"text":"Synthetic context"}}})).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            line.clear();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let request: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(request["op"], "delivery_event");
            assert_eq!(request["delivery"], "synthetic-delivery");
            assert_eq!(request["what"], "delivered");
            writeln!(stream, "{}", json!({"result":"ok","data":{}})).unwrap();
        });
        let mut child = crate::command(dir.path())
            .args(["hook", event, "--harness", "gemini-cli"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&serde_json::to_vec(&payload).unwrap())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{out:?}");
        assert_eq!(
            json(&out),
            json!({"hookSpecificOutput":{"hookEventName":event,"additionalContext":"Synthetic context"}})
        );
        server.join().unwrap();
    }
}
