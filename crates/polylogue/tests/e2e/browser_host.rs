//! The browser host through the real binary: `install-agent chrome` writes
//! and removes the host manifest, and the binary started as Chrome starts
//! it (with the extension's origin) answers a native messaging exchange.

use std::io::Write;
use std::process::Stdio;

fn frame(v: &serde_json::Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(v).unwrap();
    let mut out = (bytes.len() as u32).to_ne_bytes().to_vec();
    out.extend(bytes);
    out
}

fn frames(mut bytes: &[u8]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    while bytes.len() >= 4 {
        let len = u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as usize;
        out.push(serde_json::from_slice(&bytes[4..4 + len]).unwrap());
        bytes = &bytes[4 + len..];
    }
    out
}

#[test]
fn install_agent_chrome_writes_and_removes_the_host_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config");
    std::fs::create_dir_all(config.join("chromium")).unwrap();
    let run = |args: &[&str]| {
        crate::command(dir.path())
            .env("XDG_CONFIG_HOME", &config)
            .args(args)
            .output()
            .unwrap()
    };
    let out = run(&["install-agent", "chrome"]);
    assert!(!out.status.success(), "an extension ID is required");
    let out = run(&[
        "install-agent",
        "chrome",
        "--extension-id",
        "abcdefghijklmnop",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let path = config.join("chromium/NativeMessagingHosts/com.polylogue.browser_capture.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        manifest["allowed_origins"][0],
        "chrome-extension://abcdefghijklmnop/"
    );
    assert_eq!(manifest["type"], "stdio");
    assert!(std::path::Path::new(manifest["path"].as_str().unwrap()).is_absolute());
    // Only the existing browser's directory is written.
    assert!(!config.join("google-chrome").exists());
    let out = run(&["install-agent", "chrome", "--remove"]);
    assert!(out.status.success());
    assert!(!path.exists());
    let out = run(&["install-agent", "codex"]);
    assert!(!out.status.success());
}

#[test]
fn started_as_chrome_starts_it_the_binary_is_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = crate::command(dir.path())
        .arg("chrome-extension://abcdefghijklmnop/")
        .env("POLYLOGUE_ARCHIVE", dir.path().join("archive"))
        .env("POLYLOGUE_HOME", dir.path().join("home"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = Vec::new();
    input.extend(frame(&serde_json::json!({
        "type": "request", "version": 1, "method": "GET", "path": "/v1/status", "headers": {}
    })));
    input.extend(frame(&serde_json::json!({ "type": "body_end" })));
    for _ in 0..4 {
        input.extend(frame(&serde_json::json!({ "type": "response_next" })));
    }
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let got = frames(&out.stdout);
    assert_eq!(got[0]["type"], "response");
    assert_eq!(got[0]["status"], 200);
    assert_eq!(got.last().unwrap()["type"], "response_end");
    let body: Vec<u8> = got
        .iter()
        .filter(|f| f["type"] == "response_body")
        .flat_map(|f| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(f["data"].as_str().unwrap())
                .unwrap()
        })
        .collect();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["daemon"], "no_answer");
}

#[test]
fn a_native_selection_note_is_readable_by_cli_after_daemon_restart() {
    use base64::Engine as _;
    use serde_json::{Value, json};
    use std::io::{BufRead, BufReader};
    struct OwnedDaemon(std::process::Child);
    impl Drop for OwnedDaemon {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let state = dir.path().join("state");
    let command = || {
        let mut command = crate::command(dir.path());
        command
            .env("POLYLOGUE_HOME", &home)
            .env("POLYLOGUE_ARCHIVE", &archive)
            .env("XDG_STATE_HOME", &state);
        command
    };
    let start = || {
        let mut child =
            polylogue_testkit::spawn_daemon(command().arg("daemon").stderr(Stdio::piped()))
                .unwrap();
        let mut line = String::new();
        BufReader::new(child.stderr.as_mut().unwrap())
            .read_line(&mut line)
            .unwrap();
        assert!(line.contains("serving"), "{line}");
        OwnedDaemon(child)
    };
    let native = |path: &str, body: &Value| -> (u64, Value) {
        let mut child = command()
            .arg("chrome-extension://abcdefghijklmnop/")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let bytes = serde_json::to_vec(body).unwrap();
        let mut input = frame(
            &json!({"type":"request","version":1,"method":"POST","path":path,"headers":{"content-type":"application/json"}}),
        );
        input.extend(frame(&json!({"type":"body","sequence":0,"data":base64::engine::general_purpose::STANDARD.encode(bytes)})));
        input.extend(frame(&json!({"type":"body_end"})));
        for _ in 0..4 {
            input.extend(frame(&json!({"type":"response_next"})));
        }
        child.stdin.take().unwrap().write_all(&input).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let messages = frames(&output.stdout);
        let response = messages.iter().find(|v| v["type"] == "response").unwrap();
        let mut body = vec![];
        for part in messages.iter().filter(|v| v["type"] == "response_body") {
            body.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(part["data"].as_str().unwrap())
                    .unwrap(),
            );
        }
        (
            response["status"].as_u64().unwrap(),
            serde_json::from_slice(&body).unwrap(),
        )
    };
    let daemon = start();
    let capture = json!({"schema_version":1,"polylogue_capture_kind":"browser_llm_session",
        "provenance":{"captured_at":"2026-10-24T00:00:00Z","source_url":"https://chatgpt.com/c/passage"},
        "session":{"provider":"chatgpt","provider_session_id":"passage","turns":[{"provider_turn_id":"native-message","role":"assistant","text":"Exact  selected\nquote"}]}});
    assert_eq!(native("/v1/captures", &capture).0, 200);
    loop {
        let status = command().args(["status", "--json"]).output().unwrap();
        assert!(status.status.success(), "{status:?}");
        let status: Value = serde_json::from_slice(&status.stdout).unwrap();
        if status["store"]["messages"] == 1 {
            break;
        }
    }
    let note = json!({"request_id":"019c6e27-e55b-73d1-87d8-4e01f1f75043","body":"My edited annotation","quote":"Exact  selected\nquote",
        "source":{"provider":"chatgpt","conversation_id":"passage","message_id":"native-message","page_url":"https://chatgpt.com/c/passage","selected_at":"2026-10-24T00:00:00Z"}});
    let (status, saved) = native("/v1/notes", &note);
    assert_eq!(status, 200, "{saved}");
    assert_eq!(saved["note"]["repeat"], false);
    drop(daemon);
    let _daemon = start();
    let (status, retried) = native("/v1/notes", &note);
    assert_eq!(status, 200, "{retried}");
    assert_eq!(retried["note"]["repeat"], true);
    let output = command()
        .args(["note", "list", "session:chatgpt:passage", "--json"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let answer: Value = serde_json::from_slice(&output.stdout).unwrap();
    let notes = &answer["rows"];
    assert_eq!(notes.as_array().unwrap().len(), 1);
    assert_eq!(notes[0]["id"], note["request_id"]);
    assert_eq!(notes[0]["author"], "operator");
    let value: Value = serde_json::from_str(notes[0]["value"].as_str().unwrap()).unwrap();
    assert_eq!(value["body"], note["body"]);
    assert_eq!(value["quote"], note["quote"]);
    assert_eq!(value["source"], note["source"]);
}
