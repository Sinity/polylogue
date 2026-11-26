//! The browser host through the real binary: `install chrome` writes
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
    let out = run(&["install", "chrome"]);
    assert!(!out.status.success(), "an extension ID is required");
    let out = run(&["install", "chrome", "--extension-id", "abcdefghijklmnop"]);
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
    assert_eq!(
        manifest["path"].as_str().unwrap(),
        std::fs::canonicalize(crate::BIN).unwrap().to_str().unwrap()
    );
    // Only the existing browser's directory is written.
    assert!(!config.join("google-chrome").exists());
    let out = run(&["install", "chrome", "--remove"]);
    assert!(out.status.success());
    assert!(!path.exists());
    let out = run(&["install", "unknown-harness"]);
    assert!(!out.status.success());
}

#[test]
fn started_as_chrome_starts_it_the_binary_is_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    )
    .native_command()
    .arg("chrome-extension://abcdefghijklmnop/")
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
    let dir = tempfile::tempdir().unwrap();
    let environment = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let command = || environment.command();
    let start = || crate::cli::start_daemon(&environment);
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
    crate::cli::wait_status(&environment, |status| status["store"]["messages"] == 1);
    let note = json!({"request_id":"019c6e27-e55b-73d1-87d8-4e01f1f75043","body":"My edited annotation","quote":"Exact  selected\nquote",
        "source":{"provider":"chatgpt","conversation_id":"passage","message_id":"native-message","page_url":"https://chatgpt.com/c/passage","selected_at":"2026-10-24T00:00:00Z"}});
    let mut missing = note.clone();
    missing["request_id"] = json!("019c6e27-e55b-73d1-87d8-4e01f1f75044");
    missing["source"]["message_id"] = json!("not-captured");
    let (status, refused) = native("/v1/notes", &missing);
    assert_eq!(status, 409, "{refused}");
    let (status, saved) = native("/v1/notes", &note);
    assert_eq!(status, 200, "{saved}");
    assert_eq!(saved["note"]["repeat"], false);
    drop(daemon);
    let _daemon = start();
    let (status, retried) = native("/v1/notes", &note);
    assert_eq!(status, 200, "{retried}");
    assert_eq!(retried["note"]["repeat"], true);
    let output = command()
        .args([
            "--json",
            r#"annotations where about:"session:chatgpt:passage""#,
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let answer: Value = serde_json::from_slice(&output.stdout).unwrap();
    let listing = crate::cli::listing(answer["value"].clone());
    let notes = &listing["rows"];
    assert_eq!(notes.as_array().unwrap().len(), 1);
    assert_eq!(notes[0]["id"], note["request_id"]);
    assert_eq!(notes[0]["author"], "operator");
    let value = &notes[0]["value"];
    assert_eq!(
        value,
        &json!(
            "My edited annotation\n\nQuoted selection:\nExact  selected\nquote\n\nSource: https://chatgpt.com/c/passage\nSelected at: 2026-10-24T00:00:00Z"
        )
    );
    let mut changed = note.clone();
    changed["body"] = json!("a different annotation under the same draft ID");
    let (status, refused) = native("/v1/notes", &changed);
    assert_eq!(status, 409, "{refused}");
}

/// Web chats driven as a person drives them: `send`, `answer`, `start --in`
/// and `import PROVIDER` on the command line against a running daemon, and
/// the popup's Allow through the host's answer route.
#[test]
fn web_chat_acts_run_through_the_command_line_and_the_popup() {
    use serde_json::{Value, json};
    let dir = tempfile::tempdir().unwrap();
    let environment = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let _daemon = crate::cli::start_daemon(&environment);
    let cli = |args: &[&str]| -> String {
        let output = environment.command().args(args).output().unwrap();
        assert!(output.status.success(), "{args:?}: {output:?}");
        String::from_utf8(output.stdout).unwrap()
    };
    let json_of = |args: &[&str]| -> Value {
        let mut all = vec!["--json"];
        all.extend(args);
        serde_json::from_str(&cli(&all)).unwrap()
    };
    let found = |value: &Value, key: &str, wanted: &str| {
        value
            .to_string()
            .contains(&format!("\"{key}\":\"{wanted}\""))
    };

    // Settings are written once the first index build has finished.
    crate::cli::wait_status(&environment, |_| true);
    cli(&["set", "browser.actions", "on"]);
    let held = json_of(&[
        "session:chatgpt:conv-1",
        "send",
        "a turn the operator approves",
    ]);
    assert!(found(&held, "status", "awaiting_approval"), "{held}");
    let allowed = json_of(&["session:chatgpt:conv-1", "answer", "allow"]);
    assert!(found(&allowed, "status", "queued"), "{allowed}");

    let draft = json_of(&[
        "start",
        "a new draft",
        "--in",
        "chatgpt",
        "--model",
        "GPT X",
    ]);
    assert!(
        found(&draft, "status", "queued") && draft.to_string().contains("\"submit\":false"),
        "{draft}"
    );
    let submitted = json_of(&["start", "a new conversation", "--in", "chatgpt", "--submit"]);
    assert!(
        submitted.to_string().contains("\"needs_approval\":false"),
        "{submitted}"
    );

    // The popup's Deny reaches the same answer.
    let other = json_of(&[
        "https://chatgpt.com/c/conv-2",
        "send",
        "a turn the operator denies",
    ]);
    let id = other
        .to_string()
        .split("\"id\":\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_owned();
    let mut child = environment
        .command()
        .arg("chrome-extension://abcdefghijklmnop/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let body = br#"{"value":"deny"}"#;
    let mut input = frame(&json!({"type":"request","version":1,"method":"POST",
        "path": format!("/v1/chat-actions/{id}/answer"), "headers":{"content-type":"application/json"}}));
    use base64::Engine as _;
    input.extend(frame(&json!({"type":"body","sequence":0,"data":base64::engine::general_purpose::STANDARD.encode(body)})));
    input.extend(frame(&json!({"type":"body_end"})));
    for _ in 0..4 {
        input.extend(frame(&json!({"type":"response_next"})));
    }
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let output = child.wait_with_output().unwrap();
    let response = frames(&output.stdout)
        .into_iter()
        .find(|v| v["type"] == "response")
        .unwrap();
    assert_eq!(response["status"], 200, "{output:?}");

    let job = json_of(&["import", "chatgpt", "--since", "2026-01-01"]);
    assert!(found(&job, "status", "waiting"), "{job}");
    let cancelled = json_of(&["import", "chatgpt", "--cancel"]);
    assert!(found(&cancelled, "status", "cancelled"), "{cancelled}");
    let help = cli(&["help", "backfill"]);
    assert!(help.contains("import PROVIDER"), "{help}");
}
