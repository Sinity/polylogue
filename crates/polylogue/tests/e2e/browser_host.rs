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

fn native(environment: &crate::Fixture, message: &serde_json::Value) -> serde_json::Value {
    use base64::Engine as _;
    use serde_json::json;
    let mut child = environment
        .native_command()
        .arg("chrome-extension://abcdefghijklmnop/")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let bytes = serde_json::to_vec(message).unwrap();
    let mut input = Vec::new();
    if bytes.len() <= 64 * 1024 {
        input.extend(frame(message));
    } else {
        let count = bytes.len().div_ceil(64 * 1024);
        for (sequence, bytes) in bytes.chunks(64 * 1024).enumerate() {
            input.extend(frame(&json!({"version":2,"type":"part","sequence":sequence,
                "last":sequence + 1 == count,"data":base64::engine::general_purpose::STANDARD.encode(bytes)})));
        }
    }
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{out:?}");
    let got = frames(&out.stdout);
    for f in &got {
        assert_eq!(f["version"], 2);
        assert!(serde_json::to_vec(f).unwrap().len() < 1024 * 1024);
    }
    let response = if got[0]["type"] == "part" {
        let mut joined = Vec::new();
        for (sequence, part) in got.iter().enumerate() {
            assert_eq!(part["sequence"], sequence);
            assert_eq!(part["last"], sequence + 1 == got.len());
            joined.extend(
                base64::engine::general_purpose::STANDARD
                    .decode(part["data"].as_str().unwrap())
                    .unwrap(),
            );
        }
        serde_json::from_slice(&joined).unwrap()
    } else {
        assert_eq!(got.len(), 1);
        got[0].clone()
    };
    assert_eq!(response["type"], "answer");
    response["answer"].clone()
}

fn fixture(dir: &std::path::Path) -> crate::Fixture {
    crate::Fixture::new(&dir.join("archive"), &dir.join("home"), &dir.join("config"))
}
fn capture(id: &str, text: &str) -> serde_json::Value {
    serde_json::json!({"schema_version":1,"polylogue_capture_kind":"browser_llm_session",
        "provenance":{"captured_at":"2026-10-24T00:00:00Z","source_url":"https://chatgpt.com/c/example"},
        "session":{"provider":"chatgpt","provider_session_id":id,"turns":[{"provider_turn_id":"m1","role":"user","text":text}]}})
}

#[test]
fn capture_with_daemon_down_is_spooled() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let environment = fixture(dir.path());
    let body = capture("offline", "saved with the daemon down").to_string();
    let saved = native(
        &environment,
        &json!({"version":2,"type":"capture","capture":body}),
    );
    assert_eq!(saved["outcome"], "ok");
    assert_eq!(saved["value"]["new"], true);
    let spool = native(
        &environment,
        &json!({"version":2,"type":"spooled","provider":"chatgpt","ids":["offline","absent"]}),
    );
    assert_eq!(spool["value"]["spooled"], json!([true, false]));
    let answer = native(
        &environment,
        &json!({"version":2,"type":"call","operation":"internal conversations",
        "arguments":{"provider":"chatgpt","ids":["offline"]},"request_id":"offline-check"}),
    );
    assert_eq!(answer["error"]["code"], "daemon_required");
    assert_eq!(
        answer["error"]["reason"],
        json!({"name":"daemon_did_not_answer","fields":{}})
    );
}

#[test]
fn capture_over_one_megabyte_arrives_in_parts_and_is_stored_whole() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let environment = fixture(dir.path());
    let body = capture(
        "large",
        &"A long conversation with Unicode: 🦀 ".repeat(40_000),
    )
    .to_string();
    assert!(body.len() > 1024 * 1024);
    let saved = native(
        &environment,
        &json!({"version":2,"type":"capture","capture":body}),
    );
    assert_eq!(saved["outcome"], "ok");
    let captures: Vec<_> =
        std::fs::read_dir(dir.path().join("state/polylogue/browser-capture/chatgpt"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "json")
            })
            .collect();
    assert_eq!(captures.len(), 1);
    let stored = std::fs::read(&captures[0]).unwrap();
    let start =
        polylogue_model::capture_log::MAGIC.len() + polylogue_model::capture_log::HEADER_LEN;
    assert_eq!(&stored[start..], body.as_bytes());
    assert_eq!(
        saved["value"]["hash"],
        blake3::hash(body.as_bytes()).to_hex().as_str()
    );
}

#[test]
fn host_forwards_an_operation_by_name() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let environment = fixture(dir.path());
    let _daemon = crate::cli::start_daemon(&environment);
    crate::cli::wait_status(&environment, |_| true);
    let answer = native(
        &environment,
        &json!({"version":2,"type":"call","operation":"internal conversations",
        "arguments":{"provider":"chatgpt","ids":["not-captured"]},"request_id":"hidden-operation"}),
    );
    assert_eq!(answer["outcome"], "ok");
    assert_eq!(
        answer["value"],
        json!({"provider":"chatgpt","conversations":[{"id":"not-captured","session":"chatgpt:not-captured","messages":null,"downloading":null,"last_captured_at":null}]})
    );
    let refused = native(
        &environment,
        &json!({"version":2,"type":"call","operation":"internal capture-files",
        "arguments":{"capture":"{}"},"request_id":"invalid-capture"}),
    );
    assert_eq!(refused["outcome"], "error");
    assert_eq!(refused["error"]["reason"]["name"], "capture_unreadable");
}

#[test]
fn a_large_daemon_answer_arrives_in_parts() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let environment = fixture(dir.path());
    let _daemon = crate::cli::start_daemon(&environment);
    crate::cli::wait_status(&environment, |_| true);
    let ids: Vec<_> = (0..1000).map(|i| format!("conversation-{i}")).collect();
    let answer = native(
        &environment,
        &json!({"version":2,"type":"call","operation":"internal conversations",
        "arguments":{"provider":"chatgpt","ids":ids},"request_id":"large-answer"}),
    );
    assert_eq!(answer["outcome"], "ok");
    assert!(serde_json::to_vec(&answer).unwrap().len() > 64 * 1024);
    let expected: Vec<_> = ids
        .iter()
        .map(|id| {
            json!({"id":id,"session":format!("chatgpt:{id}"),
        "messages":null,"downloading":null,"last_captured_at":null})
        })
        .collect();
    assert_eq!(
        answer["value"],
        json!({"provider":"chatgpt","conversations":expected})
    );
}

#[test]
fn a_different_native_version_is_refused_with_a_reload_message() {
    use serde_json::json;
    let dir = tempfile::tempdir().unwrap();
    let answer = native(
        &fixture(dir.path()),
        &json!({"version":1,"type":"call","operation":"set","arguments":{}}),
    );
    assert_eq!(answer["outcome"], "error");
    assert_eq!(answer["error"]["code"], "refused");
    assert_eq!(
        answer["error"]["reason"],
        json!({"name":"native_version_mismatch","fields":{"version":1}})
    );
    assert_eq!(
        answer["error"]["message"],
        "the browser host needs version 2; reload the extension"
    );
}

#[test]
fn a_native_selection_note_is_readable_by_cli_after_daemon_restart() {
    use serde_json::{Value, json};
    let dir = tempfile::tempdir().unwrap();
    let environment = crate::Fixture::new(
        &dir.path().join("archive"),
        &dir.path().join("home"),
        &dir.path().join("config"),
    );
    let command = || environment.command();
    let start = || crate::cli::start_daemon(&environment);
    let native = |operation: &str, body: &Value| -> (u64, Value) {
        let message = if operation == "capture" {
            json!({"version":2,"type":"capture","capture":body.to_string()})
        } else {
            let mut args = body["source"].clone();
            args["body"] = body["body"].clone();
            args["quote"] = body["quote"].clone();
            json!({"version":2,"type":"call","operation":operation,"arguments":args,"request_id":body["request_id"]})
        };
        let answer = native(&environment, &message);
        if answer["outcome"] == "error" {
            let failure: command_surface::Failure =
                serde_json::from_value(answer["error"].clone()).unwrap();
            (
                u64::from(failure.kind.http_status().unwrap()),
                answer["error"].clone(),
            )
        } else {
            (200, answer["value"].clone())
        }
    };
    let daemon = start();
    let capture = json!({"schema_version":1,"polylogue_capture_kind":"browser_llm_session",
        "provenance":{"captured_at":"2026-10-24T00:00:00Z","source_url":"https://chatgpt.com/c/passage"},
        "session":{"provider":"chatgpt","provider_session_id":"passage","turns":[{"provider_turn_id":"native-message","role":"assistant","text":"Exact  selected\nquote"}]}});
    assert_eq!(native("capture", &capture).0, 200);
    crate::cli::wait_status(&environment, |status| status["store"]["messages"] == 1);
    let note = json!({"request_id":"019c6e27-e55b-73d1-87d8-4e01f1f75043","body":"My edited annotation","quote":"Exact  selected\nquote",
        "source":{"provider":"chatgpt","conversation_id":"passage","message_id":"native-message","page_url":"https://chatgpt.com/c/passage","selected_at":"2026-10-24T00:00:00Z"}});
    let mut missing = note.clone();
    missing["request_id"] = json!("019c6e27-e55b-73d1-87d8-4e01f1f75044");
    missing["source"]["message_id"] = json!("not-captured");
    let (status, refused) = native("internal selection-note", &missing);
    assert_eq!(status, 409, "{refused}");
    let (status, saved) = native("internal selection-note", &note);
    assert_eq!(status, 200, "{saved}");
    assert_eq!(saved["repeat"], false);
    drop(daemon);
    let _daemon = start();
    let (status, retried) = native("internal selection-note", &note);
    assert_eq!(status, 200, "{retried}");
    assert_eq!(retried["repeat"], true);
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
    let (status, refused) = native("internal selection-note", &changed);
    assert_eq!(status, 409, "{refused}");
}

/// Web chats driven as a person drives them: `send`, `answer`, `start --in`
/// and `import PROVIDER` on the command line against a running daemon, and
/// the popup's Allow through the shared operation.
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
    // A command line without a terminal is not the operator
    // (docs/authorship.md), so its submit waits for the operator's answer.
    let submitted = json_of(&["start", "a new conversation", "--in", "chatgpt", "--submit"]);
    assert!(
        submitted.to_string().contains("\"needs_approval\":true")
            && found(&submitted, "status", "awaiting_approval"),
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
    let response = native(
        &environment,
        &json!({"version":2,"type":"call","operation":"internal chat answer",
        "arguments":{"id":id,"value":"deny"},"request_id":"popup-deny"}),
    );
    assert_eq!(response["outcome"], "ok", "{response}");

    let job = json_of(&["import", "chatgpt", "--since", "2026-01-01"]);
    assert!(found(&job, "status", "waiting"), "{job}");
    let cancelled = json_of(&["import", "chatgpt", "--cancel"]);
    assert!(found(&cancelled, "status", "cancelled"), "{cancelled}");
    let help = cli(&["help", "backfill"]);
    assert!(help.contains("import PROVIDER"), "{help}");
}
