//! The browser host as the extension uses it: whole native messaging
//! exchanges against a real daemon on a temporary home.

use super::native::{read_frame, serve, write_frame};
use super::*;
use crate::test_daemon::{self, TestDaemon};
use base64::Engine as _;
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn host(w: &TestDaemon) -> Arc<BrowserHost> {
    Arc::new(BrowserHost::at(&w.config, &w.daemon.archive).unwrap())
}

struct Answer {
    status: u16,
    headers: Value,
    body: Vec<u8>,
    frames: Vec<Value>,
}

impl Answer {
    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

/// One exchange as the extension's `nativeFetch` makes it.
fn exchange(host: &Arc<BrowserHost>, method: &str, path: &str, body: &[u8]) -> Answer {
    let mut input = Vec::new();
    write_frame(
        &mut input,
        &json!({ "type": "request", "version": 1, "endpoint": "http://127.0.0.1:8765",
                 "receiver_id": null, "method": method, "path": path,
                 "headers": {"content-type": "application/json", "x-request-id": "req-1"} }),
    )
    .unwrap();
    for (i, chunk) in body.chunks(64 * 1024).enumerate() {
        write_frame(
            &mut input,
            &json!({ "type": "body", "sequence": i, "data": B64.encode(chunk) }),
        )
        .unwrap();
    }
    write_frame(&mut input, &json!({ "type": "body_end" })).unwrap();
    for _ in 0..64 {
        write_frame(&mut input, &json!({ "type": "response_next" })).unwrap();
    }
    let mut out = Vec::new();
    serve(Arc::clone(host), &mut input.as_slice(), &mut out).unwrap();
    let mut frames = Vec::new();
    let mut reader = out.as_slice();
    while let Some(f) = read_frame(&mut reader).unwrap() {
        frames.push(f);
    }
    let response = frames
        .iter()
        .find(|f| f["type"] == "response")
        .cloned()
        .unwrap_or_else(|| panic!("no response frame: {frames:?}"));
    let mut body = Vec::new();
    for f in frames.iter().filter(|f| f["type"] == "response_body") {
        body.extend(B64.decode(f["data"].as_str().unwrap()).unwrap());
    }
    Answer {
        status: response["status"].as_u64().unwrap() as u16,
        headers: response["headers"].clone(),
        body,
        frames,
    }
}

fn capture(provider_session: &str, text: &str) -> Vec<u8> {
    json!({
        "polylogue_capture_kind": "browser_capture",
        "schema_version": 1,
        "capture_id": format!("chatgpt:{provider_session}"),
        "source": "browser-extension",
        "provenance": {"captured_at": "2026-03-01T10:00:00Z", "url": "https://chatgpt.com/c/x"},
        "session": {
            "provider": "chatgpt",
            "provider_session_id": provider_session,
            "title": "captured",
            "turns": [{"provider_turn_id": "t1", "role": "user", "text": text}]
        }
    })
    .to_string()
    .into_bytes()
}

#[test]
fn a_capture_is_spooled_durably_and_archived_by_the_loop() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let status = exchange(&host, "GET", "/v1/status", b"");
    assert_eq!(status.status, 200);
    let frames = &status.frames;
    assert_eq!(frames[0]["type"], "response");
    assert_eq!(frames[0]["api_schema"], API_SCHEMA);
    assert_eq!(frames[0]["receiver_id"], host.receiver_id.as_str());
    assert_eq!(status.json()["daemon"], "running");

    // A body over one 64 KiB frame arrives whole, acknowledged piece by piece.
    let text = "captured in the browser ".repeat(5000);
    let bytes = capture("conv-1", &text);
    assert!(bytes.len() > 64 * 1024);
    let a = exchange(&host, "POST", "/v1/browser-captures", &bytes);
    assert_eq!(a.status, 200, "{:?}", a.json());
    let acks = a.frames.iter().filter(|f| f["type"] == "body_ack").count();
    assert_eq!(acks, bytes.len().div_ceil(64 * 1024));
    assert_eq!(a.headers["x-request-id"], "req-1");
    let ack = a.json();
    let sha = sha256_hex(&bytes);
    assert_eq!(ack["outcome"], "accepted");
    assert_eq!(ack["content_hash"], sha.as_str());
    assert_eq!(ack["submitted_content_hash"], sha.as_str());
    let spooled = host.spool.join(ack["artifact_ref"].as_str().unwrap());
    let stored = std::fs::read(&spooled).unwrap();
    let body_start =
        polylogue_model::capture_log::MAGIC.len() + polylogue_model::capture_log::HEADER_LEN;
    assert_eq!(&stored[body_start..], bytes);
    assert!(spooled.starts_with(w.home.join(".local/state/polylogue/browser-capture")));
    // The same bytes again change nothing.
    let again = exchange(&host, "POST", "/v1/browser-captures", &bytes);
    assert_eq!(again.json()["outcome"], "noop");

    // The daemon's loop reads the spool; archive state follows it.
    loop {
        let s = exchange(
            &host,
            "GET",
            "/v1/archive-state?provider=chatgpt&provider_session_id=conv-1",
            b"",
        )
        .json();
        assert_eq!(s["spooled"], true);
        if s["state"] == "archived" {
            assert_eq!(s["indexed_session_id"], "chatgpt:conv-1");
            assert_eq!(s["indexed_message_count"], 1);
            break;
        }
        assert_eq!(s["state"], "spooled_only");
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Settings that read only a home under `dir`.
fn config_at(dir: &std::path::Path) -> polylogue_engine::Config {
    let mut c = polylogue_engine::Config::from_env();
    c.home = dir.join("home");
    c.tool_dirs.clear();
    c
}

#[test]
fn without_the_daemon_the_index_is_unknown_not_missing() {
    let dir = tempfile::tempdir().unwrap();
    let archive = polylogue_store::Archive::at(&dir.path().join("archive"));
    let host = Arc::new(BrowserHost::at(&config_at(dir.path()), &archive).unwrap());
    let a = exchange(
        &host,
        "POST",
        "/v1/browser-captures",
        &capture("conv-2", "hi"),
    );
    assert_eq!(a.status, 200);
    let s = exchange(
        &host,
        "GET",
        "/v1/archive-state?provider=chatgpt&provider_session_id=conv-2",
        b"",
    )
    .json();
    assert_eq!(s["state"], "spooled_unknown_index");
    assert_eq!(s["indexed_message_count"], Value::Null);
    let s = exchange(
        &host,
        "GET",
        "/v1/archive-state?provider=chatgpt&provider_session_id=never",
        b"",
    )
    .json();
    assert_eq!(s["state"], "unknown");
    assert_eq!(
        exchange(&host, "GET", "/v1/status", b"").json()["daemon"],
        "not_running"
    );
}

#[test]
fn malformed_exchanges_are_refused_with_an_error_frame() {
    let dir = tempfile::tempdir().unwrap();
    let archive = polylogue_store::Archive::at(&dir.path().join("archive"));
    let host = Arc::new(BrowserHost::at(&config_at(dir.path()), &archive).unwrap());
    let run = |frames: &[Value]| {
        let mut input = Vec::new();
        for f in frames {
            write_frame(&mut input, f).unwrap();
        }
        let mut out = Vec::new();
        serve(Arc::clone(&host), &mut input.as_slice(), &mut out).unwrap();
        read_frame(&mut out.as_slice()).unwrap()
    };
    assert_eq!(
        run(&[json!({"type": "body"})]).unwrap()["error"],
        "native_request_invalid"
    );
    assert_eq!(
        run(&[json!({"type": "request", "version": 1, "method": "GET", "path": "/api/v1/status"})])
            .unwrap()["error"],
        "native_route_forbidden"
    );
    // A body frame out of sequence.
    assert_eq!(
        run(&[
            json!({"type": "request", "version": 1, "method": "POST", "path": "/v1/browser-captures"}),
            json!({"type": "body", "sequence": 3, "data": ""}),
        ])
        .unwrap()["error"],
        "native_request_frame_invalid"
    );
    // A spool entry that is a symlink is not followed.
    let rel = artifact_ref("chatgpt", "conv-3");
    let target = dir.path().join("elsewhere");
    std::fs::write(&target, b"untouched").unwrap();
    std::fs::create_dir_all(host.spool.join("chatgpt")).unwrap();
    std::os::unix::fs::symlink(&target, host.spool.join(&rel)).unwrap();
    let a = exchange(
        &host,
        "POST",
        "/v1/browser-captures",
        &capture("conv-3", "x"),
    );
    assert_eq!(a.status, 500);
    assert_eq!(std::fs::read(&target).unwrap(), b"untouched");
}

fn request(op: &str, conversation: &str, policy: &str, key: &str) -> actions::ActionRequest {
    serde_json::from_value(json!({
        "idempotency_key": key,
        "provider": "chatgpt",
        "operation": op,
        "target": {"conversation_id": conversation},
        "text": "summarize this",
        "presentation": {"model_slug": "gpt-x", "model_label": "GPT X", "effort_label": "standard"},
        "submit_policy": policy
    }))
    .unwrap()
}

/// Queues `req` as the operator does; the action, or the refusal's code.
fn enqueue(w: &TestDaemon, req: actions::ActionRequest) -> Result<Value, String> {
    match crate::daemon::service::handle(
        crate::daemon::Request::ChatEnqueue {
            action: Box::new(req),
        },
        &w.daemon,
    ) {
        crate::daemon::Response::Ok { data } => Ok(data["action"].clone()),
        crate::daemon::Response::Error { kind, .. } => Err(kind),
    }
}

fn report(host: &Arc<BrowserHost>, id: &str, who: &str, outcome: &str) -> Answer {
    exchange(
        host,
        "POST",
        &format!("/v1/browser-actions/{id}/events"),
        json!({"owner_instance_id": who, "outcome": outcome, "phase": outcome})
            .to_string()
            .as_bytes(),
    )
}

#[test]
fn the_extension_claims_and_reports_actions_the_operator_queued() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let claim = |who: &str| {
        exchange(
            &host,
            "GET",
            &format!("/v1/browser-actions?claim_by={who}"),
            b"",
        )
        .json()
    };
    // Off by default: nothing queues and nothing is handed out.
    assert_eq!(
        enqueue(
            &w,
            request("conversation.create", "new", "stage_only", "k1")
        )
        .unwrap_err(),
        "browser_actions_off"
    );
    w.set("browser.actions", "on");
    assert!(host.actions_enabled());
    let a = enqueue(
        &w,
        request("conversation.create", "new", "stage_only", "k1"),
    )
    .unwrap();
    let a_id = a["action_id"].as_str().unwrap().to_owned();
    // The extension cannot queue.
    let body =
        serde_json::to_vec(&request("conversation.create", "new", "stage_only", "k0")).unwrap();
    assert_eq!(
        exchange(&host, "POST", "/v1/browser-actions", &body).status,
        403
    );

    let got = claim("tab-a");
    assert_eq!(got["actions"][0]["action_id"], a_id.as_str());
    assert_eq!(got["actions"][0]["status"], "claimed");
    // One action out at a time.
    let b = enqueue(
        &w,
        request("conversation.create", "new", "stage_only", "k2"),
    )
    .unwrap();
    let b_id = b["action_id"].as_str().unwrap().to_owned();
    assert_eq!(claim("tab-b")["actions"], json!([]));

    // Only the owner reports; a draft-only action cannot submit.
    assert_eq!(report(&host, &a_id, "tab-b", "drafted").status, 409);
    assert_eq!(report(&host, &a_id, "tab-a", "submitted").status, 409);
    let done = report(&host, &a_id, "tab-a", "drafted");
    assert_eq!(done.status, 200);
    assert_eq!(done.json()["action"]["status"], "drafted");
    assert_eq!(claim("tab-b")["actions"][0]["action_id"], b_id.as_str());

    // A submit into an existing conversation waits for the operator.
    let held = enqueue(
        &w,
        request("conversation.reply", "conv-9", "submit_once", "k3"),
    )
    .unwrap();
    assert_eq!(held["status"], "awaiting_approval");
    let held_id = held["action_id"].as_str().unwrap().to_owned();
    report(&host, &b_id, "tab-b", "failed");
    assert_eq!(claim("tab-a")["actions"], json!([]));
    let approve = exchange(
        &host,
        "POST",
        &format!("/v1/browser-actions/{}/approval", held_id),
        br#"{"extension_instance_id":"popup","decision":"approve"}"#,
    );
    assert_eq!(approve.status, 200);

    // The kill switch: nothing is handed out once the setting is off.
    w.set("browser.actions", "off");
    let v = claim("tab-a");
    assert_eq!(v["actions"], json!([]));
    assert_eq!(v["enabled"], false);
    w.set("browser.actions", "on");
    assert_eq!(claim("tab-a")["actions"][0]["action_id"], held_id.as_str());
}

#[test]
fn the_host_manifest_allows_only_the_named_extension() {
    let m = native::manifest(std::path::Path::new("/bin/polylogue"), "abcdefghijklmnop");
    assert_eq!(m["name"], HOST_NAME);
    assert_eq!(m["type"], "stdio");
    assert_eq!(
        m["allowed_origins"],
        json!(["chrome-extension://abcdefghijklmnop/"])
    );
}

#[test]
fn a_failed_capture_write_removes_its_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("capture.json");
    std::fs::create_dir(&target).unwrap();
    assert!(super::write_durable(&target, b"new content").is_err());
    let entries: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(entries, [std::ffi::OsString::from("capture.json")]);
}

#[test]
fn native_inspection_uses_the_capture_parser_without_writing_a_spool() {
    let dir = tempfile::tempdir().unwrap();
    let archive = polylogue_store::Archive::at(&dir.path().join("archive"));
    let host = Arc::new(BrowserHost::at(&config_at(dir.path()), &archive).unwrap());
    let answer = exchange(
        &host,
        "POST",
        "/v1/browser-captures/inspect",
        &capture("inspect-session", "hello"),
    );
    assert_eq!(answer.status, 200, "{:?}", answer.json());
    assert_eq!(answer.json()["summary"]["turn_count"], 1);
    assert_eq!(std::fs::read_dir(&host.spool).unwrap().count(), 0);
    assert_eq!(
        exchange(&host, "POST", "/v1/browser-captures/inspect", b"{}").status,
        422
    );
}

#[test]
fn a_capture_larger_than_axums_default_body_limit_is_accepted() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let bytes = capture("conv-large", &"a long conversation ".repeat(200_000));
    assert!(bytes.len() > 3 * 1024 * 1024);
    let inspected = exchange(&host, "POST", "/v1/browser-captures/inspect", &bytes);
    assert_eq!(inspected.status, 200, "{:?}", inspected.json());
    let sent = exchange(&host, "POST", "/v1/browser-captures", &bytes);
    assert_eq!(sent.status, 200, "{:?}", sent.json());
    assert_eq!(sent.json()["outcome"], "accepted");
}

#[test]
fn native_raw_grok_capture_reaches_an_ordinary_archive_read() {
    let w = test_daemon::start("fixture");
    let host = host(&w);
    let body = serde_json::to_vec(&json!({
        "polylogue_capture_kind":"browser_llm_session", "schema_version":1,
        "session":{"provider":"grok", "provider_session_id":"native-raw", "turns":[]},
        "raw_provider_payload":{
            "conversation":{"conversationId":"native-raw", "title":"Native fixture"},
            "responses":{"responses":[{"responseId":"raw-r1", "sender":"human", "message":"raw provider question", "fileAttachmentsMetadata":[{"fileMetadataId":"raw-file", "fileName":"notes.txt", "fileMimeType":"text/plain", "fileUri":"users/u/raw-file/content"}]}]}
        }
    })).unwrap();
    let inspected = exchange(&host, "POST", "/v1/browser-captures/inspect", &body);
    assert_eq!(inspected.status, 200, "{:?}", inspected.json());
    assert_eq!(
        inspected.json()["attachments"][0]["provider_attachment_id"],
        "raw-file"
    );
    let mut envelope: Value = serde_json::from_slice(&body).unwrap();
    envelope["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "content_base64":B64.encode(b"native asset bytes")}]);
    // The extension streams schema_version before the capture marker.
    let mut fields = envelope.as_object().unwrap().clone();
    fields.remove("schema_version");
    let streamed = format!(
        "{{\"schema_version\":1,{}",
        &serde_json::to_string(&fields).unwrap()[1..]
    );
    let sent = exchange(&host, "POST", "/v1/browser-captures", streamed.as_bytes());
    assert_eq!(sent.status, 200);
    loop {
        if let Some(crate::daemon::Response::Ok { data }) =
            host.ask(&crate::daemon::Request::Read {
                target: crate::daemon::ReadTarget::Session(
                    polylogue_model::SessionId::parse("grok:native-raw").unwrap(),
                ),
                view: "full".into(),
                format: "markdown".into(),
                with_parents: false,
                budget: None,
                cursor: None,
            })
            && data["content"]
                .as_str()
                .is_some_and(|text| text.contains("raw provider question"))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let hash = polylogue_model::BlobHash::of(b"native asset bytes");
    let messages = w
        .daemon
        .reads
        .reader()
        .unwrap()
        .messages(
            &polylogue_model::SessionId::parse("grok:native-raw").unwrap(),
            false,
        )
        .unwrap();
    assert!(messages[0].blocks.iter().any(
        |block| matches!(block, polylogue_model::Block::File(file) if file.blob == Some(hash))
    ));
}

#[test]
fn reader_location_is_typed_unavailable_when_no_reader_is_running() {
    let dir = tempfile::tempdir().unwrap();
    let archive = polylogue_store::Archive::at(&dir.path().join("archive"));
    let host = Arc::new(BrowserHost::at(&config_at(dir.path()), &archive).unwrap());
    let response = exchange(&host, "GET", "/v1/reader-location", b"");
    assert_eq!(response.status, 503);
    assert_eq!(response.json()["error"], "reader_unavailable");
}
