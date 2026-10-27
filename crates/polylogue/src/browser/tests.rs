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

/// One exchange as the extension's `exchange` (`background/host.js`) makes it.
fn exchange(host: &Arc<BrowserHost>, method: &str, path: &str, body: &[u8]) -> Answer {
    let mut input = Vec::new();
    write_frame(
        &mut input,
        &json!({ "type": "request", "version": 1, "method": method, "path": path,
                 "headers": {"content-type": "application/json"} }),
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
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {"captured_at": "2026-03-01T10:00:00Z", "source_url": "https://chatgpt.com/c/x"},
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

/// Polls `answer` until `done` holds of it: the daemon's loop reads the
/// spool in its own time.
fn until(mut answer: impl FnMut() -> Value, done: impl Fn(&Value) -> bool) -> Value {
    for _ in 0..600 {
        let v = answer();
        if done(&v) {
            return v;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("the daemon never reached the expected state");
}

fn conversations(host: &Arc<BrowserHost>, ids: &[&str]) -> Value {
    let body = json!({"provider": "chatgpt", "ids": ids}).to_string();
    exchange(host, "POST", "/v1/conversations", body.as_bytes()).json()
}

#[test]
fn a_capture_is_stored_durably_and_archived_by_the_loop() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let status = exchange(&host, "GET", "/v1/status", b"");
    assert_eq!(status.status, 200);
    assert_eq!(status.frames[0]["type"], "response");
    assert_eq!(status.json()["daemon"], "running");
    assert_eq!(status.json()["chat_actions"], false);

    // A body over one 64 KiB frame arrives whole, acknowledged piece by piece.
    let text = "captured in the browser ".repeat(5000);
    let bytes = capture("conv-1", &text);
    assert!(bytes.len() > 64 * 1024);
    let a = exchange(&host, "POST", "/v1/captures", &bytes);
    assert_eq!(a.status, 200, "{:?}", a.json());
    let acks = a.frames.iter().filter(|f| f["type"] == "body_ack").count();
    assert_eq!(acks, bytes.len().div_ceil(64 * 1024));
    let stored = a.json();
    assert_eq!(stored["new"], true);
    assert_eq!(stored["conversation"], "conv-1");
    assert_eq!(stored["hash"], blake3::hash(&bytes).to_hex().as_str());
    let log = host.capture_log("chatgpt", "conv-1");
    let on_disk = std::fs::read(&log).unwrap();
    let body_start =
        polylogue_model::capture_log::MAGIC.len() + polylogue_model::capture_log::HEADER_LEN;
    assert_eq!(&on_disk[body_start..], bytes);
    assert!(log.starts_with(w.home.join(".local/state/polylogue/browser-capture")));
    // The same bytes again, as an extension resending what it still holds
    // after a restart, change nothing.
    let again = exchange(&host, "POST", "/v1/captures", &bytes);
    assert_eq!(again.json()["new"], false);
    assert_eq!(std::fs::read(&log).unwrap(), on_disk);

    // The daemon's loop reads the spool; the conversation's state follows.
    let state = until(
        || conversations(&host, &["conv-1", "never"]),
        |v| v["conversations"][0]["state"] == "archived",
    );
    let row = &state["conversations"][0];
    assert_eq!(row["session"], "chatgpt:conv-1");
    assert_eq!(row["messages"], 1);
    let captured_at = "2026-03-01T10:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    assert_eq!(row["last_captured_at"], captured_at);
    assert_eq!(state["conversations"][1]["state"], "missing");
}

/// Settings that read only a home under `dir`.
fn config_at(dir: &std::path::Path) -> polylogue_engine::Config {
    let mut c = polylogue_engine::Config::from_env();
    c.home = dir.join("home");
    c.tool_dirs.clear();
    c
}

fn host_without_daemon(dir: &std::path::Path) -> Arc<BrowserHost> {
    let archive = polylogue_store::Archive::at(&dir.join("archive"));
    Arc::new(BrowserHost::at(&config_at(dir), &archive).unwrap())
}

#[test]
fn without_the_daemon_a_capture_is_stored_and_the_index_is_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
    let a = exchange(&host, "POST", "/v1/captures", &capture("conv-2", "hi"));
    assert_eq!(a.status, 200);
    let state = conversations(&host, &["conv-2", "never"]);
    assert_eq!(state["daemon"], "no_answer");
    assert_eq!(state["conversations"][0]["state"], "spooled");
    assert_eq!(state["conversations"][0]["messages"], Value::Null);
    assert_eq!(state["conversations"][1]["state"], "unknown");
    assert_eq!(
        exchange(&host, "GET", "/v1/status", b"").json()["daemon"],
        "no_answer"
    );
    let jobs = exchange(&host, "GET", "/v1/backfill", b"");
    assert_eq!(jobs.status, 503);
    assert_eq!(jobs.json()["error"], "daemon_required");
}

#[test]
fn malformed_exchanges_are_refused_with_an_error_frame() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
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
            json!({"type": "request", "version": 1, "method": "POST", "path": "/v1/captures"}),
            json!({"type": "body", "sequence": 3, "data": ""}),
        ])
        .unwrap()["error"],
        "native_request_frame_invalid"
    );
    // A spool entry that is a symlink is not followed.
    let log = host.capture_log("chatgpt", "conv-3");
    let target = dir.path().join("elsewhere");
    std::fs::write(&target, b"untouched").unwrap();
    std::fs::create_dir_all(log.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&target, &log).unwrap();
    let a = exchange(&host, "POST", "/v1/captures", &capture("conv-3", "x"));
    assert_eq!(a.status, 500);
    assert_eq!(a.json()["error"], "write_failed");
    assert_eq!(std::fs::read(&target).unwrap(), b"untouched");
    // Not a capture.
    let a = exchange(&host, "POST", "/v1/captures", b"[1]");
    assert_eq!(
        (a.status, a.json()["error"].clone()),
        (400, json!("capture_not_json"))
    );
}

fn chat_request(conversation: Option<&str>, submit: bool) -> actions::ChatRequest {
    actions::ChatRequest {
        provider: "chatgpt".into(),
        conversation: conversation.map(str::to_owned),
        project: None,
        text: "summarize this".into(),
        model: "GPT X".into(),
        effort: Some("Standard".into()),
        submit,
    }
}

/// Queues an action as the operator does; the action, or the refusal's code.
fn send(
    w: &TestDaemon,
    key: &str,
    action: actions::ChatRequest,
    files: Vec<actions::ChatFile>,
) -> Result<Value, String> {
    let request = crate::daemon::Request::ChatSend {
        request_id: key.into(),
        action,
        files,
    };
    match crate::daemon::service::handle(request, &w.daemon) {
        crate::daemon::Response::Ok { data } => Ok(data["action"].clone()),
        crate::daemon::Response::Error { kind, .. } => Err(kind),
    }
}

fn report(host: &Arc<BrowserHost>, id: &str, by: &str, outcome: &str) -> Answer {
    exchange(
        host,
        "POST",
        &format!("/v1/chat-actions/{id}/report"),
        json!({"by": by, "outcome": outcome, "phase": outcome})
            .to_string()
            .as_bytes(),
    )
}

#[test]
fn the_extension_claims_and_reports_actions_the_operator_queued() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let claim = |by: &str| {
        exchange(
            &host,
            "GET",
            &format!("/v1/chat-actions?claim_by={by}"),
            b"",
        )
        .json()
    };
    // Off by default: nothing queues and nothing is handed out.
    assert_eq!(
        send(&w, "k1", chat_request(None, false), vec![]).unwrap_err(),
        "browser_actions_off"
    );
    w.set("browser.actions", "on");
    assert_eq!(
        exchange(&host, "GET", "/v1/status", b"").json()["chat_actions"],
        true
    );
    let file = actions::ChatFile {
        name: "notes.txt".into(),
        media_type: "text/plain".into(),
        bytes: b"attached bytes".to_vec(),
    };
    let a = send(&w, "k1", chat_request(None, false), vec![file.clone()]).unwrap();
    let a_id = a["id"].as_str().unwrap().to_owned();
    // The same request again is the same action.
    let again = send(&w, "k1", chat_request(None, false), vec![file]).unwrap();
    assert_eq!(again["id"], a_id.as_str());

    let got = claim("tab-a");
    assert_eq!(got["action"]["id"], a_id.as_str());
    assert_eq!(got["action"]["status"], "claimed");
    assert_eq!(got["action"]["model"], "GPT X");
    let attachment = &got["action"]["attachments"][0];
    let bytes = exchange(
        &host,
        "GET",
        &format!(
            "/v1/chat-actions/{a_id}/attachments/{}",
            attachment["id"].as_str().unwrap()
        ),
        b"",
    );
    assert_eq!(bytes.body, b"attached bytes");
    assert_eq!(bytes.headers["content-type"], "text/plain");
    // One action out at a time.
    let b = send(&w, "k2", chat_request(None, false), vec![]).unwrap();
    let b_id = b["id"].as_str().unwrap().to_owned();
    assert_eq!(claim("tab-b")["action"], Value::Null);

    // Only the holder reports; a draft-only action cannot submit.
    assert_eq!(report(&host, &a_id, "tab-b", "drafted").status, 409);
    assert_eq!(report(&host, &a_id, "tab-a", "submitted").status, 409);
    let done = report(&host, &a_id, "tab-a", "drafted");
    assert_eq!(done.status, 200);
    assert_eq!(done.json()["action"]["status"], "drafted");
    assert_eq!(claim("tab-b")["action"]["id"], b_id.as_str());

    // A submit into an existing conversation waits for the operator.
    let held = send(&w, "k3", chat_request(Some("conv-9"), true), vec![]).unwrap();
    assert_eq!(held["status"], "awaiting_approval");
    let held_id = held["id"].as_str().unwrap().to_owned();
    report(&host, &b_id, "tab-b", "failed");
    assert_eq!(claim("tab-a")["action"], Value::Null);
    let approve = exchange(
        &host,
        "POST",
        &format!("/v1/chat-actions/{held_id}/decision"),
        br#"{"by":"popup","approve":true}"#,
    );
    assert_eq!(approve.status, 200);

    // The switch: nothing is handed out once the setting is off.
    w.set("browser.actions", "off");
    let v = claim("tab-a");
    assert_eq!(v["action"], Value::Null);
    assert_eq!(v["enabled"], false);
    w.set("browser.actions", "on");
    assert_eq!(claim("tab-a")["action"]["id"], held_id.as_str());
    let listed = exchange(&host, "GET", "/v1/chat-actions", b"").json();
    assert_eq!(listed["actions"].as_array().unwrap().len(), 3);
}

fn backfill_report(host: &Arc<BrowserHost>, id: &str, by: &str, body: Value) -> Answer {
    let mut body = body;
    body["by"] = json!(by);
    exchange(
        host,
        "POST",
        &format!("/v1/backfill/{id}/report"),
        body.to_string().as_bytes(),
    )
}

#[test]
fn a_backfill_job_runs_through_the_daemon_and_survives_an_extension_restart() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let start = |provider: &str| {
        crate::daemon::service::handle(
            crate::daemon::Request::BackfillStart {
                provider: provider.into(),
                since: Some("2026-01-01".into()),
            },
            &w.daemon,
        )
    };
    let crate::daemon::Response::Ok { data } = start("chatgpt") else {
        panic!("the job did not start")
    };
    let id = data["job"]["id"].as_str().unwrap().to_owned();
    assert_eq!(data["job"]["status"], "waiting");
    let claim = |by: &str| {
        exchange(
            &host,
            "POST",
            "/v1/backfill/claim",
            json!({"by": by}).to_string().as_bytes(),
        )
        .json()
    };
    let job = claim("ext-1");
    assert_eq!(job["job"]["id"], id.as_str());
    assert_eq!(job["job"]["position"], Value::Null);

    // The extension captures what the provider listed and reports.
    let bytes = capture("listed-1", "an older conversation");
    assert_eq!(exchange(&host, "POST", "/v1/captures", &bytes).status, 200);
    let progress = json!({"kind": "progress", "position": "0:28",
        "counts": {"listed": 28, "captured": 1, "unchanged": 27, "failed": 0}});
    assert_eq!(
        backfill_report(&host, &id, "ext-1", progress.clone()).status,
        200
    );
    assert_eq!(backfill_report(&host, &id, "ext-2", progress).status, 409);

    // A restarted extension instance claims its job again and continues
    // from the reported position.
    let job = claim("ext-1");
    assert_eq!(job["job"]["id"], id.as_str());
    assert_eq!(job["job"]["position"], "0:28");
    assert_eq!(job["job"]["captured"], 1);
    // Another instance has nothing to take.
    assert_eq!(claim("ext-2")["job"], Value::Null);

    let finished = json!({"kind": "finished", "position": "3:0",
        "counts": {"listed": 40, "captured": 2, "unchanged": 38, "failed": 0}});
    let done = backfill_report(&host, &id, "ext-1", finished);
    assert_eq!(done.json()["job"]["status"], "finished");
    let listed = exchange(&host, "GET", "/v1/backfill", b"").json();
    assert_eq!(listed["jobs"][0]["status"], "finished");
    assert_eq!(listed["jobs"][0]["listed"], 40);
    // The conversation it captured is archived like any other capture.
    until(
        || conversations(&host, &["listed-1"]),
        |v| v["conversations"][0]["state"] == "archived",
    );
    let crate::daemon::Response::Error { kind, .. } = start("gemini") else {
        panic!("gemini has no conversation list")
    };
    assert_eq!(kind, "unsupported_provider");
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
fn reading_a_captures_files_uses_the_capture_parser_without_writing_the_spool() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
    let answer = exchange(
        &host,
        "POST",
        "/v1/captures/files",
        &capture("files-session", "hello"),
    );
    assert_eq!(answer.status, 200, "{:?}", answer.json());
    assert_eq!(answer.json()["conversation"]["messages"], 1);
    assert_eq!(std::fs::read_dir(&host.spool).unwrap().count(), 0);
    assert_eq!(
        exchange(&host, "POST", "/v1/captures/files", b"{}").status,
        422
    );
}

#[test]
fn a_capture_larger_than_axums_default_body_limit_is_accepted() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let bytes = capture("conv-large", &"a long conversation ".repeat(200_000));
    assert!(bytes.len() > 3 * 1024 * 1024);
    let read = exchange(&host, "POST", "/v1/captures/files", &bytes);
    assert_eq!(read.status, 200, "{:?}", read.json());
    let sent = exchange(&host, "POST", "/v1/captures", &bytes);
    assert_eq!(sent.status, 200, "{:?}", sent.json());
    assert_eq!(sent.json()["new"], true);
}

/// A Grok capture whose question names one file, `raw-file`.
fn grok_capture() -> Value {
    json!({
        "polylogue_capture_kind":"browser_llm_session", "schema_version":1,
        "session":{"provider":"grok", "provider_session_id":"grok-raw", "turns":[]},
        "raw_provider_payload":{
            "conversation":{"conversationId":"grok-raw", "title":"Grok fixture"},
            "responses":{"responses":[{"responseId":"raw-r1", "sender":"human", "message":"raw provider question", "fileAttachmentsMetadata":[{"fileMetadataId":"raw-file", "fileName":"notes.txt", "fileMimeType":"text/plain", "fileUri":"users/u/raw-file/content"}]}]}
        }
    })
}

/// The file blobs of the Grok fixture's question once the loop has read
/// a capture holding `text`.
fn grok_file_blobs(
    w: &TestDaemon,
    host: &Arc<BrowserHost>,
    text: &str,
) -> Vec<Option<polylogue_model::BlobHash>> {
    let session = polylogue_model::SessionId::parse("grok:grok-raw").unwrap();
    until(
        || match host.ask(&crate::daemon::Request::Read {
            start: None,
            target: crate::daemon::ReadTarget::Session(session.clone()),
            view: "full".into(),
            format: "markdown".into(),
            with_parents: false,
            budget: None,
            cursor: None,
        }) {
            Some(crate::daemon::Response::Ok { data }) => data["content"].clone(),
            _ => Value::Null,
        },
        |content| content.as_str().is_some_and(|t| t.contains(text)),
    );
    let messages = w
        .daemon
        .reads
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap();
    messages[0]
        .blocks
        .iter()
        .filter_map(|block| match block {
            polylogue_model::Block::File(file) => Some(file.blob),
            _ => None,
        })
        .collect()
}

#[test]
fn a_grok_capture_with_fetched_file_bytes_reaches_an_ordinary_archive_read() {
    let w = test_daemon::start("fixture");
    let host = host(&w);
    let body = serde_json::to_vec(&grok_capture()).unwrap();
    let read = exchange(&host, "POST", "/v1/captures/files", &body);
    assert_eq!(read.status, 200, "{:?}", read.json());
    assert_eq!(read.json()["files"][0]["id"], "raw-file");
    assert_eq!(read.json()["files"][0]["url"], "users/u/raw-file/content");
    assert_eq!(read.json()["files"][0]["stored"], Value::Null);
    let mut capture = grok_capture();
    capture["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "content_base64":B64.encode(b"fetched file bytes")}]);
    let sent = exchange(
        &host,
        "POST",
        "/v1/captures",
        &serde_json::to_vec(&capture).unwrap(),
    );
    assert_eq!(sent.status, 200);
    let hash = polylogue_model::BlobHash::of(b"fetched file bytes");
    assert_eq!(
        grok_file_blobs(&w, &host, "raw provider question"),
        [Some(hash)]
    );
}

/// The extension downloads a file once: a later capture names the stored
/// bytes by their hash, and the archive still links the file to them.
#[test]
fn a_file_the_archive_holds_is_named_by_its_hash_instead_of_sent_again() {
    let w = test_daemon::start("fixture");
    let host = host(&w);
    let mut first = grok_capture();
    first["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "content_base64":B64.encode(b"fetched file bytes")}]);
    let sent = exchange(
        &host,
        "POST",
        "/v1/captures",
        &serde_json::to_vec(&first).unwrap(),
    );
    assert_eq!(sent.status, 200);
    let hash = polylogue_model::BlobHash::of(b"fetched file bytes");
    grok_file_blobs(&w, &host, "raw provider question");

    let mut later = grok_capture();
    later["raw_provider_payload"]["responses"]["responses"][0]["message"] =
        json!("raw provider question, edited");
    let read = exchange(
        &host,
        "POST",
        "/v1/captures/files",
        &serde_json::to_vec(&later).unwrap(),
    );
    assert_eq!(read.status, 200, "{:?}", read.json());
    assert_eq!(
        read.json()["files"][0]["stored"],
        json!({"blob": hash.to_string(), "size": 18})
    );
    later["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "blob": hash.to_string(), "size_bytes": 18}]);
    let sent = exchange(
        &host,
        "POST",
        "/v1/captures",
        &serde_json::to_vec(&later).unwrap(),
    );
    assert_eq!(sent.status, 200);
    assert_eq!(grok_file_blobs(&w, &host, "edited"), [Some(hash)]);
}

#[test]
fn the_reader_address_is_unavailable_when_no_reader_is_running() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
    let response = exchange(&host, "GET", "/v1/reader", b"");
    assert_eq!(response.status, 503);
    assert_eq!(response.json()["error"], "reader_unavailable");
}

mod notes;
mod repeated_captures;
