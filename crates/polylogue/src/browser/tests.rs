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
    data: Value,
}
impl Answer {
    fn json(&self) -> Value {
        self.data.clone()
    }
}
fn exchange(host: &Arc<BrowserHost>, message: Value) -> Answer {
    let mut input = Vec::new();
    super::native::write_message(&mut input, &message).unwrap();
    let mut out = Vec::new();
    serve(Arc::clone(host), &mut input.as_slice(), &mut out).unwrap();
    let frame = read_frame(&mut out.as_slice()).unwrap().unwrap();
    assert_eq!(frame["version"], 2);
    assert_eq!(frame["type"], "answer");
    let answer: command_surface::Answer<Value> =
        serde_json::from_value(frame["answer"].clone()).unwrap();
    match crate::protocol::into_value(answer) {
        Ok(data) => Answer { status: 200, data },
        Err(error) => Answer {
            status: error.kind.http_status().unwrap_or(503),
            data: json!({"error":error.code,"message":error.message,"reason":error.reason()}),
        },
    }
}
fn call(host: &Arc<BrowserHost>, operation: &str, arguments: Value) -> Answer {
    exchange(
        host,
        json!({"version":2,"type":"call","operation":operation,
        "arguments":arguments,"request_id":uuid::Uuid::now_v7().to_string()}),
    )
}
fn store_capture(host: &Arc<BrowserHost>, bytes: &[u8]) -> Answer {
    exchange(
        host,
        json!({"version":2,"type":"capture","capture":std::str::from_utf8(bytes).unwrap()}),
    )
}
fn capture_files(host: &Arc<BrowserHost>, bytes: &[u8]) -> Answer {
    call(
        host,
        "internal capture-files",
        json!({"capture":std::str::from_utf8(bytes).unwrap()}),
    )
}
fn save_note(host: &Arc<BrowserHost>, bytes: &[u8]) -> Answer {
    let note: Value = serde_json::from_slice(bytes).unwrap();
    let mut args = note["source"].clone();
    args["body"] = note["body"].clone();
    args["quote"] = note["quote"].clone();
    exchange(
        host,
        json!({"version":2,"type":"call","operation":"internal selection-note",
        "arguments":args,"request_id":note["request_id"]}),
    )
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
    call(
        host,
        "internal conversations",
        json!({"provider": "chatgpt", "ids": ids}),
    )
    .json()
}

#[test]
fn a_capture_is_stored_durably_and_archived_by_the_loop() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let status = call(&host, "set", json!({"key":"browser.actions"}));
    assert_eq!(status.status, 200);
    assert_eq!(status.json()["value"]["type"], "boolean");
    assert_eq!(status.json()["value"]["value"], false);

    // A large logical message arrives whole.
    let text = "captured in the browser ".repeat(5000);
    let bytes = capture("conv-1", &text);
    assert!(bytes.len() > 64 * 1024);
    let a = store_capture(&host, &bytes);
    assert_eq!(a.status, 200, "{:?}", a.json());
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
    let again = store_capture(&host, &bytes);
    assert_eq!(again.json()["new"], false);
    assert_eq!(std::fs::read(&log).unwrap(), on_disk);

    // The daemon's loop reads the spool; the conversation's state follows.
    let state = until(
        || conversations(&host, &["conv-1", "never"]),
        |v| v["conversations"][0]["messages"].is_number(),
    );
    let row = &state["conversations"][0];
    assert_eq!(row["session"], "chatgpt:conv-1");
    assert_eq!(row["messages"], 1);
    let captured_at = "2026-03-01T10:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    assert_eq!(row["last_captured_at"], captured_at);
    assert_eq!(state["conversations"][1]["messages"], Value::Null);
}

/// Settings that read only a home under `dir`.
fn config_at(dir: &std::path::Path) -> polylogue_engine::Config {
    let mut c = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    c.home = dir.join("home");
    c.tool_dirs.clear();
    c
}

fn host_without_daemon(dir: &std::path::Path) -> Arc<BrowserHost> {
    let archive = polylogue_store::Archive::at(&dir.join("archive"));
    let config = config_at(dir);
    Arc::new(BrowserHost::at(&config, &archive).unwrap())
}

fn chat_request(conversation: Option<&str>, submit: bool) -> actions::ChatRequest {
    actions::ChatRequest {
        provider: "chatgpt".into(),
        conversation: conversation.map(str::to_owned),
        project: None,
        text: "summarize this".into(),
        model: Some("GPT X".into()),
        effort: Some("Standard".into()),
        submit,
    }
}

/// Queues an action as the operator does: `start --in chatgpt` for a new
/// conversation, `send` on its session for an existing one. The action, or
/// the refusal's code.
fn send(
    w: &TestDaemon,
    key: &str,
    action: actions::ChatRequest,
    files: Vec<actions::ChatFile>,
) -> Result<Value, String> {
    let dir = w.home.join("attached");
    std::fs::create_dir_all(&dir).unwrap();
    let attach: Vec<String> = files
        .iter()
        .map(|file| {
            let path = dir.join(&file.name);
            std::fs::write(&path, &file.bytes).unwrap();
            path.to_string_lossy().into_owned()
        })
        .collect();
    let request = match &action.conversation {
        None => crate::test_daemon::Call::new(
            "start",
            json!({"text": action.text, "in": action.provider, "model": action.model,
                   "effort": action.effort, "submit": action.submit, "attach": attach}),
        ),
        Some(conversation) => crate::test_daemon::Call::new(
            "rows",
            json!({"selection": format!("session:{}:{conversation}", action.provider),
                   "steps": [{"verb": "send", "text": action.text, "attach": attach}]}),
        ),
    };
    // The operator queues these, as on the command line; only an operator's
    // submit of a new conversation goes out without an answer.
    let operator = polylogue_model::authorship::Authorship {
        author: polylogue_model::authorship::Author::Operator,
        via: polylogue_model::authorship::Channel::Cli,
    };
    let data = match request.id(key).authorship(operator).answer(&w.daemon) {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(data),
            ..
        } => data,
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } => return Err(error.code.unwrap_or_default()),
        other => panic!("a queued action is ok or refused, got {other:?}"),
    };
    if action.conversation.is_none() {
        return Ok(data["action"].clone());
    }
    let step = &data["reply"]["steps"][0]["outcome"];
    match step["error"]["code"].as_str() {
        Some(code) => Err(code.to_owned()),
        None => Ok(step["reply"]["reply"]["reached"][0]["action"].clone()),
    }
}

fn report(host: &Arc<BrowserHost>, id: &str, by: &str, outcome: &str) -> Answer {
    call(
        host,
        "internal chat report",
        json!({"id":id,"update":json!({"by": by, "outcome": outcome, "phase": outcome})}),
    )
}

#[test]
fn the_extension_claims_and_reports_actions_the_operator_queued() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let claim = |by: &str| {
        let answer = call(&host, "internal chat claim", json!({"by":by}));
        assert_eq!(answer.status, 200, "claim by {by}");
        answer.json()
    };
    // Off by default: nothing queues and nothing is handed out.
    assert_eq!(
        send(&w, "k1", chat_request(None, false), vec![]).unwrap_err(),
        "browser_actions_off"
    );
    w.set("browser.actions", "on");
    assert_eq!(
        call(&host, "set", json!({"key":"browser.actions"})).json()["value"]["value"],
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
    let bytes = call(
        &host,
        "internal chat attachment",
        json!({"id":a_id,"attachment":attachment["id"]}),
    );
    assert_eq!(
        B64.decode(bytes.json()["bytes"].as_str().unwrap()).unwrap(),
        b"attached bytes"
    );
    assert_eq!(bytes.json()["media_type"], "text/plain");
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
    // The popup lists what waits for the operator from the unit `chat_actions`.
    let listed = call(
        &host,
        "rows",
        json!({"selection":"chat_actions where state:awaiting_approval"}),
    )
    .json();
    assert_eq!(
        listed["reply"]["rows"].as_array().unwrap().len(),
        1,
        "{listed}"
    );
    assert_eq!(listed["reply"]["rows"][0]["id"], held_id.as_str());
    assert_eq!(listed["reply"]["rows"][0]["state"], "awaiting_approval");
    assert_eq!(listed["reply"]["rows"][0]["conversation"], "conv-9");
    assert_eq!(report(&host, &b_id, "tab-b", "failed").status, 200);
    assert_eq!(claim("tab-a")["action"], Value::Null);
    let approve = call(
        &host,
        "internal chat answer",
        json!({"id":held_id,"value":"allow"}),
    );
    assert_eq!(approve.status, 200);
    assert_eq!(approve.json()["action"]["status"], "queued");
    // The first answer wins; a late one is refused with what won.
    let late = call(
        &host,
        "internal chat answer",
        json!({"id":held_id,"value":"deny"}),
    );
    assert_eq!(late.status, 409, "{:?}", late.json());
    assert!(
        late.json()["message"]
            .as_str()
            .unwrap()
            .starts_with(&format!(
                "already_answered: the held turn {held_id} was allowed by "
            ))
    );

    // The switch: nothing is handed out once the setting is off.
    w.set("browser.actions", "off");
    let v = claim("tab-a");
    assert_eq!(v["action"], Value::Null);
    assert_eq!(v["enabled"], false);
    w.set("browser.actions", "on");
    assert_eq!(claim("tab-a")["action"]["id"], held_id.as_str());
    let listed = call(
        &host,
        "rows",
        json!({"selection":"chat_actions where state:awaiting_approval"}),
    )
    .json();
    assert_eq!(listed["reply"]["rows"], json!([]));
}

fn import_job_report(host: &Arc<BrowserHost>, id: &str, by: &str, body: Value) -> Answer {
    call(
        host,
        "internal import report",
        json!({"id":id,"by":by,"report":body}),
    )
}

#[test]
fn a_import_job_runs_through_the_daemon_and_survives_an_extension_restart() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let start = |provider: &str| {
        crate::test_daemon::Call::new(
            "import",
            serde_json::json!({"paths": [provider], "since": Some("2026-01-01")}),
        )
        .answer(&w.daemon)
    };
    let Some(data) = start("chatgpt").value().cloned() else {
        panic!("the job did not start")
    };
    let id = data["job"]["id"].as_str().unwrap().to_owned();
    assert_eq!(data["job"]["status"], "waiting");
    let claim = |by: &str| {
        let answer = call(&host, "internal import claim", json!({"by": by}));
        assert_eq!(answer.status, 200, "claim by {by}");
        answer.json()
    };
    let job = claim("ext-1");
    assert_eq!(job["job"]["id"], id.as_str());
    assert_eq!(job["job"]["position"], Value::Null);

    // The extension captures what the provider listed and reports.
    let bytes = capture("listed-1", "an older conversation");
    assert_eq!(store_capture(&host, &bytes).status, 200);
    let progress = json!({"kind": "progress", "position": "0:28",
        "counts": {"listed": 28, "captured": 1, "unchanged": 27, "failed": 0}});
    assert_eq!(
        import_job_report(&host, &id, "ext-1", progress.clone()).status,
        200
    );
    assert_eq!(import_job_report(&host, &id, "ext-2", progress).status, 409);

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
    let done = import_job_report(&host, &id, "ext-1", finished);
    assert_eq!(done.json()["job"]["status"], "finished");
    let listed = call(&host, "rows", json!({"selection":"import_jobs"})).json();
    assert_eq!(listed["reply"]["rows"][0]["id"], id.as_str());
    assert_eq!(listed["reply"]["rows"][0]["state"], "finished");
    assert_eq!(listed["reply"]["rows"][0]["listed"], 40);
    assert_eq!(listed["reply"]["rows"][0]["position"], "3:0");
    assert_eq!(listed["reply"]["rows"][0]["claimed_by"], Value::Null);
    // The conversation it captured is archived like any other capture.
    until(
        || conversations(&host, &["listed-1"]),
        |v| v["conversations"][0]["messages"].is_number(),
    );
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = start("gemini")
    else {
        panic!("gemini has no conversation list")
    };
    assert_eq!(error.code.as_deref(), Some("unsupported_provider"));
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
    let w = test_daemon::start("x");
    let host = host(&w);
    let answer = capture_files(&host, &capture("files-session", "hello"));
    assert_eq!(answer.status, 200, "{:?}", answer.json());
    assert_eq!(answer.json()["conversation"]["messages"], 1);
    assert_eq!(std::fs::read_dir(&host.spool).unwrap().count(), 0);
    assert_eq!(capture_files(&host, b"{}").status, 400);
}

/// law: capture-holds-conversation
#[test]
fn a_capture_without_conversation_content_is_refused_and_not_spooled() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
    // What a stale extension background sent: identity and provenance only.
    let empty = json!({
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {"captured_at": "2026-11-02T16:20:59Z", "source_url": "https://chatgpt.com/c/empty"},
        "session": {"provider": "chatgpt", "provider_session_id": "empty", "session_kind": "standard", "attachments": []}
    });
    let a = store_capture(&host, empty.to_string().as_bytes());
    assert_eq!(a.status, 400);
    assert_eq!(a.json()["reason"]["name"], "capture_has_no_conversation");
    let mut empty_raw = empty.clone();
    empty_raw["raw_provider_payload"] = json!({});
    let b = store_capture(&host, empty_raw.to_string().as_bytes());
    assert_eq!(b.json()["reason"]["name"], "capture_has_no_conversation");
    assert_eq!(
        exchange(
            &host,
            json!({"version":2,"type":"spooled","provider":"chatgpt","ids":["empty"]})
        )
        .json()["spooled"],
        json!([false])
    );
}

#[test]
fn a_large_capture_is_accepted() {
    let w = test_daemon::start("x");
    let host = host(&w);
    let bytes = capture("conv-large", &"a long conversation ".repeat(200_000));
    assert!(bytes.len() > 3 * 1024 * 1024);
    let read = capture_files(&host, &bytes);
    assert_eq!(read.status, 200, "{:?}", read.json());
    let sent = store_capture(&host, &bytes);
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
        || match crate::client::call(
            &host.socket,
            &crate::client::request(
                "rows",
                serde_json::json!({"selection": format!("session:{session}"), "steps":[{"verb":"read", "format":"markdown"}]}),
                uuid::Uuid::now_v7().to_string(),
            ),
        ) {
            Ok(answer)
                if answer.value().is_some_and(|value| {
                    value["reply"]["steps"][0]["outcome"]["status"] == "done"
                }) =>
            {
                let page = crate::test_answers::read(answer);
                let mut output = Vec::new();
                polylogue_render::encode_document(
                    &page.document,
                    polylogue_render::Format::Markdown,
                    polylogue_render::EncodeOptions {
                        commands: Some(crate::commands::declarations()),
                        ..Default::default()
                    },
                    &mut output,
                )
                .unwrap();
                Value::String(String::from_utf8(output).unwrap())
            }
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
fn grok_capture_with_fetched_file_bytes_reaches_an_archive_read() {
    let w = test_daemon::start("fixture");
    let host = host(&w);
    let body = serde_json::to_vec(&grok_capture()).unwrap();
    let read = capture_files(&host, &body);
    assert_eq!(read.status, 200, "{:?}", read.json());
    assert_eq!(read.json()["files"][0]["id"], "raw-file");
    assert_eq!(read.json()["files"][0]["url"], "users/u/raw-file/content");
    let mut capture = grok_capture();
    capture["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "content_base64":B64.encode(b"fetched file bytes")}]);
    let sent = store_capture(&host, &serde_json::to_vec(&capture).unwrap());
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
    let sent = store_capture(&host, &serde_json::to_vec(&first).unwrap());
    assert_eq!(sent.status, 200);
    let hash = polylogue_model::BlobHash::of(b"fetched file bytes");
    grok_file_blobs(&w, &host, "raw provider question");

    let mut later = grok_capture();
    later["raw_provider_payload"]["responses"]["responses"][0]["message"] =
        json!("raw provider question, edited");
    let read = capture_files(&host, &serde_json::to_vec(&later).unwrap());
    assert_eq!(read.status, 200, "{:?}", read.json());
    assert_eq!(
        call(
            &host,
            "internal stored-files",
            json!({"provider":"grok","conversation":"grok-raw"})
        )
        .json()["files"][0],
        json!({"id":"raw-file","blob": hash.to_string(), "size": 18})
    );
    later["session"]["attachments"] = json!([{"provider_attachment_id":"raw-file", "message_provider_id":"raw-r1", "blob": hash.to_string(), "size_bytes": 18}]);
    let sent = store_capture(&host, &serde_json::to_vec(&later).unwrap());
    assert_eq!(sent.status, 200);
    assert_eq!(grok_file_blobs(&w, &host, "edited"), [Some(hash)]);
}

#[test]
fn a_frame_claiming_more_than_arrives_fails_at_the_bytes_that_exist() {
    // A header claiming 4 GiB followed by three bytes: the reader fails at
    // the bytes that exist.
    let mut input = u32::MAX.to_ne_bytes().to_vec();
    input.extend_from_slice(b"{}x");
    let error = read_frame(&mut input.as_slice()).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    assert_eq!(error.to_string(), "frame ended after 3 of 4294967295 bytes");
    let mut whole = Vec::new();
    write_frame(
        &mut whole,
        &json!({"version":2,"type":"spooled","provider":"chatgpt","ids":[]}),
    )
    .unwrap();
    assert_eq!(
        read_frame(&mut whole.as_slice()).unwrap(),
        Some(json!({"version":2,"type":"spooled","provider":"chatgpt","ids":[]}))
    );
}

mod notes;
mod repeated_captures;

#[test]
fn expired_sending_action_accepts_only_its_holders_late_confirmation() {
    use std::sync::atomic::{AtomicI64, Ordering};
    let now = Arc::new(AtomicI64::new(1_800_000_000_000));
    let clock = Arc::clone(&now);
    let w = test_daemon::start_with_clock(
        "synthetic chat context",
        polylogue_model::WallClock::new(move || {
            jiff::Timestamp::from_millisecond(clock.load(Ordering::SeqCst)).unwrap()
        }),
    );
    w.set("browser.actions", "on");
    let host = host(&w);
    let queued = send(&w, "late-capture", chat_request(None, true), vec![]).unwrap();
    let id = queued["id"].as_str().unwrap();
    assert_eq!(
        call(&host, "internal chat claim", json!({"by":"tab-a"})).json()["action"]["id"],
        id
    );
    for phase in ["sending", "claimed"] {
        let response = call(
            &host,
            "internal chat report",
            json!({"id":id,"update":json!({
                "by":"tab-a", "outcome":"progress", "phase":phase,
            })}),
        );
        assert_eq!(response.status, 200, "{}", response.json());
    }
    now.fetch_add(polylogue_model::user::CHAT_CLAIM_MS + 1, Ordering::SeqCst);
    let listed = crate::test_daemon::Call::new("rows", json!({"selection": "chat_actions"}))
        .answer(&w.daemon);
    let action = &listed.value().expect("chat actions")["reply"]["rows"][0];
    assert_eq!(action["state"], "outcome_unknown");
    assert_eq!(action["phase"], "claimed");
    assert_eq!(action["claimed_by"], Value::Null);
    assert_eq!(
        call(&host, "internal chat claim", json!({"by":"tab-b"})).json()["action"],
        Value::Null
    );
    assert_eq!(report(&host, id, "tab-a", "progress").status, 409);
    let missing_receipt = report(&host, id, "tab-a", "submitted");
    assert_eq!(missing_receipt.status, 400);
    assert_eq!(
        missing_receipt.json()["error"],
        "submission_receipt_required"
    );
    let confirmation = |by| {
        call(
            &host,
            "internal chat report",
            json!({"id":id,"update":json!({
                "by":by, "outcome":"submitted", "phase":"submitted",
                "receipt":{"conversation":"conversation-1", "turn":"captured-turn"},
            })}),
        )
    };
    assert_eq!(confirmation("tab-b").status, 409);
    let response = confirmation("tab-a");
    assert_eq!(response.status, 200, "{}", response.json());
    assert_eq!(response.json()["action"]["status"], "submitted");
    assert_eq!(
        response.json()["action"]["receipt"]["turn"],
        "captured-turn"
    );
}

#[test]
fn malformed_captures_and_spool_symlinks_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let host = host_without_daemon(dir.path());
    let refused = store_capture(&host, b"[1]").json();
    assert_eq!(
        refused["reason"],
        json!({"name":"capture_not_json","fields":{}})
    );
    let log = host.capture_log("chatgpt", "linked");
    let target = dir.path().join("elsewhere");
    std::fs::write(&target, b"untouched").unwrap();
    std::fs::create_dir_all(log.parent().unwrap()).unwrap();
    std::os::unix::fs::symlink(&target, &log).unwrap();
    assert_eq!(
        store_capture(&host, &capture("linked", "x")).json()["error"],
        "io"
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"untouched");
}
