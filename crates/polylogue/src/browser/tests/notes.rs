use super::*;
use crate::daemon::{server, service};

/// Every current note assertion.
fn notes(host: &BrowserHost) -> Vec<Value> {
    let Some(Ok(data)) = host
        .ask(
            "query",
            serde_json::json!({"query": "assertions where kind:note"}),
        )
        .map(crate::protocol::into_value)
    else {
        panic!("notes unavailable")
    };
    data["rows"].as_array().unwrap().clone()
}

fn selection() -> Value {
    json!({
        "request_id":"019c6e27-e55b-73d1-87d8-4e01f1f75043",
        "body":"My own reading of the passage",
        "quote":"Exact  provider\npassage",
        "source": {"provider":"chatgpt","conversation_id":"selection-session","message_id":"t1",
            "page_url":"https://chatgpt.com/c/selection-session","selected_at":"2026-10-24T00:00:00Z"}
    })
}

#[test]
fn selected_passage_notes_survive_restart_and_identical_retries() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().join("home");
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let mut runtime =
        crate::test_daemon::start_runtime(&root, config.clone(), service::ArchiveMode::Standalone)
            .unwrap();
    let host = Arc::new(BrowserHost::at(&config, &runtime.daemon.archive).unwrap());
    let mut socket =
        server::Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations))
            .unwrap();
    assert_eq!(
        exchange(
            &host,
            "POST",
            "/v1/captures",
            &capture("selection-session", "Exact  provider\npassage")
        )
        .status,
        200
    );
    loop {
        if matches!(crate::protocol::into_value(crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon)), Ok(data) if data["store"]["messages"] == 1 && data["busy"] == false)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let input = selection();
    let saved = exchange(&host, "POST", "/v1/notes", input.to_string().as_bytes());
    assert_eq!(saved.status, 200, "{}", saved.json());
    assert_eq!(saved.json()["note"]["id"], input["request_id"]);
    assert_eq!(saved.json()["note"]["repeat"], false);
    runtime.stop_accepting();
    socket.shutdown().unwrap();
    runtime.shutdown().unwrap();
    drop(runtime);
    let mut runtime =
        crate::test_daemon::start_runtime(&root, config, service::ArchiveMode::Standalone).unwrap();
    let mut socket =
        server::Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations))
            .unwrap();
    let retry = exchange(&host, "POST", "/v1/notes", input.to_string().as_bytes());
    assert_eq!(retry.status, 200, "{}", retry.json());
    assert_eq!(retry.json()["note"]["id"], input["request_id"]);
    assert_eq!(retry.json()["note"]["repeat"], true);
    let data = notes(&host);
    assert_eq!(data.len(), 1);
    // This in-process host uses the actual test process socket identity.
    // The external browser-host test checks Extension -> Operator attribution.
    let request = runtime
        .daemon
        .reads
        .reader()
        .unwrap()
        .user_request(input["request_id"].as_str().unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(data[0]["author"], request.authorship.author.to_string());
    assert_eq!(
        data[0]["about"],
        "message:chatgpt:t1 session:chatgpt:selection-session"
    );
    assert_eq!(
        data[0]["value"],
        json!(
            "My own reading of the passage\n\nQuoted selection:\nExact  provider\npassage\n\nSource: https://chatgpt.com/c/selection-session\nSelected at: 2026-10-24T00:00:00Z"
        )
    );
    let mut changed = input;
    changed["body"] = "An edit needs a new note ID".into();
    let conflict = exchange(&host, "POST", "/v1/notes", changed.to_string().as_bytes());
    assert_eq!(conflict.status, 409, "{}", conflict.json());
    runtime.stop_accepting();
    socket.shutdown().unwrap();
    runtime.shutdown().unwrap();
}

#[test]
fn missing_or_mismatched_browser_note_targets_never_acknowledge_a_save() {
    let w = test_daemon::start("unrelated");
    let host = host(&w);
    let input = selection();
    let missing = exchange(&host, "POST", "/v1/notes", input.to_string().as_bytes());
    assert_eq!(missing.status, 409, "{}", missing.json());
    for field in ["message_id", "conversation_id", "page_url"] {
        let mut bad = input.clone();
        bad["source"][field] = "".into();
        assert_eq!(
            exchange(&host, "POST", "/v1/notes", bad.to_string().as_bytes()).status,
            400
        );
    }
    let data = notes(&host);
    assert!(data.is_empty());
}

#[test]
fn a_native_message_id_shared_by_two_conversations_is_not_guessed() {
    let w = test_daemon::start("unrelated");
    let host = host(&w);
    for session in ["selection-session", "another-session"] {
        assert_eq!(
            exchange(
                &host,
                "POST",
                "/v1/captures",
                &capture(session, "shared native ID")
            )
            .status,
            200
        );
    }
    w.wait_for(|status| status["store"]["sessions"] == 3);
    let refused = exchange(
        &host,
        "POST",
        "/v1/notes",
        selection().to_string().as_bytes(),
    );
    assert_eq!(refused.status, 409, "{}", refused.json());
    let data = notes(&host);
    assert!(data.is_empty());
}
