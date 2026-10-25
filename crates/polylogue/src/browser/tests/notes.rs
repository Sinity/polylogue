use super::*;
use crate::daemon::{Request, Response, server, service};

/// Every note, as the `notes` query lists them.
fn notes(ctx: &Ctx) -> Vec<Value> {
    let Some(Response::Ok { data }) = ctx.ask(&Request::Query {
        text: "notes".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    }) else {
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
fn selected_passage_notes_survive_restart_and_identical_native_retries() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().join("home");
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let mut runtime =
        service::start(&root, config.clone(), service::ArchiveMode::Standalone).unwrap();
    let ctx = Arc::new(Ctx::at(&config, &runtime.daemon.archive).unwrap());
    let mut socket = server::Server::start(Arc::clone(&runtime.daemon)).unwrap();
    assert_eq!(
        exchange(
            &ctx,
            "POST",
            "/v1/browser-captures",
            &capture("selection-session", "Exact  provider\npassage")
        )
        .status,
        200
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if matches!(service::handle(Request::Status, &runtime.daemon), Response::Ok {data} if data["store"]["messages"] == 1)
        {
            break;
        }
        assert!(Instant::now() < deadline, "capture did not converge");
        std::thread::sleep(Duration::from_millis(10));
    }
    let input = selection();
    let saved = exchange(
        &ctx,
        "POST",
        "/v1/notes/selection",
        input.to_string().as_bytes(),
    );
    assert_eq!(saved.status, 200, "{}", saved.json());
    assert_eq!(saved.json()["note"]["id"], input["request_id"]);
    assert_eq!(saved.json()["note"]["repeat"], false);
    runtime.stop_accepting();
    socket.shutdown().unwrap();
    runtime.shutdown().unwrap();
    drop(runtime);
    let mut runtime = service::start(&root, config, service::ArchiveMode::Standalone).unwrap();
    let mut socket = server::Server::start(Arc::clone(&runtime.daemon)).unwrap();
    let retry = exchange(
        &ctx,
        "POST",
        "/v1/notes/selection",
        input.to_string().as_bytes(),
    );
    assert_eq!(retry.status, 200, "{}", retry.json());
    assert_eq!(retry.json()["note"]["id"], input["request_id"]);
    assert_eq!(retry.json()["note"]["repeat"], true);
    let data = notes(&ctx);
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["author"], "operator");
    assert_eq!(
        data[0]["target"],
        "message:chatgpt:t1 session:chatgpt:selection-session"
    );
    let value: Value = serde_json::from_str(data[0]["value"].as_str().unwrap()).unwrap();
    assert_eq!(value["body"], input["body"]);
    assert_eq!(value["quote"], input["quote"]);
    assert_eq!(value["source"], input["source"]);
    let mut changed = input;
    changed["body"] = "An edit needs a new note ID".into();
    let conflict = exchange(
        &ctx,
        "POST",
        "/v1/notes/selection",
        changed.to_string().as_bytes(),
    );
    assert_eq!(conflict.status, 409, "{}", conflict.json());
    runtime.stop_accepting();
    socket.shutdown().unwrap();
    runtime.shutdown().unwrap();
}

#[test]
fn missing_or_mismatched_browser_note_targets_never_acknowledge_a_save() {
    let w = test_daemon::start("unrelated");
    let ctx = ctx(&w);
    let input = selection();
    let missing = exchange(
        &ctx,
        "POST",
        "/v1/notes/selection",
        input.to_string().as_bytes(),
    );
    assert_eq!(missing.status, 409, "{}", missing.json());
    for field in ["message_id", "conversation_id", "page_url"] {
        let mut bad = input.clone();
        bad["source"][field] = "".into();
        assert_eq!(
            exchange(
                &ctx,
                "POST",
                "/v1/notes/selection",
                bad.to_string().as_bytes()
            )
            .status,
            400
        );
    }
    let data = notes(&ctx);
    assert!(data.is_empty());
}

#[test]
fn a_native_message_id_shared_by_two_conversations_is_not_guessed() {
    let w = test_daemon::start("unrelated");
    let ctx = ctx(&w);
    for session in ["selection-session", "another-session"] {
        assert_eq!(
            exchange(
                &ctx,
                "POST",
                "/v1/browser-captures",
                &capture(session, "shared native ID")
            )
            .status,
            200
        );
    }
    w.wait_for(|status| status["store"]["sessions"] == 3);
    let refused = exchange(
        &ctx,
        "POST",
        "/v1/notes/selection",
        selection().to_string().as_bytes(),
    );
    assert_eq!(refused.status, 409, "{}", refused.json());
    let data = notes(&ctx);
    assert!(data.is_empty());
}
