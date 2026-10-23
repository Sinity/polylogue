use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn loaded(home: &ClaudeHome, text: &str) -> Backing {
    let path = home.dir.path().join("sinex.toml");
    std::fs::write(&path, text).unwrap();
    crate::sinex::configured_at(&path)
}

fn primary(home: &ClaudeHome, restore: bool) -> Backing {
    loaded(
        home,
        &format!(
            "mode = 'primary'\nnats = 'invalid://'\nstore = {:?}\n{}",
            home.dir.path().join("sinex-store"),
            if restore {
                "archive = 'synthetic-archive'\n"
            } else {
                ""
            }
        ),
    )
}

fn seeded() -> ClaudeHome {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "Retained question"));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    drop(engine);
    drop(_lock);
    w
}

fn data(request: Request, runtime: &Runtime) -> serde_json::Value {
    match handle(request, &runtime.daemon) {
        Response::Ok { data } => data,
        other => panic!("read refused: {other:?}"),
    }
}

fn blocked(runtime: &Runtime, path: &std::path::Path) {
    assert!(runtime.daemon.writes.is_none());
    assert!(runtime.engine.is_none());
    assert!(runtime.signals.is_none());
    assert!(runtime.signals_stop.is_none());
    assert!(runtime.publisher.is_none());
    let status = data(Request::Status, runtime);
    assert_eq!(status["backing"]["state"], "blocked");
    assert_eq!(status["backing"]["path"], path.to_str().unwrap());
    assert!(!status["backing"]["error"].as_str().unwrap().is_empty());
    let ask = crate::briefing::Ask::Recall {
        text: "question".into(),
        cwd: None,
        for_session: None,
    };
    let requests = vec![
        Request::Import {
            request_id: "import".into(),
            paths: vec![],
        },
        Request::Setting {
            request_id: None,
            key: "test".into(),
            value: Some("changed".into()),
            machine: false,
        },
        Request::Context {
            request_id: "delivery".into(),
            ask,
            sender: Some("hook".into()),
        },
        Request::Hook {
            event: "synthetic".into(),
            payload: serde_json::json!({}),
        },
        Request::Keep {
            request_id: "keep".into(),
            text: "sessions".into(),
            name: None,
        },
        Request::Excise {
            request_id: "remove".into(),
            target: "session:claude-code:s1".into(),
            reason: None,
            dry_run: false,
        },
        Request::Embed {
            what: "build".into(),
            query: None,
            limit: None,
            yes: true,
        },
    ];
    for request in requests {
        assert!(
            matches!(handle(request, &runtime.daemon), Response::Error { kind, detail: Some(detail), .. }
            if kind == "backing_unavailable" && detail["config_path"] == path.to_str().unwrap())
        );
    }
}

#[test]
fn malformed_config_keeps_reads_and_status_but_creates_no_writing_workers() {
    for text in [
        "mode = [",
        "mode = 'unknown'\nnats = 'invalid://'\nstore = '/unused'\n",
    ] {
        let w = seeded();
        w.append(
            "pending.jsonl",
            &user("later", "pending", "Must wait for configuration"),
        );
        let backing = loaded(&w, text);
        let runtime = start(&w.root, w.config(64 << 20), backing).unwrap();
        blocked(&runtime, &w.dir.path().join("sinex.toml"));
        let shown = data(
            Request::Read {
                session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                view: "full".into(),
                format: "json".into(),
                logical: false,
                budget: None,
                cursor: None,
            },
            &runtime,
        );
        assert_eq!(shown["display_title"], "Retained question");
        assert_eq!(shown["messages"].as_array().unwrap().len(), 1);
        assert_eq!(w.count("SELECT count(*) AS n FROM message"), 1);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM session WHERE id='claude-code:pending'"),
            0
        );
        assert!(
            data(
                Request::Setting {
                    request_id: None,
                    key: "test".into(),
                    value: None,
                    machine: false
                },
                &runtime
            )["value"]
                .is_null()
        );
        let context = data(
            Request::Context {
                request_id: "read-context".into(),
                ask: crate::briefing::Ask::Recall {
                    text: "Retained".into(),
                    cwd: None,
                    for_session: None,
                },
                sender: None,
            },
            &runtime,
        );
        assert!(context["delivery"].is_null());
        assert_eq!(w.count("SELECT count(*) AS n FROM a.delivery"), 0);
        assert!(w.file("pending.jsonl").exists());
    }
}

#[test]
fn missing_config_is_standalone_and_nonempty_primary_is_readable_offline() {
    let w = seeded();
    let missing = crate::sinex::configured_at(&w.dir.path().join("missing.toml"));
    let mut runtime = start(&w.root, w.config(64 << 20), missing).unwrap();
    assert!(runtime.daemon.writes.is_some());
    assert_eq!(
        data(Request::Status, &runtime)["backing"]["state"],
        "standalone"
    );
    runtime.shutdown().unwrap();
    let runtime = start(&w.root, w.config(64 << 20), primary(&w, true)).unwrap();
    assert!(runtime.daemon.writes.is_some());
    assert!(runtime.publisher.is_some());
    assert_eq!(
        data(Request::Status, &runtime)["backing"]["state"],
        "primary"
    );
    assert_eq!(w.texts("s1"), ["Retained question"]);
}

#[test]
fn required_empty_restore_failure_never_collects_fresh_sources() {
    let w = ClaudeHome::new();
    w.append(
        "new.jsonl",
        &user("new", "new", "Cannot become a fresh standalone archive"),
    );
    let runtime = start(&w.root, w.config(64 << 20), primary(&w, true)).unwrap();
    blocked(&runtime, &w.dir.path().join("sinex.toml"));
    let state = data(Request::Status, &runtime)["backing"].clone();
    assert_eq!(state["mode"], "primary");
    assert!(
        state["error"]
            .as_str()
            .unwrap()
            .contains("restoring archive synthetic-archive")
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM input_ref"), 0);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 0);
}

#[test]
fn inventory_errors_are_blocked_startup_not_a_nonempty_primary() {
    let w = seeded();
    let archive = rusqlite::Connection::open(w.root.join("archive.db")).unwrap();
    // An invalid publication identity does not change the excision generation
    // or invalidate the retained read index.
    archive.execute("INSERT INTO note(id,root,kind,author,machine,created_at,commit_seq) VALUES ('','','note','operator','synthetic',1,1)", []).unwrap();
    drop(archive);
    let runtime = start(&w.root, w.config(64 << 20), primary(&w, false)).unwrap();
    blocked(&runtime, &w.dir.path().join("sinex.toml"));
    let state = data(Request::Status, &runtime)["backing"].clone();
    assert!(
        state["error"]
            .as_str()
            .unwrap()
            .contains("local Sinex inventory")
    );
    assert!(
        state["error"]
            .as_str()
            .unwrap()
            .contains("invalid publication unit identity")
    );
    assert_eq!(w.texts("s1"), ["Retained question"]);
}
