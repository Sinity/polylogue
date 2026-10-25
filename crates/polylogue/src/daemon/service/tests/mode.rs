use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn loaded(home: &ClaudeHome, text: &str) -> ArchiveMode {
    let path = home.dir.path().join("sinex.toml");
    std::fs::write(&path, text).unwrap();
    crate::sinex::configured_at(&path)
}

/// Primary mode on a Sinex that cannot be reached.
fn primary(home: &ClaudeHome) -> ArchiveMode {
    loaded(
        home,
        &format!(
            "mode = 'primary'\nnats = 'invalid://'\nstore = {:?}\n",
            home.dir.path().join("sinex-store"),
        ),
    )
}

/// A Sinex that holds nothing and accepts nothing.
struct Empty;

impl polylogue_store::primary::Sinex for Empty {
    fn send_part(
        &self,
        _: &polylogue_store::primary::Part,
        _: &mut dyn std::io::Read,
    ) -> polylogue_store::Result<String> {
        Err(polylogue_store::StoreError::Sinex("synthetic".into()))
    }
    fn send_record(&self, _: &polylogue_store::primary::Record) -> polylogue_store::Result<String> {
        Err(polylogue_store::StoreError::Sinex("synthetic".into()))
    }
    fn excise(&self, _: &[String]) -> polylogue_store::Result<()> {
        Err(polylogue_store::StoreError::Sinex("synthetic".into()))
    }
    fn held(&self, _: i64) -> polylogue_store::Result<Vec<polylogue_store::primary::Held>> {
        Ok(vec![])
    }
    fn send_sessions(&self, _: &polylogue_store::ReadPool) -> polylogue_store::Result<()> {
        Ok(())
    }
}

fn seeded() -> ClaudeHome {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "Stored question"));
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
    let status = data(Request::Status, runtime);
    assert_eq!(status["mode"]["state"], "blocked");
    assert_eq!(status["mode"]["path"], path.to_str().unwrap());
    assert!(!status["mode"]["error"].as_str().unwrap().is_empty());
    let ask = crate::briefing::Ask::Recall {
        text: "question".into(),
        cwd: None,
        for_session: None,
    };
    let requests =
        vec![
        Request::ChatUpload { bytes: vec![1, 2, 3] },
        Request::ChatEnqueue { action: serde_json::from_value(serde_json::json!({
            "provider":"chatgpt", "operation":"conversation.create", "text":"Synthetic draft",
            "presentation":{"model_slug":"x","model_label":"X","effort_label":"standard"}
        })).unwrap() },
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
            if kind == "sinex_unavailable" && detail["config_path"] == path.to_str().unwrap())
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
        let mode = loaded(&w, text);
        let runtime = start(&w.root, w.config(64 << 20), mode).unwrap();
        blocked(&runtime, &w.dir.path().join("sinex.toml"));
        let shown = data(
            Request::Read {
                target: crate::daemon::ReadTarget::Session(
                    polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                ),
                view: "full".into(),
                format: "json".into(),
                logical: false,
                budget: None,
                cursor: None,
            },
            &runtime,
        );
        assert_eq!(shown["display_title"], "Stored question");
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
                    text: "Stored".into(),
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
fn missing_config_is_standalone() {
    let w = seeded();
    let missing = crate::sinex::configured_at(&w.dir.path().join("missing.toml"));
    let runtime = start(&w.root, w.config(64 << 20), missing).unwrap();
    assert!(runtime.daemon.writes.is_some());
    assert_eq!(
        data(Request::Status, &runtime)["mode"]["state"],
        "standalone"
    );
}

#[test]
fn primary_without_sinex_or_a_local_copy_does_not_start() {
    let w = ClaudeHome::new();
    w.append(
        "new.jsonl",
        &user("new", "new", "Cannot become a fresh standalone archive"),
    );
    let refused = start(&w.root, w.config(64 << 20), primary(&w));
    let error = refused.err().expect("no archive to open").to_string();
    assert!(error.contains("no local copy"), "{error}");
    assert!(!w.root.join("archive.db").exists());
    assert!(!w.root.join("sinex.db").exists());
}

#[test]
fn primary_with_a_local_copy_reads_offline_and_pauses_writes() {
    let w = ClaudeHome::new();
    w.append(
        "later.jsonl",
        &user("later", "later", "Must wait for Sinex"),
    );
    let store = w.dir.path().join("sinex-store");
    drop(polylogue_store::Archive::open_primary(&w.root, &store, &Empty).unwrap());
    let runtime = start(&w.root, w.config(64 << 20), primary(&w)).unwrap();
    blocked(&runtime, &w.dir.path().join("sinex.toml"));
    assert_eq!(
        data(Request::Status, &runtime)["mode"]["configured"],
        "primary"
    );
    assert!(!w.root.join("archive.db").exists());
}
