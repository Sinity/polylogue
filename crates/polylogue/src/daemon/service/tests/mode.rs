use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn path(home: &ClaudeHome) -> std::path::PathBuf {
    home.dir.path().join("config/polylogue/sinex.toml")
}

fn startup(home: &ClaudeHome) -> crate::config::Startup {
    use crate::config::{EnvSnapshot, Identity, Overrides, ProcessInputs, Startup};
    Startup::load(
        &ProcessInputs {
            environment: EnvSnapshot::from_values([
                ("HOME".into(), home.home.clone().into_os_string()),
                (
                    "XDG_CONFIG_HOME".into(),
                    home.dir.path().join("config").into_os_string(),
                ),
            ]),
            output: crate::config::OutputConfig::new(false, false, true, None),
            clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
            identity: Some(Identity {
                machine: "synthetic-machine".into(),
                host: "synthetic-host".into(),
            }),
            workers: Some(1),
            cwd: Some(home.dir.path().into()),
        },
        Overrides {
            archive: Some(home.root.clone()),
            listen: Some("127.0.0.1:0".parse().unwrap()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn loaded(home: &ClaudeHome, text: &str) -> crate::config::Startup {
    let path = path(home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
    startup(home)
}

fn primary(home: &ClaudeHome) -> crate::config::Startup {
    loaded(
        home,
        &format!(
            "mode = 'primary'\nnats = 'nats://127.0.0.1:1'\nstore = {:?}\n",
            home.dir.path().join("sinex-store"),
        ),
    )
}

fn launch(startup: crate::config::Startup) -> std::io::Result<Runtime> {
    let attempt = startup
        .mode
        .needs_engine()
        .then(polylogue_testkit::stopped_attempt);
    start(
        startup.config,
        startup.engine,
        startup.mode,
        polylogue_engine::Lifetime::UntilStopped,
        attempt,
    )
}

fn read_request() -> crate::test_daemon::Call {
    crate::test_daemon::Call::new(
        "read",
        serde_json::json!({"target": "claude-code:s1", "format": "json"}),
    )
}

fn seeded() -> ClaudeHome {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "Stored question"));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    drop(engine);
    drop(_lock);
    w
}

fn blocked(runtime: &Runtime, path: &std::path::Path) {
    assert!(runtime.daemon.writes.is_none());
    assert!(runtime.engine.is_none());
    assert!(runtime.signals.is_none());
    assert!(runtime.signals_stop.is_none());
    // Sinex is unavailable while the configuration is broken, so status is degraded.
    let (status, _) = crate::test_answers::degraded(
        crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon),
    );
    assert_eq!(status["mode"]["state"], "blocked");
    assert_eq!(status["mode"]["path"], path.to_str().unwrap());
    assert!(!status["mode"]["error"].as_str().unwrap().is_empty());
    let requests = vec![
        crate::test_daemon::Call::new(
            "chat send",
            serde_json::json!({"provider": "chatgpt", "text": "Synthetic draft", "model": "X"}),
        )
        .id("send"),
        crate::test_daemon::Call::new("backfill start", serde_json::json!({"provider": "chatgpt"})),
        crate::test_daemon::Call::new("import", serde_json::json!({})).id("import"),
        crate::test_daemon::Call::new(
            "setting",
            serde_json::json!({"key": Some("browser.actions"), "value": Some("on")}),
        ),
        crate::test_daemon::Call::new("recall", serde_json::json!({"query": "question"}))
            .id("delivery")
            .surface(crate::commands::Surface::Hook),
        crate::test_daemon::Call::new("keep", serde_json::json!({"query": "sessions"})).id("keep"),
        crate::test_daemon::Call::new("embed build", serde_json::json!({"yes": true})),
    ];
    for request in requests {
        assert!(
            matches!((request).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. }
            if error.code.as_deref() == Some("sinex_unavailable")
                && error.hint.as_deref().is_some_and(|hint| hint.contains(path.to_str().unwrap())))
        );
    }
}

#[test]
fn malformed_config_keeps_primary_status_and_refuses_reads_and_writes() {
    for text in [
        "mode = [",
        "mode = 'unknown'\nnats = 'invalid://'\nstore = '/unused'\n",
    ] {
        let w = ClaudeHome::new();
        let store = w.dir.path().join("sinex-store");
        drop(Archive::open_primary(&w.root, &store).unwrap());
        w.append(
            "pending.jsonl",
            &user("later", "pending", "Must wait for configuration"),
        );
        let startup = loaded(&w, text);
        assert!(startup.config.standalone_day().is_err());
        let runtime = launch(startup).unwrap();
        blocked(&runtime, &path(&w));
        assert!(
            matches!((read_request()).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("sinex_unavailable"))
        );
        assert!(
            crate::test_answers::ok(
                crate::test_daemon::Call::new(
                    "setting",
                    serde_json::json!({"key": Some("browser.actions")})
                )
                .answer(&runtime.daemon)
            )["authored"]
                .is_null()
        );
        assert!(w.file("pending.jsonl").exists());
        assert!(!w.root.join("archive.db").exists());
    }
}

#[test]
fn malformed_primary_configuration_does_not_reclassify_a_standalone_archive() {
    let w = seeded();
    let refused = launch(loaded(&w, "mode = ["));
    assert!(refused.is_err());
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 1);
    assert!(
        Archive::open_or_create(&w.root).is_ok(),
        "failed startup retained its lock"
    );
}

#[test]
fn missing_config_is_standalone() {
    let w = seeded();
    let startup = startup(&w);
    assert!(matches!(startup.mode, ArchiveMode::Standalone));
    let runtime = launch(startup).unwrap();
    assert!(runtime.daemon.writes.is_some());
    assert_eq!(
        crate::test_answers::ok(
            crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon)
        )["mode"]["state"],
        "standalone"
    );
}

#[test]
fn primary_outage_preserves_raw_custody_without_standalone_fallback_or_owner_day() {
    for existing in [false, true] {
        let w = ClaudeHome::new();
        let store = w.dir.path().join("sinex-store");
        if existing {
            drop(Archive::open_primary(&w.root, &store).unwrap());
        }
        w.append(
            "later.jsonl",
            &user("later", "later", "Must wait for Sinex"),
        );
        let startup = primary(&w);
        assert!(startup.config.standalone_day().is_err());
        let runtime = launch(startup).unwrap();
        assert!(runtime.daemon.archive.is_primary());
        assert!(runtime.daemon.writes.is_some());
        // Sinex is not running, so status is degraded.
        let (status, _) = crate::test_answers::degraded(
            crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon),
        );
        assert_eq!(status["mode"]["state"], "primary");
        assert!(
            matches!((read_request()).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. }
            if matches!(error.code.as_deref(), Some("primary_context_missing" | "primary_context_unavailable")))
        );
        let ModeState::Primary { connection } = &runtime.daemon.mode else {
            panic!("primary mode required")
        };
        while connection.status().unwrap().prepared_materials == 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(connection.status().unwrap().prepared_bytes > 0);
        assert!(w.file("later.jsonl").exists());
        assert!(!w.root.join("archive.db").exists());
        let reader = runtime.daemon.reads.reader().unwrap();
        assert_eq!(reader.confirmed_checkpoint().unwrap(), None);
        assert_eq!(
            reader
                .query("SELECT count(*) AS n FROM message", &[])
                .unwrap()[0]["n"],
            0
        );
    }
}
