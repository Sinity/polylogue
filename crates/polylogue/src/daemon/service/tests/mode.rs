use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn path(home: &ClaudeHome) -> std::path::PathBuf {
    home.root.join("sinex.toml")
}

fn startup(home: &ClaudeHome) -> crate::config::Startup {
    use crate::config::{Client, EnvSnapshot, Identity, Overrides, ProcessInputs, Startup};
    let process = ProcessInputs {
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
        memory_bytes: Some(4 << 30),
        workers: Some(1),
        cwd: Some(home.dir.path().into()),
    };
    Client::load(
        &process,
        Overrides {
            archive: Some(home.root.clone()),
            listen: Some("127.0.0.1:0".parse().unwrap()),
            ..Default::default()
        },
    )
    .and_then(|client| Startup::extend(client, &process))
    .unwrap()
}

fn loaded(home: &ClaudeHome, text: &str) -> crate::config::Startup {
    let path = path(home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
    startup(home)
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
            "start",
            serde_json::json!({"in": "chatgpt", "text": "Synthetic draft", "model": "X"}),
        )
        .id("send"),
        crate::test_daemon::Call::new("import", serde_json::json!({"paths": ["chatgpt"]})),
        crate::test_daemon::Call::new("import", serde_json::json!({})).id("import"),
        crate::test_daemon::Call::new(
            "set",
            serde_json::json!({"key": Some("browser.actions"), "value": Some("on")}),
        ),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"question", "steps":[{"verb":"brief"}]}),
        )
        .id("delivery")
        .channel(command_surface::Via::Hook),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"sessions", "steps":[{"verb":"save"}]}),
        )
        .id("keep"),
        crate::test_daemon::Call::new("embed", serde_json::json!({"yes": true})),
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
fn malformed_primary_configuration_does_not_reclassify_a_standalone_archive() {
    let w = seeded();
    let runtime = launch(loaded(&w, "mode = [")).unwrap();
    blocked(&runtime, &path(&w));
    drop(runtime);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 1);
    assert!(
        Archive::open_or_create(&w.root).is_ok(),
        "failed startup kept its lock"
    );
}

#[test]
fn missing_config_is_standalone() {
    let w = seeded();
    let startup = startup(&w);
    assert!(matches!(startup.mode, ArchiveMode::Standalone));
    let runtime = launch(startup).unwrap();
    assert!(runtime.daemon.writes.is_some());
    // The mode is in every status answer, also one degraded while the
    // first pass still computes session figures.
    let status =
        crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon);
    let status = status.value().unwrap_or_else(|| panic!("{status:?}"));
    assert_eq!(status["mode"]["state"], "standalone");
}

/// The names and bytes of every file under `root`, to show a refused start
/// left the archive as it was.
fn contents(root: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    let mut found = std::collections::BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.insert(path.clone(), std::fs::read(&path).unwrap());
            }
        }
    }
    found
}

fn refused_as_not_built(started: std::io::Result<Runtime>) {
    let Err(error) = started else {
        panic!("primary mode started");
    };
    let reason = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<StoreError>());
    assert!(
        matches!(
            reason,
            Some(StoreError::Refused(polylogue_model::Reason::ModeNotBuilt { mode })) if mode == "primary"
        ),
        "{error}"
    );
}

#[test]
fn primary_mode_is_not_built_until_its_parts_land() {
    // An archive whose meta.json records primary mode.
    let w = ClaudeHome::new();
    std::fs::create_dir_all(&w.root).unwrap();
    std::fs::write(w.root.join("meta.json"), "{\"mode\":\"primary\"}\n").unwrap();
    std::fs::write(w.root.join("index.db"), b"synthetic primary index").unwrap();
    let before = contents(&w.root);
    refused_as_not_built(launch(startup(&w)));
    assert_eq!(contents(&w.root), before);

    // A configuration asking for primary mode, for a new archive.
    let w = ClaudeHome::new();
    refused_as_not_built(launch(loaded(&w, "mode = 'primary'\n")));
    assert!(!w.root.join("meta.json").exists());
    assert!(!w.root.join("archive.db").exists());
}
