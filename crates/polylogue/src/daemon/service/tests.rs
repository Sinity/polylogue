use super::*;
use crate::test_daemon;

fn read(
    d: &Daemon,
    view: &str,
    format: &str,
    budget: Option<usize>,
    cursor: Option<&str>,
) -> Response {
    handle(
        Request::Read {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            view: view.into(),
            format: format.into(),
            logical: false,
            budget,
            cursor: cursor.map(Into::into),
        },
        d,
    )
}

#[test]
fn a_read_honours_its_view_format_and_budget() {
    let w = test_daemon::start("a prompt long enough to need two pages of one token each");
    let Response::Ok { data } = read(&w.daemon, "full", "markdown", None, None) else {
        panic!("markdown read failed")
    };
    assert_eq!(data["format"], "markdown");
    assert!(
        data["content"]
            .as_str()
            .unwrap()
            .contains("a prompt long enough")
    );

    let Response::Ok { data } = read(&w.daemon, "full", "json", None, None) else {
        panic!("json read failed")
    };
    assert_eq!(data["messages"].as_array().unwrap().len(), 1);
    // A view that keeps no block of the session leaves no message.
    let Response::Ok { data } = read(&w.daemon, "actions", "json", None, None) else {
        panic!("json read failed")
    };
    assert!(data["messages"].as_array().unwrap().is_empty());

    let Response::Ok { data: first } = read(&w.daemon, "full", "json", Some(5), None) else {
        panic!("paged read failed")
    };
    let next = first["next"].as_str().expect("a second page").to_owned();
    let Response::Ok { data: second } = read(&w.daemon, "full", "json", Some(5), Some(&next))
    else {
        panic!("paged read failed")
    };
    assert_ne!(first["text"], second["text"]);

    for (view, format, cursor) in [
        ("nope", "json", None),
        ("full", "pdf", None),
        ("full", "json", Some("m0.b0.c0")),
    ] {
        assert!(
            matches!(read(&w.daemon, view, format, None, cursor), Response::Error { ref kind, .. } if kind == "bad_request"),
            "{view} {format} {cursor:?}"
        );
    }
}

#[test]
fn excision_format_refusal_has_a_distinct_public_error() {
    let error = StoreError::ExcisionFormat {
        input: "input1".into(),
        format: "export-file".into(),
        detail: "ZIP members need decoding".into(),
    };
    assert!(
        matches!(store_error(&error), Response::Error { kind, detail: Some(detail), .. }
        if kind == "unsupported_excision" && detail["input"] == "input1" && detail["format"] == "export-file")
    );
    let error = StoreError::BlobCorrupt(polylogue_model::BlobHash::of(b"synthetic"));
    assert!(matches!(store_error(&error), Response::Error { kind, .. } if kind == "store"));
}

fn config(home: &std::path::Path) -> Config {
    let mut config = Config::from_env();
    config.home = home.to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    config
}

#[test]
fn runtime_shutdown_wakes_signals_releases_lock_and_refuses_old_handles() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut runtime = start(&root, config(dir.path()), Backing::Standalone).unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    assert!(daemon.archive.lock_held());
    let (done, stopped) = sync_channel(1);
    let thread = std::thread::spawn(move || {
        let result = runtime.shutdown();
        done.send(result).unwrap();
    });
    stopped
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("shutdown waited for signal timer")
        .unwrap();
    thread.join().unwrap();
    assert!(
        matches!(handle(Request::Status, &daemon), Response::Error { kind, .. } if kind == "daemon_stopping")
    );
    assert!(
        Archive::open_or_create(&root).is_ok(),
        "runtime retained its writer lock"
    );
}

#[test]
fn startup_failure_after_workers_start_releases_the_runtime_and_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let weak;
    {
        let runtime = start(&root, config(dir.path()), Backing::Standalone).unwrap();
        weak = Arc::downgrade(&runtime.daemon);
        // A directory cannot be replaced by the Unix socket binder.
        std::fs::create_dir(runtime.daemon.archive.socket()).unwrap();
        assert!(crate::daemon::server::Server::start(Arc::clone(&runtime.daemon)).is_err());
    }
    assert!(weak.upgrade().is_none());
    assert!(Archive::open_or_create(&root).is_ok());
}

#[test]
fn a_panicking_test_drops_its_in_process_daemon_workers() {
    let (sent, received) = sync_channel(1);
    assert!(
        std::panic::catch_unwind(|| {
            let world = test_daemon::start("panic fixture");
            sent.send(Arc::downgrade(&world.daemon)).unwrap();
            panic!("synthetic test panic");
        })
        .is_err()
    );
    assert!(received.recv().unwrap().upgrade().is_none());
}

#[test]
fn repeated_test_daemon_drops_release_every_shared_daemon_reference() {
    for _ in 0..3 {
        let world = test_daemon::start("owned fixture");
        let weak = Arc::downgrade(&world.daemon);
        drop(world);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn an_old_daemon_handle_does_not_prevent_empty_archive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut runtime = start(&root, config(dir.path()), Backing::Standalone).unwrap();
    let old = Arc::clone(&runtime.daemon);
    assert!(matches!(handle(Request::Status, &old), Response::Ok { .. }));
    runtime.shutdown().unwrap();
    assert!(matches!(
        old.reads.reader(),
        Err(StoreError::ReadPoolClosed)
    ));
    let mut next = start(&root, config(dir.path()), Backing::Standalone).unwrap();
    assert!(matches!(
        handle(Request::Status, &next.daemon),
        Response::Ok { .. }
    ));
    next.shutdown().unwrap();
    assert!(
        matches!(handle(Request::Status, &old), Response::Error { kind, .. } if kind == "daemon_stopping")
    );
}

#[cfg(feature = "sinex")]
#[test]
fn standalone_start_does_not_resolve_operator_backing() {
    const CHILD: &str = "POLYLOGUE_TEST_STANDALONE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        assert!(
            crate::sinex::configured().is_some(),
            "synthetic primary configuration was not loaded"
        );
        let dir = tempfile::tempdir().unwrap();
        let runtime = start(
            &dir.path().join("archive"),
            config(dir.path()),
            Backing::Standalone,
        )
        .unwrap();
        assert!(runtime.publisher.is_none());
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("polylogue");
    std::fs::create_dir_all(&config).unwrap();
    // A configured primary is deliberately present in the child process.
    // Its invalid endpoint cannot contact a real account if isolation fails.
    std::fs::write(
        config.join("sinex.toml"),
        format!(
            "mode = 'primary'\nnats = 'invalid://'\nstore = {:?}\n",
            dir.path().join("sinex-store")
        ),
    )
    .unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "daemon::service::tests::standalone_start_does_not_resolve_operator_backing",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("XDG_CONFIG_HOME", dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("1 passed"),
        "child did not execute its control: {stdout}"
    );
    assert!(!dir.path().join("sinex-store").exists());
}

#[test]
fn admission_stop_releases_full_pool_requests_before_workers_or_admitted_readers_finish() {
    let dir = tempfile::tempdir().unwrap();
    let mut runtime = start(
        &dir.path().join("archive"),
        config(dir.path()),
        Backing::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    let mut requests = Vec::new();
    let entered = Arc::new(std::sync::Barrier::new(9));
    for _ in 0..8 {
        let daemon = Arc::clone(&daemon);
        let entered = Arc::clone(&entered);
        requests.push(std::thread::spawn(move || {
            entered.wait();
            handle(Request::Status, &daemon)
        }));
    }
    entered.wait();
    runtime.stop_admission();
    for request in requests {
        assert!(
            matches!(request.join().unwrap(), Response::Error { kind, .. } if kind == "daemon_stopping")
        );
    }
    runtime.shutdown().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    assert!(
        matches!(handle(Request::Status, &daemon), Response::Error { kind, .. } if kind == "daemon_stopping")
    );
}

#[test]
fn invalid_reader_resource_configuration_refuses_startup_and_releases_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    for (workers, memory) in [(0, 128 << 20), (2, 3 * 3072 - 1)] {
        let mut config = config(dir.path());
        config.workers = workers;
        config.reader_memory_bytes = memory;
        assert!(start(&root, config, Backing::Standalone).is_err());
        assert!(Archive::open_or_create(&root).is_ok());
    }
}
