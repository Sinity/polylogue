use super::*;

#[tokio::test]
async fn a_signal_checkpoint_failure_survives_worker_join_and_releases_the_archive() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let mut engine = crate::test_daemon::engine_config(directory.path().join("home"));
    engine.reads_sources = false;
    let mut runtime =
        crate::test_daemon::start_runtime(&root, engine, ArchiveMode::Standalone).unwrap();
    runtime.pause_signals().unwrap();
    let conn = rusqlite::Connection::open(runtime.daemon.archive.archive_db()).unwrap();
    // Change only the synthetic checkpoint after startup, so setup
    // cannot fail through an unrelated initialization read of authored state.
    conn.execute(
        "INSERT OR REPLACE INTO signal_state(owner,body,commit_seq) VALUES(?1,?2,0)",
        rusqlite::params![runtime.daemon.config.requests.machine(), [0_u8].as_slice()],
    )
    .unwrap();
    drop(conn);
    let expected = runtime
        .daemon
        .reads
        .reader()
        .unwrap()
        .signal_checkpoint(runtime.daemon.config.requests.machine())
        .unwrap_err()
        .to_string();
    let _ = runtime.worker_events.send_replace(None);
    runtime.start_signals().unwrap();
    let mut workers = runtime.worker_events();
    let exited = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        workers.wait_for(|event| *event == Some("signal worker terminated")),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(*exited, Some("signal worker terminated"));
    drop(exited);
    let error = runtime.shutdown().unwrap_err();
    let held: serde_json::Value = serde_json::from_str(&error.to_string()).unwrap();
    assert_eq!(held["reason"], expected);
    assert_eq!(
        runtime.daemon.signals.lock().unwrap()["held"][0]["reason"],
        expected
    );
    // Returning the worker failure must not skip engine/writer teardown or the
    // exclusive claim. Failed startup leaves the original checkpoint untouched.
    let (reopened, _claim) = polylogue_store::Archive::open_or_create(&root).unwrap();
    let conn = rusqlite::Connection::open(reopened.archive_db()).unwrap();
    let body: Vec<u8> = conn
        .query_row("SELECT body FROM signal_state", [], |row| row.get(0))
        .unwrap();
    assert_eq!(body, [0]);
}
