use super::*;

#[tokio::test]
async fn a_signal_checkpoint_failure_survives_worker_join_and_releases_the_archive() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let mut engine = crate::test_daemon::engine_config(directory.path().join("home"));
    engine.reads_sources = false;
    let (archive, claim) = polylogue_store::Archive::open_or_create(&root).unwrap();
    let conn = rusqlite::Connection::open(archive.archive_db()).unwrap();
    // Corrupt content is a real worker failure, not an instruction to bootstrap
    // a new checkpoint or to silently continue with empty state.
    conn.execute(
        "INSERT INTO signal_state(owner,body,commit_seq) VALUES(?1,?2,0)",
        rusqlite::params![engine.machine, [0_u8].as_slice()],
    )
    .unwrap();
    drop(conn);
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let expected = reads
        .reader()
        .unwrap()
        .signal_checkpoint(&engine.machine)
        .unwrap_err()
        .to_string();
    drop(reads);
    drop(claim);
    drop(archive);

    let mut runtime =
        crate::test_daemon::start_runtime(&root, engine, ArchiveMode::Standalone).unwrap();
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
