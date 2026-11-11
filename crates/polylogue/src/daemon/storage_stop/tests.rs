use super::*;

fn failure(root: &Path) -> StorageFailure {
    StorageFailure {
        path: root.join("archive.db-wal"),
        message: "Input/output error (os error 5)".into(),
        errno: Some(5),
    }
}

#[test]
fn a_remembered_failure_refuses_this_boot_and_names_the_file_and_errno() {
    let root = tempfile::tempdir().unwrap();
    assert!(this_boot(root.path()).is_none());
    remember(root.path(), &failure(root.path()));
    let stopped = this_boot(root.path()).expect("the failure is kept for this boot");
    assert!(stopped.0.contains("archive.db-wal"), "{}", stopped.0);
    assert!(stopped.0.contains("errno 5"), "{}", stopped.0);
    std::fs::remove_file(marker(root.path())).unwrap();
}

#[test]
fn a_failure_from_another_boot_does_not_refuse() {
    let root = tempfile::tempdir().unwrap();
    let path = marker(root.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, format!("another-boot\n{}", failure(root.path()))).unwrap();
    assert!(this_boot(root.path()).is_none());
    std::fs::remove_file(path).unwrap();
}

/// The writer's failed sync reaches the daemon as a worker failure, which
/// stops it; the failure names the register's log.
#[test]
fn a_failed_register_sync_stops_the_daemon_and_names_the_log() {
    use polylogue_model::user::UserWrite;
    let daemon = crate::test_daemon::start("synthetic");
    let mut events = daemon.worker_events();
    polylogue_store::fault::arm(
        "write.register_sync",
        "",
        polylogue_store::fault::Action::Fail,
        0,
        1,
    );
    let context = daemon
        .daemon
        .operation_context(&crate::operation::Context::worker_authorship(
            "storage-test",
        ))
        .unwrap();
    let write = UserWrite::Assert {
        assertions: vec![polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Signal,
            key: Some("storage-test".into()),
            value: Some(serde_json::json!({"name": "storage-test", "query": "sessions"})),
            valid_until: None,
            links: vec![],
        }],
    };
    let writes = daemon.daemon.writes.as_ref().unwrap();
    // The write commits, its sync fails, and the writer stops before answering.
    let answer = writes.user(context.request("storage-test".into(), write));
    assert!(
        matches!(answer, Err(polylogue_store::StoreError::WriterGone)),
        "{answer:?}"
    );
    let stopped = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap()
        .block_on(async {
            loop {
                if let Some(reason) = *events.borrow_and_update() {
                    return reason;
                }
                events.changed().await.unwrap();
            }
        });
    assert_eq!(stopped, "the register failed a disk sync");
    let failure = polylogue_store::storage_failure::recorded().unwrap();
    assert!(failure.path.ends_with("archive.db-wal"), "{failure}");
    polylogue_store::fault::clear();
}
