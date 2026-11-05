use super::*;
use polylogue_store::Archive;

#[test]
fn socket_shutdown_unblocks_idle_and_incomplete_clients() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let mut server =
        Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations)).unwrap();
    let _idle = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    let mut partial = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    partial.write_all(b"{\"op\":").unwrap();
    let (done, stopped) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::spawn(move || {
        done.send(server.shutdown()).unwrap();
    });
    stopped
        .recv_timeout(Duration::from_secs(2))
        .expect("socket readers did not stop")
        .unwrap();
    thread.join().unwrap();
    assert!(!runtime.daemon.archive.socket().exists());
    drop(runtime);
}

#[test]
fn acknowledged_socket_write_survives_runtime_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let mut runtime =
        crate::test_daemon::start_runtime(&root, config, service::ArchiveMode::Standalone).unwrap();
    let mut server =
        Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    let request = command_surface::Request {
        operation: "setting".into(),
        arguments: serde_json::json!({"key":"browser.actions","value":"on"}),
        request_id: "durable-socket-setting".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response).unwrap();
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_str(&response).unwrap();
    assert!(answer.value().is_some(), "{answer:?}");
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    let reads = polylogue_store::ReadPool::open(&Archive::at(&root)).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .setting_value("browser.actions", "")
            .unwrap(),
        Some(polylogue_model::settings::Value::Boolean(true))
    );
}

#[test]
fn dropping_a_stopped_socket_owner_does_not_remove_the_next_socket() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let first =
        crate::test_daemon::start_runtime(&root, config, service::ArchiveMode::Standalone).unwrap();
    let mut old = Server::start(Arc::clone(&first.daemon), Arc::clone(&first.operations)).unwrap();
    old.shutdown().unwrap();
    let _server = Server::start(Arc::clone(&first.daemon), Arc::clone(&first.operations)).unwrap();
    drop(old);
    assert!(UnixStream::connect(first.daemon.archive.socket()).is_ok());
}

#[test]
fn application_shutdown_joins_socket_clients_while_all_reader_slots_remain_held() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    let mut server = Server::start(Arc::clone(&daemon), Arc::clone(&runtime.operations)).unwrap();
    let mut stream = UnixStream::connect(daemon.archive.socket()).unwrap();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&crate::daemon::execution::Outgoing::new(
            &Request::Status,
            None
        ))
        .unwrap()
    )
    .unwrap();
    runtime.stop_accepting();
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}

#[test]
fn listener_quiescence_is_observable_before_joining_clients() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    let mut server = Server::start(Arc::clone(&daemon), Arc::clone(&runtime.operations)).unwrap();
    let mut stream = UnixStream::connect(daemon.archive.socket()).unwrap();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&crate::daemon::execution::Outgoing::new(
            &Request::Status,
            None
        ))
        .unwrap()
    )
    .unwrap();
    let quiescing = server.take_quiescing().unwrap();
    runtime.stop_accepting();
    server.stop.as_ref().unwrap().send(()).unwrap();
    quiescing.blocking_recv().unwrap();
    server.shutdown().unwrap();
    assert!(matches!(
        daemon.reads.reader(),
        Err(polylogue_store::StoreError::ReadPoolClosed)
    ));
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}

#[test]
fn disconnect_interrupts_sql_and_releases_the_reader_slot() {
    let daemon = crate::test_daemon::start("socket cancellation");
    let _held = daemon.daemon.reads.reader().unwrap();
    let reads = daemon.watch_reads();
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    let request = command_surface::Request {
        operation: "sql".into(),
        arguments: serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
        request_id: "disconnect-sql".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    serde_json::to_writer(&mut stream, &request).unwrap();
    stream.write_all(b"\n").unwrap();
    let limit = reads.recv_timeout(Duration::from_secs(5)).unwrap();
    crate::test_daemon::TestDaemon::wait_for_sql(&limit);
    drop(stream);
    let elapsed = daemon.assert_slot_freed();
    assert!(limit.cancellation.is_cancelled());
    eprintln!("socket disconnect released reader in {elapsed:?}");
}

#[test]
fn socket_honours_the_client_deadline_without_closing_the_connection() {
    let daemon = crate::test_daemon::start("socket deadline");
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let request = command_surface::Request {
        operation: "sql".into(),
        arguments: serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
        request_id: "bounded-sql".into(),
        deadline: command_surface::Deadline::Milliseconds(50),
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    serde_json::to_writer(&mut stream, &request).unwrap();
    stream.write_all(b"\n").unwrap();
    let mut input = BufReader::new(stream);
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    let response: command_surface::Answer<serde_json::Value> = serde_json::from_str(&line).unwrap();
    assert!(
        matches!(&response,command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.kind==command_surface::FailureKind::Timeout),
        "{response:?}"
    );
    let next = command_surface::Request {
        operation: "sql".into(),
        arguments: serde_json::json!({"statement":"SELECT 1"}),
        request_id: "after-timeout".into(),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(input.get_mut(), "{}", serde_json::to_string(&next).unwrap()).unwrap();
    line.clear();
    input.read_line(&mut line).unwrap();
    let next: command_surface::Answer<serde_json::Value> = serde_json::from_str(&line).unwrap();
    assert!(next.value().is_some(), "{next:?}");
}

#[test]
fn a_client_can_close_its_write_half_and_still_receive_the_answer() {
    let daemon = crate::test_daemon::start("socket half close");
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.write_all(b"{\"caller\":{\"controlling_terminal\":false},\"op\":\"sql\",\"statement\":\"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT count(*) AS n FROM n\"}\n").unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).unwrap();
    crate::test_answers::ok(
        serde_json::from_str::<command_surface::Answer<serde_json::Value>>(&line).unwrap(),
    );
}

#[test]
fn complete_queued_requests_receive_shutting_down_before_socket_drain() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(dir.path().join("home"));
    config.tool_dirs.clear();
    config.workers = 1;
    let runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let mut server =
        Server::start(Arc::clone(&runtime.daemon), Arc::clone(&runtime.operations)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut responses = BufReader::new(stream.try_clone().unwrap());
    let request = |id: &str| command_surface::Request {
        operation: "sql".into(),
        arguments: serde_json::json!({"statement":"SELECT 1 AS n"}),
        request_id: id.into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&request("admitted-peer")).unwrap()
    )
    .unwrap();
    let mut line = String::new();
    responses.read_line(&mut line).unwrap();
    assert!(
        serde_json::from_str::<command_surface::Answer<serde_json::Value>>(&line)
            .unwrap()
            .value()
            .is_some()
    );
    runtime.stop_accepting();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&request("queued-at-stop")).unwrap()
    )
    .unwrap();
    server.shutdown().unwrap();
    line.clear();
    responses.read_line(&mut line).unwrap();
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_str(&line).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("queued request must be refused");
    };
    assert_eq!(error.code.as_deref(), Some("shutting_down"));
    assert_eq!(error.kind, command_surface::FailureKind::NotRunning);
}

#[test]
fn a_scope_reset_during_json_serialization_publishes_only_the_safe_failure() {
    struct ResetWhileSerializing<'a>(&'a Archive);
    impl serde::Serialize for ResetWhileSerializing<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.0
                .advance_generation_for_test()
                .map_err(<S::Error as serde::ser::Error>::custom)?;
            serde::Serialize::serialize(
                &serde_json::json!({"stale":"synthetic-old-scope-value"}),
                serializer,
            )
        }
    }
    let daemon = crate::test_daemon::start("native JSON publication fence");
    let scope = daemon.daemon.reads.scope_token().unwrap();
    let (mut output, input) = UnixStream::pair().unwrap();
    input
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let answer = command_surface::Answer::ok(ResetWhileSerializing(&daemon.daemon.archive));
    write_scoped_json(
        &mut output,
        &answer,
        &crate::protocol::ReadScopes::single(Some(scope)),
    )
    .unwrap();
    let mut line = String::new();
    BufReader::new(input).read_line(&mut line).unwrap();
    assert!(!line.contains("synthetic-old-scope-value"));
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_str(&line).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("old scope must not publish a value");
    };
    assert_eq!(error.code.as_deref(), Some("scope_changed"));
}

/// law: until-idle-ends-after-work
/// law: no-sources-reads-no-source
#[test]
fn a_daemon_run_until_idle_exits_with_its_summary_and_reads_sources_only_when_asked() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let project = home.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    std::fs::write(
        project.join("s1.jsonl"),
        polylogue_testkit::claude_user("u1", "s1", "words on disk"),
    )
    .unwrap();
    let mut engine = crate::test_daemon::engine_config(home.clone());
    engine.tool_dirs.clear();
    engine.workers = 1;
    let root = dir.path().join("archive");
    let startup = crate::config::Startup::for_test(engine, &root);
    let inputs = || {
        polylogue_store::ReadPool::open(&Archive::at(&root))
            .unwrap()
            .reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM a.input", &[])
            .unwrap()[0]["n"]
            .as_i64()
            .unwrap()
    };
    let sourceless = serve(
        &startup,
        Run {
            reads_sources: false,
            lifetime: polylogue_engine::Lifetime::UntilIdle,
        },
    )
    .unwrap()
    .expect("a run until idle reports what it did");
    assert!(sourceless.work.errors.is_empty(), "{sourceless:?}");
    assert_eq!(sourceless.work.inputs_seen, 0);
    assert_eq!(inputs(), 0);
    assert!(
        !root.join("daemon.sock").exists(),
        "the socket outlived the run"
    );
    // The archive lock was released: a second run opens the same archive.
    let read = serve(
        &startup,
        Run {
            reads_sources: true,
            lifetime: polylogue_engine::Lifetime::UntilIdle,
        },
    )
    .unwrap()
    .expect("a run until idle reports what it did");
    assert!(read.work.errors.is_empty(), "{read:?}");
    assert!(read.work.inputs_changed >= 1, "{read:?}");
    assert_eq!(inputs(), 1);
}
