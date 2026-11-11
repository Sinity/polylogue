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
        serde_json::to_string(&crate::protocol::wire::Outgoing {
            request: &crate::client::request("status", serde_json::json!({}), "status".into()),
            caller: Default::default(),
            parent_deadline_ms: None,
        })
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
        serde_json::to_string(&crate::protocol::wire::Outgoing {
            request: &crate::client::request("status", serde_json::json!({}), "status".into()),
            caller: Default::default(),
            parent_deadline_ms: None,
        })
        .unwrap()
    )
    .unwrap();
    let quiescing = server.take_quiescing().unwrap();
    runtime.stop_accepting();
    server.request_stop();
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
    let elapsed = daemon.assert_read_released(&limit);
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
    stream.write_all(b"{\"caller\":{\"controlling_terminal\":false},\"operation\":\"sql\",\"request_id\":\"half-close\",\"arguments\":{\"statement\":\"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT count(*) AS n FROM n\"}}\n").unwrap();
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
    send_answer(
        &mut output,
        &answer,
        &crate::protocol::ReadScopes::single(Some(scope)),
        None,
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

/// A daemon holding `claude-code:large`, a session of many messages far
/// larger than a socket buffer and the stream buffer's memory share
/// together, so a client that stops reading stalls the stream's sends.
fn daemon_with_a_large_session(workers: usize) -> crate::test_daemon::TestDaemon {
    let seed = |home: &std::path::Path| {
        let text = "the quick brown fox jumps over the lazy dog ".repeat(100);
        let lines: String = (0..2_000)
            .map(|n| polylogue_testkit::claude_user(&format!("large-{n}"), "large", &text))
            .collect();
        std::fs::write(home.join(".claude/projects/-p/large.jsonl"), lines).unwrap();
    };
    crate::test_daemon::start_with_sources_and_workers("small session", &seed, 2_000, workers)
}

fn request_line(operation: &str, arguments: serde_json::Value, id: &str) -> String {
    request_line_with(operation, arguments, id, command_surface::Deadline::None)
}

fn request_line_with(
    operation: &str,
    arguments: serde_json::Value,
    id: &str,
    deadline: command_surface::Deadline,
) -> String {
    let request = command_surface::Request {
        operation: operation.into(),
        arguments,
        request_id: id.into(),
        deadline,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    format!("{}\n", serde_json::to_string(&request).unwrap())
}

/// A client that asks for the whole session as a stream and then reads
/// nothing; returns once the daemon has begun sending.
fn stalled_stream(daemon: &Daemon) -> UnixStream {
    let mut stream = UnixStream::connect(daemon.archive.socket()).unwrap();
    stream
        .write_all(
            request_line(
                "read",
                serde_json::json!({"target": "claude-code:large", "delivery": "stream"}),
                "stalled-stream",
            )
            .as_bytes(),
        )
        .unwrap();
    // Blocks until the first bytes arrive, without taking them.
    let mut first = [0u8; 1];
    rustix::net::recv(&stream, &mut first, rustix::net::RecvFlags::PEEK).unwrap();
    stream
}

fn answer_of(stream: &mut UnixStream, line: &str) -> command_surface::Answer<serde_json::Value> {
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    stream.write_all(line.as_bytes()).unwrap();
    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response).unwrap();
    serde_json::from_str(&response).unwrap()
}

/// Reads the rest of a stream and returns its last event's name.
fn last_event(stream: UnixStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    let mut last = String::new();
    for line in BufReader::new(stream).lines() {
        let line = line.unwrap();
        if let Some(name) = line.strip_prefix("event: ") {
            last = name.to_owned();
            if name == "complete" || name == "error" {
                break;
            }
        }
    }
    last
}

fn eventually(what: &str, done: impl Fn() -> bool) {
    let until = std::time::Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(std::time::Instant::now() < until, "{what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// `(busy, frames left)` after a truncating checkpoint of `database`.
fn truncate_wal(database: &std::path::Path) -> (i64, i64) {
    let connection = rusqlite::Connection::open(database).unwrap();
    connection
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
}

#[test]
fn stalled_client_releases_reader_and_wal_stops_growing() {
    let daemon = daemon_with_a_large_session(1);
    let d = &daemon.daemon;
    let stalled = stalled_stream(d);
    eventually("the stalled stream's reader is released", || {
        d.reads.checked_out() == 0
    });
    // A write lands in both write-ahead logs while the client still has not
    // read; no reader pins an older snapshot, so both truncate to nothing.
    let mut writer = UnixStream::connect(d.archive.socket()).unwrap();
    crate::test_answers::ok(answer_of(
        &mut writer,
        &request_line(
            "setting",
            serde_json::json!({"key": "browser.actions", "value": "on"}),
            "write-during-stall",
        ),
    ));
    for database in [d.archive.index_db(), d.archive.archive_db()] {
        assert_eq!(truncate_wal(&database), (0, 0), "{}", database.display());
        assert_eq!(
            std::fs::metadata(database.with_extension("db-wal"))
                .map(|wal| wal.len())
                .unwrap_or(0),
            0
        );
    }
    // The bytes held for the stalled client all arrive once it reads.
    assert_eq!(last_event(stalled), "complete");
}

#[test]
fn sixteen_clients_with_one_stalled_stream() {
    let daemon = daemon_with_a_large_session(1);
    let d = &daemon.daemon;
    let stalled = stalled_stream(d);
    let answers: Vec<_> = std::thread::scope(|scope| {
        let clients: Vec<_> = (0..15)
            .map(|n| {
                scope.spawn(move || {
                    let mut stream = UnixStream::connect(d.archive.socket()).unwrap();
                    answer_of(
                        &mut stream,
                        &request_line(
                            "sql",
                            serde_json::json!({"statement": format!("SELECT {n} AS n")}),
                            &format!("client-{n}"),
                        ),
                    )
                })
            })
            .collect();
        clients.into_iter().map(|c| c.join().unwrap()).collect()
    });
    for (n, answer) in answers.into_iter().enumerate() {
        let value = crate::test_answers::ok(answer);
        assert_eq!(value[0]["n"], n, "{value}");
    }
    assert_eq!(last_event(stalled), "complete");
}

#[test]
fn status_keeps_budget_during_import() {
    let daemon = crate::test_daemon::start_with_workers("status during import", 1);
    let d = &daemon.daemon;
    // Every ordinary reader is taken: one here, one by a read that runs
    // until it is cancelled, which also holds the only ordinary place.
    let _held = d.reads.reader().unwrap();
    let reads = daemon.watch_reads();
    let mut long_read = UnixStream::connect(d.archive.socket()).unwrap();
    long_read
        .write_all(
            request_line(
                "sql",
                serde_json::json!({"statement": "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
                "long-read",
            )
            .as_bytes(),
        )
        .unwrap();
    let limit = reads.recv_timeout(Duration::from_secs(10)).unwrap();
    crate::test_daemon::TestDaemon::wait_for_sql(&limit);
    // An import runs in the background place meanwhile.
    let export = daemon.home.join("import/session.jsonl");
    std::fs::create_dir_all(export.parent().unwrap()).unwrap();
    std::fs::write(
        &export,
        (0..2_000)
            .map(|n| {
                polylogue_testkit::claude_user(&format!("u{n}"), "imported", "an imported line")
            })
            .collect::<String>(),
    )
    .unwrap();
    let mut import = UnixStream::connect(d.archive.socket()).unwrap();
    import
        .write_all(
            request_line(
                "import",
                serde_json::json!({"paths": [export]}),
                "background-import",
            )
            .as_bytes(),
        )
        .unwrap();
    let mut status = UnixStream::connect(d.archive.socket()).unwrap();
    // The status class's own deadline: a status queued behind the long read
    // would answer with a timeout.
    let line = request_line_with(
        "status",
        serde_json::json!({}),
        "status-during-import",
        command_surface::Deadline::Default,
    );
    let answer = answer_of(&mut status, &line);
    crate::test_answers::ok(answer);
    assert!(
        !limit.cancellation.is_cancelled() && limit.check().is_ok(),
        "the long read was still running when status answered"
    );
}

#[test]
fn accept_wakes_on_connection_and_on_stop_only() {
    let mut daemon = crate::test_daemon::start("accept wakeups");
    let before = daemon.server_wakeups();
    let mut client = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    crate::test_answers::ok(answer_of(
        &mut client,
        &request_line(PING, serde_json::json!({}), "ping"),
    ));
    // Requests on an open connection never wake the listener.
    crate::test_answers::ok(answer_of(
        &mut client,
        &request_line(PING, serde_json::json!({}), "ping-again"),
    ));
    assert_eq!(daemon.server_wakeups(), before + 1);
    drop(client);
    daemon.stop_server().unwrap();
    assert_eq!(daemon.server_wakeups(), before + 2);
}

#[test]
fn idle_listener_makes_no_wakeups() {
    let daemon = crate::test_daemon::start("idle listener");
    let before = daemon.server_wakeups();
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(daemon.server_wakeups(), before);
}

/// The JSON of each `record` event a follow sends, as they arrive.
fn follow_records(stream: UnixStream) -> impl Iterator<Item = serde_json::Value> {
    BufReader::new(stream)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| line.strip_prefix("data: ").map(str::to_owned))
        .filter_map(|data| serde_json::from_str(&data).ok())
}

#[test]
fn a_follow_shows_what_a_session_appends_and_ends_with_its_marks() {
    let daemon = crate::test_daemon::start("followed session");
    let d = &daemon.daemon;
    let mut stream = UnixStream::connect(d.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .unwrap();
    stream
        .write_all(
            request_line_with(
                "follow",
                serde_json::json!({"target": "sessions"}),
                "follow-appends",
                command_surface::Deadline::Milliseconds(8_000),
            )
            .as_bytes(),
        )
        .unwrap();
    let mut records = follow_records(stream);
    let first = records.next().expect("the first look shows the session");
    assert_eq!(first["session"], "claude-code:s1", "{first}");
    std::fs::OpenOptions::new()
        .append(true)
        .open(daemon.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap()
        .write_all(polylogue_testkit::claude_user("u2", "s1", "appended later").as_bytes())
        .unwrap();
    let mut appended = false;
    let mut last = serde_json::Value::Null;
    for record in records {
        if record["kind"] == "message" && record["message"] == "claude-code:u2" {
            appended = true;
        }
        last = record;
    }
    assert!(appended, "the appended message was followed");
    assert_eq!(last["kind"], "marks", "{last}");
    assert!(
        last["marks"]["sessions"]["claude-code:s1"].is_object(),
        "{last}"
    );
}

#[test]
fn a_follow_whose_client_hangs_up_gives_up_its_place_at_once() {
    let mut daemon = crate::test_daemon::start("follow hang-up");
    // Nothing commits while the follow waits: only its client can end it.
    daemon.pause_signals();
    let d = &daemon.daemon;
    let mut stream = UnixStream::connect(d.archive.socket()).unwrap();
    stream
        .write_all(
            request_line(
                "follow",
                serde_json::json!({"target": "claude-code:s1"}),
                "follow-without-deadline",
            )
            .as_bytes(),
        )
        .unwrap();
    let mut first = [0u8; 1];
    rustix::net::recv(&stream, &mut first, rustix::net::RecvFlags::PEEK).unwrap();
    drop(stream);
    eventually("the follow ended when its client hung up", || {
        d.executor.running() == 0
    });
}
