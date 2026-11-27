use super::*;
use polylogue_store::Archive;

/// The final answer of the next reply on a connection: its frames, of which
/// the `answer` frame is last.
fn next_answer(input: &mut impl BufRead) -> command_surface::Answer<serde_json::Value> {
    let mut event = String::new();
    let mut line = String::new();
    loop {
        line.clear();
        assert!(
            input.read_line(&mut line).unwrap() > 0,
            "the reply ended before its answer"
        );
        let line = line.trim_end_matches('\n');
        if let Some(name) = line.strip_prefix("event: ") {
            event = name.to_owned();
        } else if let Some(data) = line.strip_prefix("data: ")
            && event == "answer"
        {
            let answer = serde_json::from_str(data).unwrap();
            line_end(input);
            return answer;
        }
    }
}

/// Takes the blank line that ends a frame.
fn line_end(input: &mut impl BufRead) {
    let mut blank = String::new();
    input.read_line(&mut blank).unwrap();
    assert_eq!(blank, "\n");
}

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
    let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
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
    let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    let request = command_surface::Request {
        operation: "set".into(),
        arguments: serde_json::json!({"key":"browser.actions","value":"on"}),
        request_id: "saved-socket-setting".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    let answer = next_answer(&mut BufReader::new(stream));
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
    let mut old = Server::start(Arc::clone(&first.daemon)).unwrap();
    old.shutdown().unwrap();
    let _server = Server::start(Arc::clone(&first.daemon)).unwrap();
    drop(old);
    UnixStream::connect(first.daemon.archive.socket()).unwrap();
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
    let mut server = Server::start(Arc::clone(&daemon)).unwrap();
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
    // A reader held across the stop ends with it.
    assert!(matches!(
        reader.query("SELECT 42 AS n", &[]),
        Err(polylogue_store::StoreError::Stopped)
    ));
}

#[test]
fn listener_stop_is_visible_before_clients_join() {
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
    let mut server = Server::start(Arc::clone(&daemon)).unwrap();
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
    let listener_stopped = server.take_listener_stopped().unwrap();
    runtime.stop_accepting();
    server.request_stop();
    listener_stopped.blocking_recv().unwrap();
    server.shutdown().unwrap();
    assert!(matches!(
        daemon.reads.reader(),
        Err(polylogue_store::StoreError::Stopped)
    ));
    // A reader held across the stop ends with it.
    assert!(matches!(
        reader.query("SELECT 42 AS n", &[]),
        Err(polylogue_store::StoreError::Stopped)
    ));
}

#[test]
fn shutdown_cancels_a_started_socket_read_before_joining_its_transport() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(dir.path().to_owned());
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        service::ArchiveMode::Standalone,
    )
    .unwrap();
    let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    let request = command_surface::Request {
        operation: "dev sql".into(),
        arguments: serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
        request_id: "read-started-before-stop".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    eventually("the read took its reader", || {
        runtime.daemon.reads.checked_out() == 1
    });
    runtime.stop_accepting();
    // The peer stays connected, and the transport has not been asked to stop:
    // the stop alone ends the read, which answers that the daemon stops.
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let answer = next_answer(&mut BufReader::new(stream));
    assert_eq!(
        answer.failure().and_then(|failure| failure.code.as_deref()),
        Some("shutting_down"),
        "{answer:?}"
    );
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    assert_eq!(runtime.daemon.reads.checked_out(), 0);
}

#[test]
fn shutdown_cancels_reads_without_revoking_a_handed_over_write() {
    use polylogue_store::writer::DispatchPhase;
    for phase in [DispatchPhase::BeforeParentCheck, DispatchPhase::Owned] {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::test_daemon::engine_config(dir.path().to_owned());
        config.tool_dirs.clear();
        config.workers = 2;
        let root = dir.path().join("archive");
        let mut runtime =
            crate::test_daemon::start_runtime(&root, config, service::ArchiveMode::Standalone)
                .unwrap();
        let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
        let mut reading = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
        let request = command_surface::Request {
            operation: "dev sql".into(),
            arguments: serde_json::json!({"statement":"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
            request_id: "read-beside-write".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        writeln!(reading, "{}", serde_json::to_string(&request).unwrap()).unwrap();
        eventually("the read took its reader", || {
            runtime.daemon.reads.checked_out() == 1
        });
        let pause = runtime
            .daemon
            .writes
            .as_ref()
            .unwrap()
            .pause_user_dispatch_for_test("handed-over-setting", phase);
        let mut writing = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
        writing
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let request = command_surface::Request {
            operation: "set".into(),
            arguments: serde_json::json!({"key":"browser.actions","value":"on"}),
            request_id: "handed-over-setting".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let outgoing = crate::protocol::wire::Outgoing {
            request: &request,
            caller: Default::default(),
            parent_deadline_ms: Some(
                polylogue_model::WallClock::new(jiff::Timestamp::now).now_ms() + 60_000,
            ),
        };
        writeln!(writing, "{}", serde_json::to_string(&outgoing).unwrap()).unwrap();
        pause.wait().unwrap();
        runtime.stop_accepting();
        reading
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let read = next_answer(&mut BufReader::new(reading));
        assert_eq!(
            read.failure().and_then(|failure| failure.code.as_deref()),
            Some("shutting_down"),
            "{read:?}"
        );
        pause.release();
        let answer = next_answer(&mut BufReader::new(writing));
        assert!(
            answer.value().is_some(),
            "handed-over write lost its outcome: {answer:?}"
        );
        server.shutdown().unwrap();
        runtime.shutdown().unwrap();
        let pool = polylogue_store::ReadPool::open(&Archive::at(&root)).unwrap();
        let reader = pool.reader().unwrap();
        assert_eq!(
            reader.setting_value("browser.actions", "").unwrap(),
            Some(polylogue_model::settings::Value::Boolean(true))
        );
        assert!(
            reader
                .user_request("handed-over-setting")
                .unwrap()
                .is_some()
        );
    }
}

#[test]
fn socket_honours_the_client_deadline_without_closing_the_connection() {
    let daemon = crate::test_daemon::start("socket deadline");
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let request = command_surface::Request {
        operation: "dev sql".into(),
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
    let response = next_answer(&mut input);
    assert!(
        matches!(&response,command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.kind==command_surface::FailureKind::Timeout),
        "{response:?}"
    );
    let next = command_surface::Request {
        operation: "dev sql".into(),
        arguments: serde_json::json!({"statement":"SELECT 1"}),
        request_id: "after-timeout".into(),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    writeln!(input.get_mut(), "{}", serde_json::to_string(&next).unwrap()).unwrap();
    let next = next_answer(&mut input);
    assert!(next.value().is_some(), "{next:?}");
}

#[test]
fn a_client_can_close_its_write_half_and_still_receive_the_answer() {
    let daemon = crate::test_daemon::start("socket half close");
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.write_all(b"{\"caller\":{\"controlling_terminal\":false},\"operation\":\"dev sql\",\"request_id\":\"half-close\",\"arguments\":{\"statement\":\"WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100000) SELECT count(*) AS n FROM n\"}}\n").unwrap();
    stream.shutdown(std::net::Shutdown::Write).unwrap();
    crate::test_answers::ok(next_answer(&mut BufReader::new(stream)));
}

#[test]
fn a_request_line_that_is_not_utf8_is_refused_and_the_connection_continues() {
    let daemon = crate::test_daemon::start("socket invalid utf8");
    let mut stream = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    stream.write_all(b"{\"operation\":\"\xff\"}\n").unwrap();
    let mut input = BufReader::new(stream);
    let refused = next_answer(&mut input);
    assert!(
        matches!(&refused, command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.code.as_deref() == Some("bad_request")),
        "{refused:?}"
    );
    let next = request_line(
        "dev sql",
        serde_json::json!({"statement":"SELECT 1"}),
        "after-bad-line",
    );
    input.get_mut().write_all(next.as_bytes()).unwrap();
    let next = next_answer(&mut input);
    assert!(next.value().is_some(), "{next:?}");
}

#[test]
fn a_request_sent_during_the_stop_answers_shutting_down_before_the_socket_drains() {
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
    let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut responses = BufReader::new(stream.try_clone().unwrap());
    let request = |id: &str| command_surface::Request {
        operation: "dev sql".into(),
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
        serde_json::to_string(&request("accepted-peer")).unwrap()
    )
    .unwrap();
    assert!(next_answer(&mut responses).value().is_some());
    runtime.stop_accepting();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&request("queued-at-stop")).unwrap()
    )
    .unwrap();
    server.shutdown().unwrap();
    let answer = next_answer(&mut responses);
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
                "rows",
                serde_json::json!({"selection":"session:claude-code:large","steps":[{"verb":"read","delivery":"stream"}]}),
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
    next_answer(&mut BufReader::new(stream))
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
            if name == "complete" || name == "answer" {
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
            "set",
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
                            "dev sql",
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
    let mut long_read = UnixStream::connect(d.archive.socket()).unwrap();
    long_read
        .write_all(
            request_line(
                "dev sql",
                serde_json::json!({"statement": "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"}),
                "long-read",
            )
            .as_bytes(),
        )
        .unwrap();
    eventually("the long read took the other reader", || {
        d.reads.checked_out() == 2
    });
    // An import runs at idle priority meanwhile.
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
    assert_eq!(
        d.reads.checked_out(),
        2,
        "the long read was still running when status answered"
    );
}

#[test]
fn the_listener_answers_connections_and_stops_on_request() {
    let mut daemon = crate::test_daemon::start("accept connections");
    let mut client = UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    for id in ["ping", "ping-again"] {
        assert_eq!(
            crate::test_answers::ok(answer_of(
                &mut client,
                &request_line(PING, serde_json::json!({}), id),
            )),
            serde_json::json!({})
        );
    }
    drop(client);
    daemon.stop_server().unwrap();
    assert_eq!(
        std::fs::metadata(daemon.daemon.archive.socket())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::NotFound
    );
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn idle_listener_makes_no_wakeups() {
    let directory = tempfile::tempdir().unwrap();
    let listener = UnixListener::bind(directory.path().join("daemon.sock")).unwrap();
    listener.set_nonblocking(true).unwrap();
    let (mut stop, stopped) = UnixStream::pair().unwrap();
    let (started, thread_id) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let current = std::fs::read_link("/proc/thread-self").unwrap();
        started
            .send(current.file_name().unwrap().to_owned())
            .unwrap();
        Accept { listener, stopped }.next().unwrap().map(|_| ())
    });
    let syscall = std::path::Path::new("/proc/self/task")
        .join(thread_id.recv().unwrap())
        .join("syscall");
    // On x86_64 Linux rustix uses ppoll: its third argument is a timespec
    // pointer, null for no timeout. Observing the blocked call distinguishes
    // an indefinite wait from any timer without measuring elapsed time.
    let timeout = loop {
        let state = std::fs::read_to_string(&syscall).unwrap();
        let fields: Vec<_> = state.split_whitespace().collect();
        if fields[0] == libc::SYS_ppoll.to_string() {
            assert_eq!(fields[2], "0x2", "listener and stop descriptors: {state}");
            break u64::from_str_radix(fields[3].trim_start_matches("0x"), 16).unwrap();
        }
        // The test runner bounds a thread that never reaches its wait.
        std::thread::yield_now();
    };
    stop.write_all(&[1]).unwrap();
    assert_eq!(thread.join().unwrap(), None);
    assert_eq!(timeout, 0, "the idle listener must wait without a timeout");
}

/// The JSON of each `record` event a follow sends, as they arrive, up to
/// the reply's final frame: the connection stays open for the next request.
fn follow_records(stream: UnixStream) -> impl Iterator<Item = serde_json::Value> {
    let mut event = String::new();
    BufReader::new(stream)
        .lines()
        .map_while(Result::ok)
        .take_while(|line| !matches!(line.strip_prefix("event: "), Some("answer" | "complete")))
        .filter_map(move |line| {
            if let Some(name) = line.strip_prefix("event: ") {
                event = name.to_owned();
                return None;
            }
            if event != "record" {
                return None;
            }
            line.strip_prefix("data: ")
                .and_then(|data| serde_json::from_str(data).ok())
        })
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
                "rows",
                serde_json::json!({"selection":"sessions","steps":[{"verb":"read","follow":true}]}),
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
fn a_follow_whose_client_hangs_up_ends_at_once_and_waits_without_a_reader() {
    let mut daemon = crate::test_daemon::start("follow hang-up");
    // Nothing commits while the follow waits: only its client can end it.
    daemon.pause_signals();
    let d = &daemon.daemon;
    let mut stream = UnixStream::connect(d.archive.socket()).unwrap();
    stream
        .write_all(
            request_line(
                "rows",
                serde_json::json!({"selection":"session:claude-code:s1","steps":[{"verb":"read","follow":true}]}),
                "follow-without-deadline",
            )
            .as_bytes(),
        )
        .unwrap();
    let mut records = follow_records(stream.try_clone().unwrap());
    let first = records.next().expect("the follow answers with its marks");
    assert_eq!(first["kind"], "marks", "{first}");
    assert_eq!(d.requests.running(), 1, "the follow is running");
    // Background workers take readers now and then; the waiting follow
    // holds none.
    eventually("the waiting follow holds no reader", || {
        d.reads.checked_out() == 0
    });
    drop(records);
    drop(stream);
    eventually("the follow ended when its client hung up", || {
        d.requests.running() == 0
    });
}

fn via(words: &[&str]) -> command_surface::Via {
    via_of_command_line(
        &words
            .iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>(),
    )
}

#[test]
fn peer_running_internal_hook_is_the_hook_channel() {
    use command_surface::Via;
    for arguments in [
        vec!["polylogue", "internal", "hook", "SessionStart"],
        vec![
            "polylogue",
            "--archive",
            "/synthetic",
            "internal",
            "hook",
            "Stop",
        ],
        vec!["polylogue", "--deadline=100", "internal", "hook", "Stop"],
    ] {
        assert_eq!(via(&arguments), Via::Hook);
    }
    assert_eq!(
        via(&["polylogue", "internal", "browser-host"]),
        Via::Extension
    );
    assert_eq!(
        via(&["polylogue", "sessions", "internal", "hook"]),
        Via::Cli
    );
}

#[test]
fn a_query_word_named_mcp_does_not_change_the_observed_channel() {
    use command_surface::Via;
    assert_eq!(via(&["polylogue", "find", "mcp"]), Via::Cli);
    assert_eq!(via(&["polylogue", "--json", "mcp"]), Via::Mcp);
    assert_eq!(via(&["polylogue", "--deadline", "100", "mcp"]), Via::Mcp);
    assert_eq!(via(&["polylogue", "--budget", "4000", "mcp"]), Via::Mcp);
    assert_eq!(via(&["polylogue", "--cursor=abc", "mcp"]), Via::Mcp);
}

#[test]
fn shutdown_drains_the_socket_within_its_budget() {
    // One connection whose request never ends, as one that ignores its
    // cancellation would: the drain hangs it up and stops waiting at its
    // budget, reporting it as left open.
    let clients = Clients::default();
    let (served, mut peer) = UnixStream::pair().unwrap();
    clients.open.locked().push((0, served));
    assert_eq!(
        clients.drain(Duration::from_millis(100), Duration::from_millis(200)),
        1
    );
    // The hang-up reached the peer: it reads the end of the stream.
    let mut rest = Vec::new();
    peer.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
    assert_eq!(std::io::Read::read_to_end(&mut peer, &mut rest).unwrap(), 0);
    // A connection that closes during the response window ends the drain
    // early, with nothing left.
    let clients = Arc::new(Clients::default());
    let (served, _peer) = UnixStream::pair().unwrap();
    clients.open.locked().push((0, served));
    let closing = Arc::clone(&clients);
    let closer = std::thread::spawn(move || {
        closing.open.locked().clear();
        closing.closed.notify_all();
    });
    assert_eq!(
        clients.drain(Duration::from_secs(30), Duration::from_secs(30)),
        0
    );
    closer.join().unwrap();
}
