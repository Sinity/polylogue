use super::*;
use crate::daemon::Response;
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
        .recv()
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
    let request = Request::Setting {
        request_id: None,
        key: "lifecycle-test".into(),
        value: Some("written".into()),
        machine: false,
    };
    writeln!(stream, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response).unwrap();
    assert!(matches!(
        serde_json::from_str::<super::super::Response>(&response).unwrap(),
        super::super::Response::Ok { .. }
    ));
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    let reads = polylogue_store::ReadPool::open(&Archive::at(&root)).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .setting("lifecycle-test", "")
            .unwrap()
            .as_deref(),
        Some("written")
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
fn listener_wait_stops_accepting_requests_before_joining_clients() {
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
    let server = Server::start(Arc::clone(&daemon), Arc::clone(&runtime.operations)).unwrap();
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
    server.stop.as_ref().unwrap().send(()).unwrap();
    server.wait(|| runtime.stop_accepting()).unwrap();
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
    let request = Request::Sql { statement: "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n".into() };
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
    let request = Request::Sql { statement: "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n".into() };
    let outgoing = crate::daemon::execution::Outgoing::new(
        &request,
        Some(crate::daemon::execution::deadline(Duration::from_millis(
            50,
        ))),
    );
    serde_json::to_writer(&mut stream, &outgoing).unwrap();
    stream.write_all(b"\n").unwrap();
    let mut input = BufReader::new(stream);
    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    let response: Response = serde_json::from_str(&line).unwrap();
    assert!(matches!(response, Response::Error { kind, .. } if kind == "deadline_exceeded"));
    input
        .get_mut()
        .write_all(b"{\"caller\":{\"controlling_terminal\":false},\"op\":\"sql\",\"statement\":\"SELECT 1\"}\n")
        .unwrap();
    line.clear();
    input.read_line(&mut line).unwrap();
    assert!(matches!(
        serde_json::from_str::<Response>(&line).unwrap(),
        Response::Ok { .. }
    ));
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
    assert!(
        matches!(
            serde_json::from_str::<Response>(&line).unwrap(),
            Response::Ok { .. }
        ),
        "{line}"
    );
}
