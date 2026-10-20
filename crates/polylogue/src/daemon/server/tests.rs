use super::*;

#[test]
fn socket_shutdown_unblocks_idle_and_incomplete_clients() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let runtime = service::start(
        &dir.path().join("archive"),
        config,
        service::Backing::Standalone,
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
        .recv_timeout(Duration::from_secs(5))
        .expect("socket readers did not stop")
        .unwrap();
    thread.join().unwrap();
    assert!(!runtime.daemon.archive.socket().exists());
    drop(runtime);
}

#[test]
fn acknowledged_socket_write_survives_runtime_shutdown() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let mut runtime = service::start(&root, config, service::Backing::Standalone).unwrap();
    let mut server = Server::start(Arc::clone(&runtime.daemon)).unwrap();
    let mut stream = UnixStream::connect(runtime.daemon.archive.socket()).unwrap();
    let request = Request::Setting {
        request_id: None,
        key: "lifecycle-test".into(),
        value: Some("retained".into()),
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
        Some("retained")
    );
}

#[test]
fn dropping_a_stopped_socket_owner_does_not_remove_the_next_socket() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let root = dir.path().join("archive");
    let first = service::start(&root, config, service::Backing::Standalone).unwrap();
    let mut old = Server::start(Arc::clone(&first.daemon)).unwrap();
    old.shutdown().unwrap();
    let _server = Server::start(Arc::clone(&first.daemon)).unwrap();
    drop(old);
    assert!(UnixStream::connect(first.daemon.archive.socket()).is_ok());
}

#[test]
fn application_shutdown_joins_socket_clients_while_all_reader_slots_remain_held() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = service::start(
        &dir.path().join("archive"),
        config,
        service::Backing::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    let mut server = Server::start(Arc::clone(&daemon)).unwrap();
    let mut stream = UnixStream::connect(daemon.archive.socket()).unwrap();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&Request::Status).unwrap()
    )
    .unwrap();
    runtime.stop_admission();
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}

#[test]
fn listener_wait_stops_application_admission_before_joining_clients() {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let runtime = service::start(
        &dir.path().join("archive"),
        config,
        service::Backing::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    let server = Server::start(Arc::clone(&daemon)).unwrap();
    let mut stream = UnixStream::connect(daemon.archive.socket()).unwrap();
    writeln!(
        stream,
        "{}",
        serde_json::to_string(&Request::Status).unwrap()
    )
    .unwrap();
    server.stop.as_ref().unwrap().send(()).unwrap();
    server.wait(|| runtime.stop_admission()).unwrap();
    assert!(matches!(
        daemon.reads.reader(),
        Err(polylogue_store::StoreError::ReadPoolClosed)
    ));
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}
