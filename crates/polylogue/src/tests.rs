use super::*;

// Unix socket paths are limited to 108 bytes, so sockets live under /tmp
// rather than the (long) managed TMPDIR. Only the socket goes there.
fn short_runtime() -> tempfile::TempDir {
    tempfile::tempdir_in("/tmp").unwrap()
}
#[test]
fn partial_writes_do_not_extend_the_deadline() {
    let (stream, _unread) = UnixStream::pair().unwrap();
    socket2::SockRef::from(&stream)
        .set_send_buffer_size(1024)
        .unwrap();
    let started = std::time::Instant::now();
    let mut transport = MetadataTransport {
        stream,
        deadline: Some(started + Duration::from_millis(80)),
        cancellation: None,
    };
    let error = transport.write_all(&vec![b'x'; 65536]).unwrap_err();
    assert!(matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ));
    assert!(started.elapsed() < Duration::from_secs(1));
}
#[test]
fn connect_to_a_full_listener_stops_at_the_deadline() {
    let temp = tempfile::tempdir_in("/tmp").unwrap();
    let path = temp.path().join("socket");
    let listener =
        socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None).unwrap();
    listener
        .bind(&socket2::SockAddr::unix(&path).unwrap())
        .unwrap();
    listener.listen(0).unwrap();
    let _queued = UnixStream::connect(&path).unwrap();
    let started = std::time::Instant::now();
    assert!(
        request_value_until(
            &path,
            &json!({"command":"progress"}),
            started + Duration::from_millis(80)
        )
        .is_err()
    );
    assert!(started.elapsed() < Duration::from_secs(1));
}
#[test]
fn stopping_one_daemon_leaves_another_archive_daemon_running() {
    let temp = tempfile::tempdir().unwrap();
    let runtime_owner = short_runtime();
    let runtime = runtime_owner.path().join("runtime");
    let mut first = Running::start(&temp.path().join("a"), &runtime).unwrap();
    let second = Running::start(&temp.path().join("b"), &runtime).unwrap();
    assert_eq!(request(first.socket()).unwrap()["outcome"], "ok");
    assert_eq!(request(second.socket()).unwrap()["outcome"], "ok");
    let mut stalled = UnixStream::connect(first.socket()).unwrap();
    stalled.write_all(b"{\"command\":").unwrap();
    let first_path = first.socket().to_owned();
    first.stop();
    assert!(!first_path.exists());
    assert!(request(second.socket()).is_ok());
    let replacement = Running::start(&temp.path().join("a"), &runtime).unwrap();
    drop(first);
    assert!(request(replacement.socket()).is_ok());
}
#[test]
fn failed_startup_removes_its_socket_and_releases_the_writer() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let runtime_owner = short_runtime();
    let runtime = runtime_owner.path().join("runtime");
    assert!(Running::start_inner(&archive, &runtime, true).is_err());
    let socket = socket_path(&fs::canonicalize(&archive).unwrap(), &runtime);
    assert!(!socket.exists());
    let running = Running::start(&archive, &runtime).unwrap();
    assert!(request(running.socket()).is_ok());
}
#[test]
fn socket_held_by_another_process_is_left_alone() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let writer = Writer::open(&archive).unwrap();
    drop(writer);
    let runtime_owner = short_runtime();
    let runtime = runtime_owner.path().join("runtime");
    let socket = socket_path(&fs::canonicalize(&archive).unwrap(), &runtime);
    fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let foreign = UnixListener::bind(&socket).unwrap();
    let inode = fs::symlink_metadata(&socket).unwrap().ino();
    assert!(Running::start(&archive, &runtime).is_err());
    assert_eq!(fs::symlink_metadata(&socket).unwrap().ino(), inode);
    assert!(UnixStream::connect(&socket).is_ok());
    drop(foreign);
}
#[test]
fn daemon_does_not_remove_a_socket_it_no_longer_owns() {
    let temp = tempfile::tempdir().unwrap();
    let runtime_owner = short_runtime();
    let running = Running::start(&temp.path().join("archive"), runtime_owner.path()).unwrap();
    let path = running.socket().to_owned();
    fs::remove_file(&path).unwrap();
    let foreign = UnixListener::bind(&path).unwrap();
    let inode = fs::symlink_metadata(&path).unwrap().ino();
    drop(running);
    assert_eq!(fs::symlink_metadata(&path).unwrap().ino(), inode);
    drop(foreign);
}
