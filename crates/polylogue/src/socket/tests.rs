use super::*;
use polylogue_store::Archive;
use std::time::{Duration, Instant};

#[test]
fn a_public_socket_appears_by_rename_after_it_listens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("daemon.sock");
    drop(UnixListener::bind(&path).unwrap());
    let events = inotify::init(inotify::CreateFlags::NONBLOCK).unwrap();
    inotify::add_watch(
        &events,
        dir.path(),
        inotify::WatchFlags::CREATE | inotify::WatchFlags::MOVED_TO,
    )
    .unwrap();
    let listener = bind(&path).unwrap();
    let mut bytes = [MaybeUninit::uninit(); 4096];
    let mut events = inotify::Reader::new(events, &mut bytes);
    let mut public = Vec::new();
    loop {
        match events.next() {
            Ok(event)
                if event
                    .file_name()
                    .is_some_and(|name| name.to_bytes() == b"daemon.sock") =>
            {
                public.push(event.events());
            }
            Ok(_) => {}
            Err(rustix::io::Errno::AGAIN) => break,
            Err(error) => panic!("{error}"),
        }
    }
    assert_eq!(public, [inotify::ReadFlags::MOVED_TO]);
    let connected = std::os::unix::net::UnixStream::connect(&path).unwrap();
    let (_accepted, _) = listener.accept().unwrap();
    drop(connected);
    assert!(!dir.path().join("daemon.sock.binding").exists());
}

#[test]
fn failed_publication_preserves_the_destination_and_removes_its_private_socket() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("daemon.sock");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("original"), b"synthetic original").unwrap();
    assert_eq!(bind(&path).unwrap_err().kind(), io::ErrorKind::IsADirectory);
    assert_eq!(
        std::fs::read(path.join("original")).unwrap(),
        b"synthetic original"
    );
    assert!(!dir.path().join("daemon.sock.binding").exists());
}

#[test]
fn private_publication_keeps_a_long_archive_socket_path_supported() {
    let dir = tempfile::tempdir().unwrap();
    let padding = 107 - dir.path().as_os_str().as_bytes().len() - "/daemon.sock".len() - 1;
    let archive = dir.path().join("a".repeat(padding));
    std::fs::create_dir(&archive).unwrap();
    let path = archive.join("daemon.sock");
    // This path fits a Unix socket, while appending .binding does not.
    drop(UnixListener::bind(&path).unwrap());
    let _listener = bind(&path).unwrap();
    std::os::unix::net::UnixStream::connect(path).unwrap();
}

#[test]
fn a_socket_path_no_client_can_connect_to_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let padding = 108 - dir.path().as_os_str().as_bytes().len() - "/daemon.sock".len() - 1;
    let archive = dir.path().join("a".repeat(padding));
    std::fs::create_dir(&archive).unwrap();
    let path = archive.join("daemon.sock");
    assert_eq!(path.as_os_str().as_bytes().len(), 108);
    let error = bind(&path).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert!(!path.exists());
}

#[test]
fn an_already_published_socket_connects_on_the_first_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("daemon.sock");
    let _listener = bind(&path).unwrap();
    crate::client::wait_for_socket(&path, Some(Instant::now() + Duration::from_secs(2))).unwrap();
}

/// Waits on another thread, which must still be waiting after a moment.
fn waiting(path: &std::path::Path) -> std::sync::mpsc::Receiver<io::Result<()>> {
    let (done, result) = std::sync::mpsc::channel();
    let path = path.to_owned();
    std::thread::spawn(move || {
        let _ = done.send(crate::client::wait_for_socket(&path, None));
    });
    assert!(
        matches!(
            result.recv_timeout(Duration::from_millis(50)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ),
        "the wait ended before anything happened"
    );
    result
}

#[test]
fn socket_publication_wakes_an_already_waiting_client() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let result = waiting(&path);
    let _listener = bind(&path).unwrap();
    result
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap();
}

#[test]
fn failed_archive_startup_wakes_an_unlimited_client() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let result = waiting(&path);
    drop(lock);
    let error = result
        .recv_timeout(Duration::from_secs(2))
        .unwrap()
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}

#[test]
fn socket_wait_times_out_at_its_deadline_and_returns_a_startup_failure_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let error = crate::client::wait_for_socket(&path, Some(Instant::now())).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    drop(lock);
    let error = crate::client::wait_for_socket(&path, None).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::NotFound);
}
