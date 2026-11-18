use super::*;
use polylogue_store::Archive;
use std::future::Future;
use std::task::Poll;
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
    assert!(bind(&path).is_err());
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

#[tokio::test]
async fn an_already_published_socket_is_seen_after_registering_the_watch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("daemon.sock");
    let _listener = bind(&path).unwrap();
    crate::client::wait_for_socket(&path, Some(Instant::now() + Duration::from_secs(2)))
        .await
        .unwrap();
}

#[tokio::test]
async fn socket_publication_wakes_an_already_waiting_client() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let wait = crate::client::wait_for_socket(&path, None);
    tokio::pin!(wait);
    std::future::poll_fn(|context| {
        assert!(wait.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
    let _listener = bind(&path).unwrap();
    tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn failed_archive_startup_wakes_an_unlimited_client() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let wait = crate::client::wait_for_socket(&path, None);
    tokio::pin!(wait);
    std::future::poll_fn(|context| {
        assert!(wait.as_mut().poll(context).is_pending());
        Poll::Ready(())
    })
    .await;
    drop(lock);
    let error = tokio::time::timeout(Duration::from_secs(2), wait)
        .await
        .unwrap()
        .unwrap_err();
    assert!(crate::cli::is_down(&error));
}

#[tokio::test]
async fn socket_wait_observes_a_startup_failure_before_registration_and_its_deadline() {
    let dir = tempfile::tempdir().unwrap();
    let (_archive, lock) = Archive::open_or_create(dir.path()).unwrap();
    let path = dir.path().join("daemon.sock");
    let error = crate::client::wait_for_socket(&path, Some(Instant::now()))
        .await
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    drop(lock);
    let error = crate::client::wait_for_socket(&path, None)
        .await
        .unwrap_err();
    assert!(crate::cli::is_down(&error));
}
