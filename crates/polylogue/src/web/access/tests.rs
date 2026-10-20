use super::*;
use std::net::TcpListener;

#[test]
fn host_must_name_this_listener() {
    assert!(host_allowed(Some("127.0.0.1:8765"), 8765));
    assert!(host_allowed(Some("localhost:8765"), 8765));
    assert!(host_allowed(Some("[::1]:8765"), 8765));
    assert!(!host_allowed(Some("evil.example:8765"), 8765));
    assert!(!host_allowed(Some("127.0.0.1:8766"), 8765));
    assert!(!host_allowed(Some("127.0.0.1"), 8765));
    assert!(!host_allowed(None, 8765));
}

#[test]
fn the_cookie_is_named_after_the_archive() {
    let name = cookie_name("abc123");
    assert_eq!(
        cookie_token(Some("x=1; polylogue_abc123=tok; y=2"), &name),
        Some("tok")
    );
    assert_eq!(cookie_token(Some("polylogue_other=tok"), &name), None);
    assert_eq!(cookie_token(None, &name), None);
}

#[test]
fn keys_differ_by_archive_and_are_twelve_hex_digits() {
    let a = archive_key(Path::new("/tmp/a-archive"));
    let b = archive_key(Path::new("/tmp/b-archive"));
    assert_eq!(a.len(), 12);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b);
}

#[test]
fn secrets_compare_whole() {
    assert!(same_secret("abc", "abc"));
    assert!(!same_secret("abc", "abd"));
    assert!(!same_secret("abc", "abcd"));
}

#[test]
fn a_local_connection_is_owned_by_this_uid() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (_server, peer) = listener.accept().unwrap();
    assert_eq!(peer, client.local_addr().unwrap());
    assert_eq!(socket_owner(peer), my_uid());
    assert!(my_uid().is_some());
}

#[test]
fn the_archive_path_is_scrubbed() {
    let root = Path::new("/home/x/.local/share/polylogue");
    assert_eq!(
        scrub("file /home/x/.local/share/polylogue/blobs/ab", root),
        "file [archive]/blobs/ab"
    );
}

#[test]
fn publish_writes_private_files() {
    let dir = tempfile::tempdir().unwrap();
    let rt = dir.path().join("k");
    publish(&rt, "tok", 1234).unwrap();
    assert_eq!(fs::read_to_string(rt.join("token")).unwrap(), "tok");
    assert_eq!(fs::read_to_string(rt.join("port")).unwrap(), "1234");
    let mode = fs::metadata(rt.join("token")).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[test]
fn republishing_restricts_existing_files_and_ignores_stale_temporary_paths() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    let other = dir.path().join("other");
    fs::write(&token, "old").unwrap();
    fs::set_permissions(&token, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(&other, "unchanged").unwrap();
    std::os::unix::fs::symlink(&other, dir.path().join("token.tmp")).unwrap();
    publish(dir.path(), "new", 4321).unwrap();
    assert_eq!(fs::read_to_string(token).unwrap(), "new");
    assert_eq!(fs::read_to_string(other).unwrap(), "unchanged");
    assert_eq!(
        fs::metadata(dir.path().join("token"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
}

#[test]
fn reader_location_uses_the_published_port_and_requires_a_live_own_listener() {
    let dir = tempfile::tempdir().unwrap();
    assert!(reader_url_in(dir.path()).is_err());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    publish(dir.path(), "synthetic-reader-token", port).unwrap();
    assert_eq!(
        reader_url_in(dir.path()).unwrap(),
        format!("http://127.0.0.1:{port}/auth#synthetic-reader-token")
    );
    drop(listener);
    assert!(reader_url_in(dir.path()).is_err());
}
