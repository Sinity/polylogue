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
fn secrets_compare_whole() {
    assert!(same_secret("abc", "abc"));
    assert!(!same_secret("abc", "abd"));
    assert!(!same_secret("abc", "abcd"));
}

#[test]
fn a_local_connection_is_owned_by_this_uid() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (server, peer) = listener.accept().unwrap();
    assert_eq!(peer, client.local_addr().unwrap());
    assert_eq!(socket_owner(peer, server.local_addr().unwrap()), my_uid());
    assert!(my_uid().is_some());
}

/// A closed connection from the same port stays listed in `TIME_WAIT` under
/// uid 0, and the port can also carry another user's connection elsewhere;
/// only the live entry with both ends names the owner, wherever it is listed.
#[test]
fn only_the_live_entry_with_both_ends_names_the_owner() {
    let table = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:9C40 0100007F:1F90 06 00000000:00000000 03:00000F9A 00000000     0        0 0 3 0000000000000000
   1: 0100007F:9C40 0100007F:0050 01 00000000:00000000 00:00000000 00000000  1001        0 4242 1 0000000000000000 20 4 30 10 -1
   2: 0100007F:9C40 0100007F:1F90 01 00000000:00000000 00:00000000 00000000  1000        0 4343 1 0000000000000000 20 4 30 10 -1
";
    assert_eq!(
        owner_in(table, "0100007F:9C40", "0100007F:1F90"),
        Some(1000)
    );
    assert_eq!(
        owner_in(table, "0100007F:9C40", "0100007F:0050"),
        Some(1001)
    );
    assert_eq!(owner_in(table, "0100007F:9C41", "0100007F:1F90"), None);
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
fn access_files_are_private() {
    let dir = tempfile::tempdir().unwrap();
    let rt = dir.path().join("k");
    write_access_files(&rt, "tok", 1234).unwrap();
    assert_eq!(fs::read_to_string(rt.join("token")).unwrap(), "tok");
    assert_eq!(fs::read_to_string(rt.join("port")).unwrap(), "1234");
    let mode = fs::metadata(rt.join("token")).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[test]
fn rewriting_access_files_restricts_existing_files_and_ignores_stale_temporary_paths() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    let other = dir.path().join("other");
    fs::write(&token, "old").unwrap();
    fs::set_permissions(&token, fs::Permissions::from_mode(0o644)).unwrap();
    fs::write(&other, "unchanged").unwrap();
    std::os::unix::fs::symlink(&other, dir.path().join("token.tmp")).unwrap();
    write_access_files(dir.path(), "new", 4321).unwrap();
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
fn reader_location_uses_the_written_port_and_requires_a_live_own_listener() {
    let dir = tempfile::tempdir().unwrap();
    assert!(reader_url_in(dir.path()).is_err());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    write_access_files(dir.path(), "synthetic-reader-token", port).unwrap();
    assert_eq!(
        reader_url_in(dir.path()).unwrap(),
        format!("http://127.0.0.1:{port}/auth#synthetic-reader-token")
    );
    drop(listener);
    assert!(reader_url_in(dir.path()).is_err());
}

#[test]
fn only_a_listener_on_127_0_0_1_counts() {
    let port = 0x1F4D;
    let table = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000:1F4D 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 1
   1: 0101A8C0:1F4D 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1001        0 2
   2: 0100007F:1F4D 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1002        0 3
";
    let loopback = std::net::SocketAddrV4::new(std::net::Ipv4Addr::LOCALHOST, port);
    // The wildcard and the LAN address listen on the same port first in the
    // table; only the loopback listener's owner is returned.
    assert_eq!(listener_in(table, &v4_hex(loopback)), Some(1002));
    let without = table.lines().take(3).collect::<Vec<_>>().join("\n");
    assert_eq!(listener_in(&without, &v4_hex(loopback)), None);
}
