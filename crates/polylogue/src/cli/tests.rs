use super::*;

#[test]
fn a_missing_daemon_is_daemon_required() {
    let dir = tempfile::tempdir().unwrap();
    let err = send(&dir.path().join("daemon.sock"), &Request::Status).unwrap_err();
    assert!(is_down(&err));
    assert!(!Archive::at(dir.path()).lock_held());
    let err = send_when_ready(&Archive::at(dir.path()), &Request::Status).unwrap_err();
    assert!(is_down(&err));
}

#[test]
fn a_starting_daemon_is_waited_for() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    // The daemon's order: take the lock, open the archive, then bind.
    let (_, lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(archive.lock_held());
    let socket = archive.socket();
    let server = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(&stream), &mut line).unwrap();
        let reply = serde_json::to_string(&Response::Ok {
            data: serde_json::json!({"ready": true}),
        })
        .unwrap();
        std::io::Write::write_all(&mut &stream, format!("{reply}\n").as_bytes()).unwrap();
        drop(lock);
    });
    let response = send_when_ready(&archive, &Request::Status).unwrap();
    assert!(matches!(response, Response::Ok { .. }));
    server.join().unwrap();
}

#[test]
fn manual_is_an_offline_topic() {
    let manual = COMMANDS.iter().find(|c| c.name == "manual").unwrap();
    assert!(matches!(manual.action, Action::Local(_)));
    let matches = command()
        .try_get_matches_from(["polylogue", "manual"])
        .unwrap();
    assert_eq!(run(&matches), std::process::ExitCode::SUCCESS);
}
