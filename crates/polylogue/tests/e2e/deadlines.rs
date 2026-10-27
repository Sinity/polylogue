use super::BIN;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::process::Command;

#[test]
fn cli_reports_deadline_refusals_with_exit_seven() {
    let directory = tempfile::tempdir().unwrap();
    let archive = directory.path().join("polylogue");
    std::fs::create_dir(&archive).unwrap();
    let listener = UnixListener::bind(archive.join("daemon.sock")).unwrap();
    let peer = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = String::new();
        BufReader::new(&stream).read_line(&mut request).unwrap();
        let request: serde_json::Value = serde_json::from_str(&request).unwrap();
        assert!(request["deadline_ms"].as_i64().is_some());
        stream.write_all(b"{\"result\":\"error\",\"kind\":\"deadline_exceeded\",\"message\":\"the read deadline was exceeded\",\"detail\":{\"count\":null}}\n").unwrap();
    });
    let output = Command::new(BIN)
        .env("XDG_DATA_HOME", directory.path())
        .env_remove("POLYLOGUE_ARCHIVE")
        .args(["--json", "status"])
        .output()
        .unwrap();
    peer.join().unwrap();
    assert_eq!(
        output.status.code(),
        Some(7),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["error"]["kind"], "deadline_exceeded");
    assert!(value["error"]["detail"]["count"].is_null());
}
