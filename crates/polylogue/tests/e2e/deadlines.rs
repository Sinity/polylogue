use super::BIN;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::process::{Command, Output};
use std::time::Duration;

fn answer(args: &[&str], delay: Duration, response: &'static [u8]) -> Output {
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
        std::thread::sleep(delay);
        stream.write_all(response).unwrap();
    });
    let output = Command::new(BIN)
        .env("XDG_DATA_HOME", directory.path())
        .env("POLYLOGUE_READ_TIMEOUT_MS", "20")
        .env_remove("POLYLOGUE_ARCHIVE")
        .args(args)
        .output()
        .unwrap();
    peer.join().unwrap();
    output
}

#[test]
fn cli_reports_deadline_refusals_with_exit_seven() {
    let output = answer(&["--json", "status"], Duration::ZERO,
        b"{\"result\":\"error\",\"kind\":\"deadline_exceeded\",\"message\":\"the read deadline was exceeded\",\"detail\":{\"count\":null}}\n");
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

#[test]
fn a_read_timeout_does_not_turn_a_write_reply_into_a_failure() {
    let output = answer(
        &["--json", "setting", "synthetic", "yes"],
        Duration::from_millis(1250),
        b"{\"result\":\"ok\",\"data\":{\"key\":\"synthetic\",\"value\":\"yes\"}}\n",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["value"], "yes");
}
