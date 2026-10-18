//! `import --wait` deadline behavior, against a fake daemon socket.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
fn receipt(complete: bool) -> Value {
    let mut value = json!({"outcome":"ok","input":{"id":"synthetic-source","chain":1,"length":7},"source_rows":{"scope":"imported_source_rows","parsed_from":{"input":"synthetic-source","original_hash":polylogue_model::BlobHash::of(b"raw"),"chain":1,"revision":1,"length":7,"parser_version":1,"interpretation":polylogue_model::BlobHash::of(b"basis")},"state":"pending","pending_work":["canonical source rows"]}});
    if complete {
        value["source_rows"]["state"] = json!("complete");
        value["source_rows"]
            .as_object_mut()
            .unwrap()
            .remove("pending_work");
    }
    value
}
fn request(stream: &UnixStream) -> Value {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}
#[derive(Clone, Copy, Debug)]
enum Peer {
    Fast,
    Delayed,
    Stalled,
    Trickle,
    InitialSettled,
    ZeroPending,
    Cancelled,
}
fn exercise(mode: Peer) {
    let work = tempfile::tempdir().unwrap();
    let short = tempfile::tempdir_in("/tmp").unwrap();
    let archive = work.path().join("archive");
    fs::create_dir(&archive).unwrap();
    let socket = polylogue::socket_path(&fs::canonicalize(&archive).unwrap(), short.path());
    fs::create_dir_all(socket.parent().unwrap()).unwrap();
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let (accepted, ready) = mpsc::channel();
    let peer = thread::spawn(move || {
        let accept = || {
            let until = Instant::now() + Duration::from_secs(5);
            loop {
                match listener.accept() {
                    Ok((stream, _)) => return stream,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < until, "peer accept deadline");
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(e) => panic!("accept: {e}"),
                }
            }
        };
        let mut first = accept();
        assert_eq!(request(&first)["command"], "import");
        writeln!(first, "{}", receipt(matches!(mode, Peer::InitialSettled))).unwrap();
        drop(first);
        if matches!(mode, Peer::InitialSettled | Peer::ZeroPending) {
            // With a zero timeout the client sends no progress request.
            thread::sleep(Duration::from_millis(150));
            assert_eq!(
                listener.accept().unwrap_err().kind(),
                std::io::ErrorKind::WouldBlock
            );
            return;
        }
        let mut stream = accept();
        assert_eq!(
            request(&stream),
            json!({"command":"progress","input":"synthetic-source"})
        );
        accepted.send(()).unwrap();
        let reply = (receipt(true).to_string() + "\n").into_bytes();
        match mode {
            Peer::Fast => stream.write_all(&reply).unwrap(),
            Peer::Delayed => {
                thread::sleep(Duration::from_millis(2400));
                assert!(
                    stream.write_all(&reply).is_err(),
                    "late peer can still write to client"
                );
            }
            Peer::Trickle => {
                // Each byte arrives in time for a per-read timeout, but the
                // whole reply does not arrive before the overall deadline.
                let mut disconnected = false;
                for byte in reply {
                    if stream.write_all(&[byte]).is_err() {
                        disconnected = true;
                        break;
                    }
                    thread::sleep(Duration::from_millis(40));
                }
                assert!(disconnected, "trickled peer did not observe client release");
            }
            Peer::Stalled | Peer::Cancelled => {
                thread::sleep(if matches!(mode, Peer::Cancelled) {
                    Duration::from_millis(200)
                } else {
                    Duration::from_millis(1300)
                });
                // The client half-closes its write side before reading, so
                // EOF here would not show that it closed the connection.
                assert!(
                    stream.write_all(&reply).is_err(),
                    "client transport remains owned after timeout/cancellation"
                );
            }
            _ => unreachable!(),
        }
    });
    let provider = work.path().join("input.jsonl");
    fs::write(&provider, b"{}\n").unwrap();
    let context = work.path().join("context.json");
    fs::write(
        &context,
        br#"{"version":1,"context":["claude-main","synthetic"]}"#,
    )
    .unwrap();
    let stdout = tempfile::tempfile().unwrap();
    let stderr = tempfile::tempfile().unwrap();
    let zero = matches!(mode, Peer::InitialSettled | Peer::ZeroPending);
    let started = Instant::now();
    let mut child = Command::new(BINARY)
        .args([
            "--archive",
            archive.to_str().unwrap(),
            "import",
            provider.to_str().unwrap(),
            "--format",
            "claude-code-jsonl",
            "--context",
            context.to_str().unwrap(),
            "--request-id",
            "01900000-0000-7000-8000-000000000599",
            "--seen-at",
            "17",
            "--machine",
            "00112233445566778899aabbccddeeff",
            "--host",
            "synthetic-collector",
            "--wait",
            "--wait-timeout",
            if zero {
                "0"
            } else if matches!(mode, Peer::Cancelled) {
                "10"
            } else {
                "1"
            },
        ])
        .env("XDG_RUNTIME_DIR", short.path())
        .stdout(Stdio::from(stdout.try_clone().unwrap()))
        .stderr(Stdio::from(stderr.try_clone().unwrap()))
        .spawn()
        .unwrap();
    if matches!(mode, Peer::Cancelled) {
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        child.kill().unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("CLI exceeded bounded test deadline")
        }
        thread::sleep(Duration::from_millis(2));
    };
    let elapsed = started.elapsed();
    use std::os::unix::fs::FileExt;
    let mut bytes = vec![0; stdout.metadata().unwrap().len() as usize];
    stdout.read_exact_at(&mut bytes, 0).unwrap();
    if !matches!(mode, Peer::Cancelled) {
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        let success = matches!(mode, Peer::Fast | Peer::InitialSettled);
        assert_eq!(
            status.code(),
            Some(if success { 0 } else { 7 }),
            "{mode:?}: {value}"
        );
        assert_eq!(value["input"], receipt(false)["input"]);
        assert_eq!(
            value["source_rows"]["parsed_from"],
            receipt(false)["source_rows"]["parsed_from"]
        );
        assert_eq!(value["request_id"], "01900000-0000-7000-8000-000000000599");
        assert_eq!(value["seen_at"], 17);
        if !success {
            assert_eq!(value["error"]["kind"], "timeout");
            assert_eq!(value["source_rows"]["state"], "pending");
        }
        // Allows scheduling jitter but not a success seconds late.
        assert!(
            elapsed < Duration::from_millis(1900),
            "{mode:?}: {elapsed:?}"
        );
    }
    peer.join().unwrap();
}
#[test]
fn complete_reply_after_the_deadline_is_a_timeout() {
    exercise(Peer::Delayed)
}
#[test]
fn stalled_progress_closes_the_connection_at_the_deadline() {
    exercise(Peer::Stalled)
}
#[test]
fn slowly_trickled_reply_does_not_extend_the_deadline() {
    exercise(Peer::Trickle)
}
#[test]
fn fast_progress_reply_succeeds_with_the_original_receipt() {
    exercise(Peer::Fast)
}
#[test]
fn zero_timeout_checks_the_import_reply_without_a_request() {
    exercise(Peer::InitialSettled);
    exercise(Peer::ZeroPending)
}
#[test]
fn interrupted_wait_closes_its_connection_without_waiting() {
    exercise(Peer::Cancelled)
}

#[test]
fn expired_deadline_does_not_connect() {
    let short = tempfile::tempdir_in("/tmp").unwrap();
    let socket = short.path().join("socket");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let result = polylogue::request_value_until(
        Path::new(&socket),
        &json!({"command":"progress"}),
        Instant::now(),
    );
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::TimedOut);
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}
