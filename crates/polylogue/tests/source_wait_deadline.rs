//! Actual CLI transport controls. Synthetic peers prove deadlines, not source settlement.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};
const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");
fn receipt(settled: bool) -> Value {
    let mut value = json!({"outcome":"ok","retained":{"id":"synthetic-source","chain":1,"length":7},"source_rows":{"scope":"imported_source_rows","basis":{"input":"synthetic-source","original_hash":polylogue_model::BlobHash::of(b"raw"),"chain":1,"revision":1,"length":7,"parser_version":1,"interpretation":polylogue_model::BlobHash::of(b"basis")},"state":"pending","obligations":["canonical source rows"]}});
    if settled {
        value["source_rows"]["state"] = json!("settled");
        value["source_rows"]
            .as_object_mut()
            .unwrap()
            .remove("obligations");
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
            // Check-once makes no progress RPC at all.
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
                let _ = stream.write_all(&reply);
            }
            Peer::Trickle => {
                // Each byte is timely under an idle timeout; the frame never
                // completes before the ONE absolute deadline.
                for byte in reply {
                    if stream.write_all(&[byte]).is_err() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(40));
                }
            }
            Peer::Stalled | Peer::Cancelled => {}
            _ => unreachable!(),
        }
        // Client must drop its exact stream on timeout/cancellation, without
        // leaving a detached blocked transport helper owning the descriptor.
        let mut byte = [0];
        match stream.read(&mut byte) {
            Ok(0) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::BrokenPipe
                ) => {}
            other => panic!("client transport remains open: {other:?}"),
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
        assert_eq!(value["retained"], receipt(false)["retained"]);
        assert_eq!(
            value["source_rows"]["basis"],
            receipt(false)["source_rows"]["basis"]
        );
        assert_eq!(value["request_id"], "01900000-0000-7000-8000-000000000599");
        assert_eq!(value["seen_at"], 17);
        if !success {
            assert_eq!(value["error"]["kind"], "timeout");
            assert_eq!(value["source_rows"]["state"], "pending");
        }
        // Allows scheduling jitter, rejects the reproduced 2.4s late success.
        assert!(
            elapsed < Duration::from_millis(1900),
            "{mode:?}: {elapsed:?}"
        );
    }
    peer.join().unwrap();
}
#[test]
fn delayed_settled_frame_cannot_succeed_after_absolute_wait_deadline() {
    exercise(Peer::Delayed)
}
#[test]
fn stalled_progress_drops_transport_at_deadline() {
    exercise(Peer::Stalled)
}
#[test]
fn trickled_frame_cannot_renew_absolute_wait_deadline() {
    exercise(Peer::Trickle)
}
#[test]
fn ordinary_fast_progress_keeps_fixed_receipt_and_succeeds() {
    exercise(Peer::Fast)
}
#[test]
fn zero_wait_checks_admission_once_without_rpc() {
    exercise(Peer::InitialSettled);
    exercise(Peer::ZeroPending)
}
#[test]
fn cancelled_waiter_closes_exact_transport_without_waiting_for_owner() {
    exercise(Peer::Cancelled)
}

#[test]
fn already_expired_transport_does_not_connect_or_write() {
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
