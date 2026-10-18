use super::*;
use std::{
    io::{BufRead, BufReader},
    os::unix::net::{UnixListener, UnixStream},
    thread,
    time::Duration,
};
#[test]
fn dialogue_refuses_unknown_block_kinds_instead_of_skipping_them() {
    for kind in ["image", "file"] {
        assert_eq!(
            View::Dialogue
                .skip_reason(WrittenBy::Human, kind, false)
                .unwrap_err()["error"]["reason"],
            "unsupported_model_block_kind"
        );
    }
    assert_eq!(
        body_kind("unknown-body").err().unwrap()["error"]["reason"],
        "unsupported_prior_block_kind"
    );
    assert_eq!(
        View::Dialogue
            .skip_reason(WrittenBy::Model, "unknown-body", false)
            .unwrap_err()["error"]["reason"],
        "unsupported_model_block_kind"
    );
    assert_eq!(
        View::Dialogue
            .skip_reason(WrittenBy::Harness, "tool_result", false)
            .unwrap(),
        Some("author")
    );
    assert_eq!(
        View::Dialogue
            .skip_reason(WrittenBy::Model, "thinking", true)
            .unwrap(),
        None
    );
    assert!(
        View::Native
            .skip_reason(WrittenBy::Model, "tool_call", false)
            .is_err()
    );
}
#[test]
fn credential_masks_and_control_escapes_are_never_split() {
    let mut w = Window {
        start: 0,
        at: 0,
        text: String::new(),
        ends: vec![],
        limit: 1024,
        full: false,
        present: true,
    };
    let mut masker = credential_masking::Masker::new();
    masker.push("🙂\u{1b} ghp_", &mut w).unwrap();
    masker.push(&"A".repeat(36), &mut w).unwrap();
    masker.finish(&mut w).unwrap();
    assert_eq!(w.text, "🙂\\x1b [credential: GitHub token, masked]");
    let boundaries = w.ends.iter().map(|(_, n)| *n).collect::<Vec<_>>();
    assert_eq!(boundaries, vec![1, 5, 6, 40]);
}
#[test]
fn continuation_tokens_hold_only_hashes_and_positions_and_reject_any_edit() {
    let key = [42; 32];
    let fingerprint = [13; 32];
    let p = Position {
        block: 2,
        scalar: 123456,
        budget: 4000,
    };
    let token = seal(&fingerprint, "block:neutral", &p, &key);
    let (f, r, q) = open(&token, &key).unwrap();
    assert_eq!(f, fingerprint);
    assert_eq!(&r, blake3::hash(b"block:neutral").as_bytes());
    assert_eq!(q.scalar, p.scalar);
    let mut bytes = token.into_bytes();
    bytes[0] = if bytes[0] == b'0' { b'1' } else { b'0' };
    assert!(open(std::str::from_utf8(&bytes).unwrap(), &key).is_err());
}

#[test]
fn full_window_stops_write_all_instead_of_an_interrupted_retry_loop() {
    let mut w = Window {
        start: 0,
        at: 0,
        text: String::new(),
        ends: vec![],
        limit: 1,
        full: false,
        present: true,
    };
    assert!(w.write_all("🙂".as_bytes()).is_err());
    assert!(w.full);
}

#[test]
fn model_transport_trickled_frame_cannot_renew_the_absolute_read_deadline() {
    let short = tempfile::tempdir_in("/tmp").unwrap();
    let path = short.path().join("socket");
    let listener = UnixListener::bind(&path).unwrap();
    let peer = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut request)
            .unwrap();
        crate::reference_read::frame(&mut stream, json!({"type":"model-page"})).unwrap();
        // Trickle a frame far longer than the client's deadline; only the
        // client giving up at that deadline ends the stream early.
        let mut disconnected = false;
        let frame = serde_json::to_vec(&json!({"type":"text","data":"x".repeat(3000)})).unwrap();
        for byte in frame {
            if stream.write_all(&[byte]).is_err() {
                disconnected = true;
                break;
            }
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            disconnected,
            "trickled peer kept the client connection open"
        );
    });
    let started = Instant::now();
    let error = request_until(
        &path,
        &json!({"command":"model-read","reference":"synthetic","budget_tokens":4000}),
        started + Duration::from_millis(100),
    )
    .unwrap_err();
    assert!(
        matches!(
            error.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ),
        "{error}"
    );
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    peer.join().unwrap();
}
#[test]
fn model_transport_blocked_response_uses_the_original_absolute_write_deadline() {
    let (stream, _peer) = UnixStream::pair().unwrap();
    socket2::SockRef::from(&stream)
        .set_send_buffer_size(4096)
        .unwrap();
    let page = json!({"synthetic_transport_control":"z".repeat(1_000_000)});
    let started = Instant::now();
    let error =
        write_response_until(&stream, &page, started + Duration::from_millis(100)).unwrap_err();
    assert!(
        matches!(
            error.kind(),
            io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
        ),
        "{error}"
    );
    // The peer never reads, so only the deadline ends the write.
    // Time spent building the page does not extend the write deadline.
    assert_eq!(
        write_response_until(&stream, &json!({}), started)
            .unwrap_err()
            .kind(),
        io::ErrorKind::TimedOut
    );
}
