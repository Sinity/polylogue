use super::super::*;

fn header() -> Value {
    json!({"type":"session","id":"test:deadline","format":"polylogue-session","version":1})
}

#[test]
fn socket_timeout_is_a_deadline_error_and_no_end_is_written() {
    let (client, mut peer) = UnixStream::pair().unwrap();
    frame(&mut peer, header()).unwrap();
    frame(&mut peer, json!({"type":"text","data":"{}"})).unwrap();
    let mut out = Vec::new();
    let started = Instant::now();
    let error = decode(
        client,
        "test:deadline",
        &mut out,
        started + Duration::from_millis(80),
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert_eq!(error.to_string(), "Session read deadline exceeded");
    assert_eq!(out, b"{}"); // no accepted end frame/newline
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[test]
fn trickled_frames_do_not_extend_the_deadline() {
    let (client, mut peer) = UnixStream::pair().unwrap();
    frame(&mut peer, header()).unwrap();
    frame(&mut peer, json!({"type":"text","data":"x"})).unwrap();
    let send = std::thread::spawn(move || {
        for _ in 1..20 {
            if frame(&mut peer, json!({"type":"text","data":"x"})).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
    });
    let mut out = Vec::new();
    let started = Instant::now();
    let error = decode(
        client,
        "test:deadline",
        &mut out,
        started + Duration::from_millis(80),
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(!out.is_empty() && out.len() < 20);
    assert!(started.elapsed() < Duration::from_secs(1));
    send.join().unwrap();
}

#[test]
fn trickled_bytes_of_one_end_frame_do_not_extend_the_deadline() {
    let (client, mut peer) = UnixStream::pair().unwrap();
    frame(&mut peer, header()).unwrap();
    let mut end = serde_json::to_vec(&json!({"type":"end","outcome":"degraded","canonical_complete":false,"missing":["source_catalog_coverage"]})).unwrap();
    end.push(b'\n');
    // One frame spread over many reads: a timeout renewed per read would
    // accept this end frame late.
    peer.write_all(&end[..1]).unwrap();
    let send = std::thread::spawn(move || {
        for byte in &end[1..] {
            std::thread::sleep(Duration::from_millis(15));
            if peer.write_all(&[*byte]).is_err() {
                break;
            }
        }
    });
    let mut out = Vec::new();
    let started = Instant::now();
    let error = decode(
        client,
        "test:deadline",
        &mut out,
        started + Duration::from_millis(80),
    )
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(out.is_empty());
    assert!(started.elapsed() < Duration::from_secs(1));
    send.join().unwrap();
}
