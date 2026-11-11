use super::*;

#[test]
fn the_producer_never_waits_for_the_transport_and_bytes_arrive_in_order() {
    let directory = tempfile::tempdir().unwrap();
    let (mut output, input) = buffer(directory.path().to_owned());
    // Three times the memory share, written before the transport takes
    // anything: the rest goes to the spill file.
    let pieces: Vec<Vec<u8>> = (0..48u8).map(|n| vec![n; MEMORY / 16]).collect();
    for piece in &pieces {
        output.write_all(piece).unwrap();
    }
    // The transport drains part, then the producer writes more while the
    // spill file still holds unsent bytes.
    let mut received = Vec::new();
    while received.len() < MEMORY {
        let Next::Bytes(bytes) = input.next_blocking().unwrap() else {
            panic!("bytes come before the end");
        };
        received.extend(bytes);
    }
    output.write_all(b"tail").unwrap();
    output.finish(Ok(()));
    loop {
        match input.next_blocking().unwrap() {
            Next::Bytes(bytes) => received.extend(bytes),
            Next::End => break,
            Next::Failed(failure) => panic!("{failure:?}"),
        }
    }
    let mut expected: Vec<u8> = pieces.concat();
    expected.extend(b"tail");
    assert_eq!(received.len(), expected.len());
    assert!(received == expected, "bytes arrive in the order written");
}

#[test]
fn a_failure_before_the_first_byte_is_the_answer_and_after_it_an_error_event() {
    let directory = tempfile::tempdir().unwrap();
    let (output, input) = buffer(directory.path().to_owned());
    output.finish(Err(Failure::new("bad_request", "synthetic refusal")));
    let Next::Failed(failure) = input.next_blocking().unwrap() else {
        panic!("a failure before any byte is reported as such");
    };
    assert_eq!(failure.code(), "bad_request");

    let (mut output, input) = buffer(directory.path().to_owned());
    output.write_all(b"event: record\ndata: {}\n\n").unwrap();
    output.finish(Err(Failure::new("io", "synthetic failure")));
    let mut text = Vec::new();
    while let Next::Bytes(bytes) = input.next_blocking().unwrap() {
        text.extend(bytes);
    }
    let text = String::from_utf8(text).unwrap();
    assert!(text.starts_with("event: record\n"), "{text}");
    assert!(text.contains("event: error\n"), "{text}");
}

#[test]
fn a_gone_transport_stops_the_producer() {
    let directory = tempfile::tempdir().unwrap();
    let (mut output, input) = buffer(directory.path().to_owned());
    drop(input);
    let error = output.write_all(b"x").unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
}
