use super::*;
use polylogue_store::{ImportObservation, InputFormat, Writer};
fn source(bytes: &[u8]) -> (tempfile::TempDir, Writer, polylogue_store::StoredSource) {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(&dir.path().join("archive")).unwrap();
    let hash = writer.put_file_blob(bytes).unwrap();
    let stored = writer
        .store_immutable_input(
            hash,
            InputFormat::ClaudeCodeJsonl,
            &["claude-main", "synthetic"],
            &ImportObservation {
                id: "input",
                machine: "m",
                host: "h",
                path: "/synthetic.jsonl",
                seen_at: 1,
            },
            4096,
        )
        .unwrap();
    let source = writer.stored_source(&stored.id).unwrap().unwrap();
    (dir, writer, source)
}
#[test]
fn counted_length_matches_masked_output_across_a_utf8_boundary() {
    let secret = format!("ghp_{}", "A".repeat(36));
    let text = format!("{}🙂 {secret}\n", "a".repeat(65535));
    let (_dir, writer, source) = source(text.as_bytes());
    let reader = writer.read_handle().unwrap();
    reader.verify_source(&source, || Ok(())).unwrap();
    let mut check = || Ok(());
    let mut count = Count(0);
    present(&reader, &source, &mut count, &mut check).unwrap();
    let mut result = Vec::new();
    present(&reader, &source, &mut result, &mut check).unwrap();
    assert_eq!(count.0, result.len() as u64);
    assert_eq!(
        String::from_utf8(result).unwrap(),
        text.replace(&secret, "[credential: GitHub token, masked]")
    );
    assert_eq!(
        reader
            .read_source_range(&source.input.id, 1, source.binding, 0, source.input.length)
            .unwrap(),
        text.as_bytes()
    );
}
#[test]
fn escaped_binary_or_invalid_utf8_input_is_refused() {
    for bytes in [
        b"{\"text\":\"\\u0067hp_secret\"}\n".as_slice(),
        b"binary\0",
        b"bad\xff",
    ] {
        let (_dir, writer, source) = source(bytes);
        let reader = writer.read_handle().unwrap();
        assert!(present(&reader, &source, &mut Count(0), &mut || Ok(())).is_err());
    }
}
#[test]
fn client_disconnect_stops_preflight_early_and_leaves_source_intact() {
    let bytes = vec![b'a'; 1024 * 1024];
    let (_dir, writer, source) = source(&bytes);
    let reader = writer.read_handle().unwrap();
    let mut full_checks = 0;
    reader
        .verify_source(&source, || {
            full_checks += 1;
            Ok(())
        })
        .unwrap();
    let (server, client) = UnixStream::pair().unwrap();
    let mut client = Some(client);
    let mut checks = 0;
    let result = preflight(&reader, &source, &mut || {
        checks += 1;
        if checks == 2 {
            drop(client.take());
        }
        if crate::session_read::peer_gone(&server) {
            Err(Failure::new(ErrorKind::Io, "input reader disconnected"))
        } else {
            Ok(())
        }
    });
    assert_eq!(result.unwrap_err().kind, ErrorKind::Io);
    assert!(checks < full_checks);
    let after = reader.source(&source.input.id).unwrap().unwrap();
    assert_eq!(
        (after.input.id, after.input.chain, after.input.length),
        (
            source.input.id.clone(),
            source.input.chain,
            source.input.length
        )
    );
    reader.verify_source(&source, || Ok(())).unwrap();
}
#[test]
fn preflight_reports_timeout_and_unsupported_input_kinds() {
    let (_dir, writer, source) = source(b"escaped\\u0061");
    let reader = writer.read_handle().unwrap();
    assert_eq!(
        preflight(&reader, &source, &mut || Ok(()))
            .unwrap_err()
            .kind,
        ErrorKind::NotYetAvailable
    );
    assert_eq!(
        preflight(&reader, &source, &mut || Err(Failure::new(
            ErrorKind::Timeout,
            "expired"
        )))
        .unwrap_err()
        .kind,
        ErrorKind::Timeout
    );
}
#[test]
fn unknown_input_returns_not_found_through_daemon_and_client() {
    let (_dir, writer, _) = source(b"neutral");
    // Unix socket paths have a short length limit; keep the socket in /tmp.
    let runtime = tempfile::tempdir_in("/tmp").unwrap();
    let socket = runtime.path().join("read.sock");
    let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    let reader = Mutex::new(writer.read_handle().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut line)
            .unwrap();
        serve(&mut stream, &serde_json::from_str(&line).unwrap(), &reader).unwrap();
    });
    let mut output = Vec::new();
    let failure = request(&socket, "unknown", &mut output).unwrap_err();
    assert_eq!(failure.kind, ErrorKind::NotFound);
    assert!(output.is_empty());
    server.join().unwrap();
}
