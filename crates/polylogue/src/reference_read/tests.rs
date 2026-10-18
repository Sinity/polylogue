use super::*;
fn exchange(frames: Vec<Value>) -> (io::Result<Value>, Vec<u8>) {
    let dir = tempfile::tempdir_in("/tmp").unwrap();
    let path = dir.path().join("socket");
    let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
    let join = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = String::new();
        BufReader::new(stream.try_clone().unwrap())
            .read_line(&mut request)
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&request).unwrap()["command"],
            "read"
        );
        for value in frames {
            frame(&mut stream, value).unwrap();
        }
    });
    let mut output = Vec::new();
    let result = request(&path, "message:synthetic:m", &mut output);
    join.join().unwrap();
    (result, output)
}
fn initial() -> Vec<Value> {
    vec![
        json!({"type":"message","id":"synthetic:m","role":"assistant","written_by":"model"}),
        json!({"type":"block","kind":"text"}),
        json!({"type":"text","data":"stored text"}),
    ]
}
#[test]
fn stream_without_a_valid_end_frame_is_an_error() {
    for end in [
        None,
        Some(json!({"type":"end","outcome":"ok"})),
        Some(json!({"type":"end","outcome":"degraded","canonical_complete":true})),
    ] {
        let mut frames = initial();
        if let Some(end) = end {
            frames.push(end);
        }
        let (result, output) = exchange(frames);
        assert!(result.is_err());
        assert!(String::from_utf8(output).unwrap().contains("stored text"));
    }
    let mut frames = initial();
    frames.push(
        json!({"type":"end","outcome":"degraded","canonical_complete":false,"missing":["history"]}),
    );
    let (result, _) = exchange(frames);
    assert_eq!(result.unwrap()["canonical_complete"], false);
}
#[test]
fn message_header_must_match_the_requested_id() {
    for id in ["synthetic:other", "m", "", "synthetic:m:extra"] {
        let mut frames = initial();
        frames[0]["id"] = json!(id);
        let (result, output) = exchange(vec![frames.remove(0)]);
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(output.is_empty());
    }
    let mut frames = initial();
    frames.push(json!({"type":"end","outcome":"degraded","canonical_complete":false}));
    let (result, output) = exchange(frames);
    assert_eq!(result.unwrap()["outcome"], "degraded");
    assert_eq!(
        String::from_utf8(output).unwrap(),
        "synthetic:m\nassistant · model\n\nstored text\n"
    );
}
#[test]
fn missing_message_writes_nothing_and_returns_the_typed_error() {
    let (result, output) = exchange(vec![failed(ErrorKind::NotFound, "reference absent")]);
    assert!(output.is_empty());
    assert_eq!(result.unwrap()["error"]["kind"], "not_found");
}
#[test]
fn control_characters_are_escaped() {
    let mut output = Vec::new();
    terminal_text(&mut output, "hello\u{1b}[2J\r\u{85}\tworld\n", true).unwrap();
    assert_eq!(
        String::from_utf8(output).unwrap(),
        "hello\\x1b[2J\\x0d\\x85\tworld\n"
    );
    let mut output = Vec::new();
    terminal_text(&mut output, "id\nwith\ttabs", false).unwrap();
    assert_eq!(String::from_utf8(output).unwrap(), "id\\x0awith\\x09tabs");
}
