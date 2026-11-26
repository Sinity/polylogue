use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::process::Output;
use std::time::{Duration, Instant};

fn answer(
    args: &[&str],
    delay: Duration,
    response: impl AsRef<[u8]> + Send + 'static,
    expected_deadline: command_surface::Deadline,
) -> Output {
    let directory = tempfile::tempdir().unwrap();
    let archive = directory.path().join("polylogue");
    std::fs::create_dir(&archive).unwrap();
    let listener = UnixListener::bind(archive.join("daemon.sock")).unwrap();
    listener.set_nonblocking(true).unwrap();
    let peer = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(5);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(accepted) => break accepted,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        Instant::now() < until,
                        "CLI never connected to the fake peer"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("accept: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = String::new();
        BufReader::new(&stream).read_line(&mut request).unwrap();
        let request: serde_json::Value = serde_json::from_str(&request).unwrap();
        let mut envelope = request;
        let caller: command_surface::CallerEvidence =
            serde_json::from_value(envelope.as_object_mut().unwrap().remove("caller").unwrap())
                .unwrap();
        assert!(!caller.controlling_terminal);
        let request: command_surface::RunnableRequest = serde_json::from_value(envelope).unwrap();
        assert_eq!(request.deadline, expected_deadline);
        if request.operation == "rows" {
            assert_eq!(request.arguments["selection"], "session:claude-code:s1");
            assert_eq!(request.arguments["steps"][0]["verb"], "read");
            assert_eq!(request.arguments["steps"][0]["delivery"], "stream");
            assert_eq!(request.arguments["steps"][0]["format"], "json");
        }
        std::thread::sleep(delay);
        stream.write_all(&framed(response.as_ref())).unwrap();
    });
    let output = crate::Fixture::new(
        &archive,
        &directory.path().join("home"),
        &directory.path().join("config"),
    )
    .command()
    .env("XDG_DATA_HOME", directory.path())
    .args(args)
    .output()
    .unwrap();
    peer.join().unwrap();
    output
}

/// A reply as the socket frames it: a finite answer is one `answer` frame;
/// a stream is already its frames.
fn framed(response: &[u8]) -> Vec<u8> {
    if response.starts_with(b"event:") {
        return response.to_vec();
    }
    let answer = std::str::from_utf8(response).unwrap().trim_end();
    format!("event: answer\ndata: {answer}\n\n").into_bytes()
}

#[test]
fn cli_reports_deadline_refusals_with_exit_seven() {
    let output = answer(&["--json", "status"], Duration::ZERO,
        b"{\"outcome\":\"error\",\"error\":{\"kind\":\"timeout\",\"code\":\"deadline_exceeded\",\"message\":\"the read deadline was exceeded; rows scanned unknown\",\"retryable\":true},\"notes\":[],\"next\":[]}\n", command_surface::Deadline::Default);
    assert_eq!(
        output.status.code(),
        Some(7),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["error"]["code"], "deadline_exceeded");
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_value(value).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("typed timeout required")
    };
    assert_eq!(error.kind, command_surface::FailureKind::Timeout);
    assert_eq!(
        error.message,
        "the read deadline was exceeded; rows scanned unknown"
    );
}

#[test]
fn a_read_timeout_does_not_turn_a_write_reply_into_a_failure() {
    let output = answer(
        &["--request-id", "synthetic-write", "--json", "set", "reading.alternates", "[]"],
        Duration::from_millis(1250),
        b"{\"outcome\":\"ok\",\"value\":{\"id\":\"synthetic-write\",\"repeat\":false,\"pending\":false},\"notes\":[],\"next\":[]}\n",
        command_surface::Deadline::Default,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let answer: command_surface::Answer<polylogue_model::user::UserOutcome> =
        serde_json::from_value(value).unwrap();
    let write_result = answer.value().expect("owned write reply");
    assert_eq!(write_result.id, "synthetic-write");
    assert!(!write_result.repeat && !write_result.pending);
}

/// A finished synthetic read stream and the document it delivers.
fn read_stream() -> (Vec<u8>, polylogue_model::read::ReadDocument) {
    use polylogue_model::read::{
        Format, HeaderKind, ItemKind, ReadBegin, ReadComplete, ReadDocument, ReadFooter,
        ReadHeader, ReadItem, ReadOptions, ReadPart, ReadRecord, Source, ValueState,
    };
    let header = ReadHeader {
        target: "session:claude-code:s1".into(),
        view: Default::default(),
        title: ValueState::NotRecorded,
        kind: HeaderKind::Record {
            record_type: "synthetic".into(),
        },
        figures: vec![],
        zone: None,
    };
    let item = ReadItem {
        source: Source::record("message:claude-code:m1"),
        kind: ItemKind::Text {
            text: "synthetic export".into(),
        },
    };
    let footer = ReadFooter {
        shown: 1,
        count: ValueState::known(1),
        ..Default::default()
    };
    let begin = ReadBegin {
        format: Format::Json,
        header: header.clone(),
        options: ReadOptions::default(),
        snapshot: polylogue_model::bundle::ReadSnapshot {
            query: format!("query:{}", "a".repeat(64)),
            query_text: "sessions".into(),
            ran_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            read_at: polylogue_model::ChangePosition {
                generation: "synthetic".into(),
                index: 1,
                archive: 1,
            },
            freshness: polylogue_model::freshness::ReadFreshness {
                local: polylogue_model::freshness::ReadCompleteness::empty(),
                sinex: polylogue_model::freshness::IndexFreshness::Standalone,
            },
            enabled_alternates: vec![],
            author: polylogue_model::Author::Operator,
            notes: vec![],
            incomplete: vec![],
        },
    };
    let record = ReadRecord::Presentation {
        parts: vec![ReadPart::Item(item.clone())],
    };
    let complete = ReadComplete {
        footer: footer.clone(),
        counts: Default::default(),
    };
    let mut wire = Vec::new();
    for (name, value) in [
        ("begin", serde_json::to_value(begin).unwrap()),
        ("record", serde_json::to_value(record).unwrap()),
        ("complete", serde_json::to_value(&complete).unwrap()),
    ] {
        wire.extend_from_slice(format!("event: {name}\ndata: {value}\n\n").as_bytes());
    }
    let steps = command_surface::Answer::ok(serde_json::json!({
        "kind":"steps", "reply":{"resolved":null,"steps":[{
            "verb":"read", "request_id":"synthetic-read", "outcome":{
                "status":"done", "reply":{"kind":"streamed","reply":complete}
            }
        }]}
    }));
    let answer = command_surface::Answer::ok(serde_json::json!({
        "kind":"steps", "reply":{"resolved":null,"steps":[]}
    }));
    for (name, reply) in [("steps", steps), ("answer", answer)] {
        let value = serde_json::to_string(&reply).unwrap();
        wire.extend_from_slice(format!("event: {name}\ndata: {value}\n\n").as_bytes());
    }
    let document = ReadDocument {
        header,
        items: vec![item],
        footer,
    };
    (wire, document)
}

#[test]
fn explicit_unbounded_read_stream_waits_beyond_the_read_default_and_finishes() {
    let (wire, expected) = read_stream();
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("read.json");
    let output = answer(
        &[
            "--deadline",
            "none",
            "session:claude-code:s1",
            "read",
            "--to",
            destination.to_str().unwrap(),
            "--format",
            "json",
        ],
        Duration::from_millis(1250),
        wire,
        command_surface::Deadline::None,
    );
    assert!(output.status.success(), "{output:?}");
    let actual: polylogue_model::read::ReadDocument =
        serde_json::from_slice(&std::fs::read(&destination).unwrap()).unwrap();
    assert_eq!(actual, expected);
}

/// The daemon gives a streamed read no default deadline, so neither does the
/// client: a stream that starts after the read default still finishes.
#[test]
fn a_read_stream_without_a_deadline_waits_for_the_daemon() {
    let (wire, expected) = read_stream();
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("read.json");
    let output = answer(
        &[
            "session:claude-code:s1",
            "read",
            "--to",
            destination.to_str().unwrap(),
            "--format",
            "json",
        ],
        Duration::from_millis(1250),
        wire,
        command_surface::Deadline::Default,
    );
    assert!(output.status.success(), "{output:?}");
    let actual: polylogue_model::read::ReadDocument =
        serde_json::from_slice(&std::fs::read(&destination).unwrap()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn a_command_timeout_overrides_the_global_default() {
    let output = answer(
        &["--deadline", "2000", "--json", "set"],
        Duration::from_millis(1250),
        b"{\"outcome\":\"ok\",\"value\":[],\"notes\":[],\"next\":[]}\n",
        command_surface::Deadline::Milliseconds(2000),
    );
    assert!(output.status.success(), "{output:?}");
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_value(value).unwrap();
    assert_eq!(answer.value().unwrap(), &serde_json::json!([]));
}

#[test]
fn a_typed_import_keeps_its_write_reply_beyond_the_aggregate_read_default() {
    let response = serde_json::to_vec(&command_surface::Answer::ok(
        polylogue_engine::ImportReport::default(),
    ))
    .unwrap();
    let output = answer(
        &["--json", "import"],
        Duration::from_millis(1250),
        [response, b"\n".to_vec()].concat(),
        command_surface::Deadline::Default,
    );
    assert!(output.status.success(), "{output:?}");
    let reply: command_surface::Answer<polylogue_engine::ImportReport> =
        serde_json::from_slice(&output.stdout).unwrap();
    reply.value().expect("typed import result");
}

#[test]
fn a_read_deadline_does_not_cancel_a_real_typed_import() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = crate::Fixture::new(
        &directory.path().join("archive"),
        &directory.path().join("home"),
        &directory.path().join("config"),
    );
    let _daemon = crate::cli::start_daemon(&fixture);
    let path = directory.path().join("conversations.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!([{
            "id":"deadline-import", "mapping":{"node":{
                "id":"node", "parent":null, "children":[], "message":{
                    "id":"deadline-message", "author":{"role":"user"},
                    "content":{"content_type":"text","parts":["synthetic import"]}
                }
            }}, "current_node":"node"
        }]))
        .unwrap(),
    )
    .unwrap();
    let output = fixture
        .command()
        .args(["--deadline", "0", "--json", "import"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let reply: command_surface::Answer<polylogue_engine::ImportReport> =
        serde_json::from_slice(&output.stdout).unwrap();
    let report = reply.value().expect("typed import result");
    assert_eq!(report.imported, [path]);
    assert!(report.refused.is_empty(), "{report:?}");

    // The session reached the archive and reads back through the daemon
    // once the loop has parsed and derived it.
    crate::cli::wait_status(&fixture, |_| true);
    let output = crate::cli::cli(&fixture, &["--deadline", "none", "--json", "sessions"]);
    assert!(output.status.success(), "{output:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let listing = crate::cli::listing(answer.value().expect("query succeeded").clone());
    let ids: Vec<_> = listing["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["chatgpt:deadline-import"]);
    let output = crate::cli::cli(
        &fixture,
        &[
            "--deadline",
            "none",
            "--json",
            "session:chatgpt:deadline-import",
            "read",
        ],
    );
    assert!(output.status.success(), "{output:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let read = answer.value().expect("read succeeded").clone();
    let items = read["reply"]["steps"][0]["outcome"]["reply"]["reply"]["document"]["items"]
        .as_array()
        .unwrap();
    let texts: Vec<_> = items
        .iter()
        .filter(|item| item["kind"]["kind"] == "text")
        .map(|item| item["kind"]["text"].as_str().unwrap())
        .collect();
    assert_eq!(texts, ["synthetic import"], "{read}");
}
