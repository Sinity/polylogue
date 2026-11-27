use super::*;
use polylogue_model::bundle::Record;
use std::path::Path;

fn begin() -> serde_json::Value {
    serde_json::json!({
        "target":"query:synthetic", "view":"transcript",
        "title":{"state":"value", "value":"Synthetic session"},
        "kind":{"kind":"query", "query":"sessions"}, "figures":[],
        "snapshot": {
            "query": format!("query:{}", "a".repeat(64)),
            "query_text": "sessions",
            "ran_at": "2026-01-01T00:00:00Z",
            "read_position": {"index_instance":"synthetic", "index":1, "archive":1},
            "freshness": {
                "local": polylogue_model::freshness::ReadCompleteness::empty(),
                "sinex": {
                    "mode":"primary", "scope":"synthetic",
                    "applied_position":1, "sinex_position":1,
                    "sinex_position_seconds_ago":0.436835857,
                    "reachable":true, "lag_positions":0, "state":"current"
                }
            },
            "enabled_alternates":[], "author":"operator"
        },
        "options": ReadOptions::default()
    })
}

fn deliver(destination: &Path, failed: bool) -> Result<u8, OutputError> {
    let engine = polylogue_engine::Config::new(
        destination.parent().unwrap().to_owned(),
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        1,
        polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
    )
    .unwrap();
    let startup = crate::config::Startup::for_test(engine, destination);
    let parsed = super::super::line::parse(
        &[
            "sessions".into(),
            "read".into(),
            "--json".into(),
            "--format".into(),
            "jsonl".into(),
            "--to".into(),
            destination.to_str().unwrap().into(),
        ],
        &Default::default(),
        true,
    )
    .unwrap();
    let mut bytes = Vec::new();
    crate::protocol::stream::event(&mut bytes, "begin", &begin()).unwrap();
    crate::protocol::stream::event(
        &mut bytes,
        "record",
        &serde_json::json!({
            "kind":"original", "record":{"record":"session", "data":{
                "ref":"session:codex:synthetic", "commit":1,
                "session":{"id":"codex:synthetic", "kind":"main"}
            }}, "parts":[]
        }),
    )
    .unwrap();
    if failed {
        let answer: command_surface::Answer<serde_json::Value> =
            command_surface::Answer::error(crate::protocol::failure(
                polylogue_model::Code::ReadOutput,
                "synthetic stream interrupted",
            ));
        crate::protocol::stream::event(&mut bytes, "answer", &answer).unwrap();
    } else {
        let complete = ReadFooter {
            counts: Counts {
                sessions: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        crate::protocol::stream::event(&mut bytes, "complete", &complete).unwrap();
        let answer = command_surface::Answer::ok(serde_json::json!({
            "kind":"steps", "reply":{"resolved":null, "steps":[{
                "verb":"read", "request_id":null,
                "outcome":{"status":"done", "reply":{"kind":"streamed", "reply":complete}, "undo":[]}
            }]}
        }));
        crate::protocol::stream::event(&mut bytes, "steps", &answer).unwrap();
        crate::protocol::stream::event(&mut bytes, "answer", &answer).unwrap();
    }
    write_read_stream(
        &startup.client,
        &parsed,
        &mut Default::default(),
        |callback| crate::client::receive(&mut bytes.as_slice(), None, callback),
    )
}

#[test]
fn streamed_primary_bundle_keeps_decimal_observation_age() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    std::fs::create_dir(&destination).unwrap();
    assert_eq!(deliver(&destination, false).unwrap(), 0);
    let records: Vec<Record> = serde_json::Deserializer::from_reader(
        std::fs::File::open(destination.join("records.jsonl")).unwrap(),
    )
    .into_iter()
    .map(Result::unwrap)
    .collect();
    assert_eq!(records.len(), 2);
    let Record::Manifest(manifest) = &records[0] else {
        panic!("manifest first")
    };
    assert_eq!(
        serde_json::to_value(&manifest.freshness).unwrap(),
        begin()["snapshot"]["freshness"]
    );
    assert!(matches!(
        manifest.freshness.sinex,
        polylogue_model::freshness::IndexFreshness::Primary {
            sinex_position_seconds_ago: Some(age), ..
        } if age == 0.436835857
    ));
    assert_eq!(records[1].reference(), Some("session:codex:synthetic"));
}

#[test]
fn streamed_primary_bundle_preserves_terminal_error_without_publication() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("bundle");
    std::fs::create_dir(&destination).unwrap();
    let Err(OutputError::Answer(answer)) = deliver(&destination, true) else {
        panic!("the stream must return its terminal failure")
    };
    let command_surface::Outcome::Error(error) = answer.outcome else {
        panic!("a failure answer is required")
    };
    assert_eq!(error.code.as_deref(), Some("read_output"));
    assert_eq!(error.message, "synthetic stream interrupted");
    assert!(
        answer
            .envelope
            .notes
            .iter()
            .any(|note| note.contains("staged bundle kept at"))
    );
    assert!(!destination.join("manifest.json").exists());
    assert!(!destination.join("records.jsonl").exists());
    let kept: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".polylogue-bundle-")
        })
        .collect();
    assert_eq!(kept.len(), 1, "{kept:?}");
    assert!(kept[0].join("_staging/begin.json").is_file());
}

/// Runs a streamed line, printed for a person, over the given events.
fn stream_line(
    root: &Path,
    words: &[&str],
    events: &[(&str, serde_json::Value)],
) -> Result<u8, OutputError> {
    let engine = polylogue_engine::Config::new(
        root.to_owned(),
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        1,
        polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
    )
    .unwrap();
    let startup = crate::config::Startup::for_test(engine, &root.join("archive"));
    let words: Vec<String> = words.iter().map(|word| (*word).to_owned()).collect();
    let parsed = super::super::line::parse(&words, &Default::default(), true).unwrap();
    let mut bytes = Vec::new();
    for (event, data) in events {
        crate::protocol::stream::event(&mut bytes, event, data).unwrap();
    }
    write_read_stream(
        &startup.client,
        &parsed,
        &mut Default::default(),
        |callback| crate::client::receive(&mut bytes.as_slice(), None, callback),
    )
}

/// docs/interfaces.md#exit-codes: a line exits with its failed step's
/// code, and a step whose output cannot be delivered here has failed.
#[test]
fn streamed_line_with_failed_delivery_exits_nonzero() {
    use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome, StepReport, StepsAnswer};
    let root = tempfile::tempdir().unwrap();
    let steps = command_surface::Answer::ok(LineAnswer::Steps(StepsAnswer {
        resolved: None,
        steps: vec![StepReport {
            verb: "read".into(),
            request_id: None,
            outcome: StepOutcome::Done {
                reply: StepAnswer::Attachment(Box::new(crate::daemon::attachment::BytePage {
                    descriptor: None,
                    input: None,
                    blob: polylogue_model::BlobHash::of(b"synthetic attachment"),
                    size: 20,
                    offset: 0,
                    content_base64: "c3ludGhldGljIGF0dGFjaG1lbnQ=".into(),
                    next: None,
                })),
                undo: vec![],
            },
        }],
    }));
    let steps = serde_json::to_value(steps).unwrap();
    assert_eq!(
        stream_line(
            root.path(),
            // An attachment's bytes are one text reply, which a directory
            // cannot take.
            &["sessions", "read", "--to", "out/"],
            &[("steps", steps.clone()), ("answer", steps)],
        )
        .unwrap(),
        2
    );
}

/// A read this stream received degraded keeps the line degraded (5), even
/// when the line's final answer is ok.
#[test]
fn write_out_keeps_the_earlier_failure_status() {
    let root = tempfile::tempdir().unwrap();
    let destination = root.path().join("reading.jsonl");
    let mut header = begin();
    header["snapshot"]["incomplete"] = serde_json::json!(["synthetic missing index"]);
    let complete = ReadFooter {
        counts: Counts {
            sessions: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let answer = serde_json::to_value(command_surface::Answer::ok(serde_json::json!({
        "kind":"steps", "reply":{"resolved":null, "steps":[{
            "verb":"read", "request_id":null,
            "outcome":{"status":"done", "reply":{"kind":"streamed", "reply":complete}, "undo":[]}
        }]}
    })))
    .unwrap();
    let code = stream_line(
        root.path(),
        &[
            "sessions",
            "read",
            "--format",
            "jsonl",
            "--to",
            destination.to_str().unwrap(),
        ],
        &[
            ("begin", header),
            (
                "record",
                serde_json::json!({
                    "kind":"original", "record":{"record":"session", "data":{
                        "ref":"session:codex:synthetic", "commit":1,
                        "session":{"id":"codex:synthetic", "kind":"main"}
                    }}, "parts":[]
                }),
            ),
            ("complete", serde_json::to_value(&complete).unwrap()),
            ("steps", answer.clone()),
            ("answer", answer),
        ],
    )
    .unwrap();
    assert_eq!(code, 5);
    assert!(destination.is_file());
}

#[test]
fn a_client_read_deadline_keeps_the_timeout_and_next_request() {
    let root = tempfile::tempdir().unwrap();
    let engine = polylogue_engine::Config::new(
        root.path().to_owned(),
        Default::default(),
        "synthetic-machine".into(),
        "synthetic-host".into(),
        1,
        polylogue_model::WallClock::fixed("2026-01-01T00:00:00Z".parse().unwrap()),
    )
    .unwrap();
    let startup = crate::config::Startup::for_test(engine, root.path());
    let parsed = super::super::line::parse(
        &[
            "session:claude-code:ID".into(),
            "read".into(),
            "--follow".into(),
            "--deadline".into(),
            "200".into(),
        ],
        &Default::default(),
        false,
    )
    .unwrap();
    for remaining in [
        std::time::Duration::ZERO,
        std::time::Duration::from_millis(1),
    ] {
        let (mut socket, _peer) = std::os::unix::net::UnixStream::pair().unwrap();
        let until = std::time::Instant::now() + remaining;
        let result = write_read_stream(
            &startup.client,
            &parsed,
            &mut Default::default(),
            |callback| crate::client::receive(&mut socket, Some(until), callback),
        );
        let Err(OutputError::Answer(answer)) = result else {
            panic!("a client deadline must return the read deadline reply");
        };
        let error = answer.failure().unwrap();
        assert_eq!(
            (
                error.code.as_deref(),
                error.kind,
                error.retryable,
                answer.exit_code(),
                answer.http_status()
            ),
            (
                Some("deadline_exceeded"),
                command_surface::FailureKind::Timeout,
                true,
                7,
                Some(504)
            )
        );
        assert_eq!(error.message, "the read deadline was exceeded");
        assert_eq!(
            error.hint.as_deref(),
            Some("rerun with `--deadline none`, or narrow to one session")
        );
        assert_eq!(answer.envelope.next.len(), 1);
        let mut expected = super::super::run_line::request(&parsed).unwrap();
        expected.deadline = command_surface::Deadline::None;
        assert_eq!(
            answer.envelope.next[0].request.arguments,
            expected.arguments
        );
        assert_eq!(answer.envelope.next[0].request.deadline, expected.deadline);
    }
}
