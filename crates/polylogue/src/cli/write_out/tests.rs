use super::*;
use polylogue_model::bundle::Record;
use std::path::Path;

fn begin() -> serde_json::Value {
    serde_json::json!({
        "format": "jsonl",
        "snapshot": {
            "query": format!("query:{}", "a".repeat(64)),
            "query_text": "sessions",
            "ran_at": "2026-01-01T00:00:00Z",
            "read_at": {"generation":"synthetic", "index":1, "archive":1},
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
        "header": {
            "target":"query:synthetic", "view":"transcript",
            "title":{"state":"value", "value":"Synthetic session"},
            "kind":{"kind":"query", "query":"sessions"}, "figures":[]
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
        let complete = ReadComplete {
            footer: ReadFooter::default(),
            counts: Counts {
                sessions: 1,
                ..Default::default()
            },
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
    write_read_stream(&startup, &parsed, &mut Default::default(), |callback| {
        crate::client::receive(&mut bytes.as_slice(), None, callback)
    })
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
