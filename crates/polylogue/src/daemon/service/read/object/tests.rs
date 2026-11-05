use super::*;
use crate::test_daemon;
use polylogue_model::{
    Author,
    assertion::{AssertionKind, NewAssertion},
    authorship::{Authorship, Channel},
    bundle::Record,
    user::UserWrite,
};

fn actor() -> Authorship {
    Authorship {
        author: Author::Operator,
        via: Channel::Cli,
    }
}
fn native_no_effect(d: &Daemon, id: &str) {
    let invocation = d.invocation(&actor());
    let write = || UserWrite::Assert {
        assertions: vec![NewAssertion {
            kind: AssertionKind::Tag,
            key: Some("read-object-test".into()),
            value: None,
            valid_until: None,
            links: vec![polylogue_model::assertion::NewAssertionLink {
                role: polylogue_model::assertion::LinkRole::About,
                reference: polylogue_model::Reference::parse("session:claude-code:synthetic")
                    .unwrap(),
            }],
        }],
    };
    d.writes
        .as_ref()
        .unwrap()
        .user(invocation.request("tag-base".into(), write()))
        .unwrap();
    d.writes
        .as_ref()
        .unwrap()
        .user(invocation.request(id.into(), write()))
        .unwrap();
}
fn read(id: &str) -> Request {
    Request::Read {
        target: format!("request:{id}"),
        options: Default::default(),
        delivery: ReadDelivery::Stream,
        format: "jsonl".into(),
        budget: None,
        with_blobs: false,
        cursor: None,
    }
}
fn frames(bytes: &[u8]) -> Vec<(String, serde_json::Value)> {
    std::str::from_utf8(bytes)
        .unwrap()
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let event = frame
                .lines()
                .find_map(|line| line.strip_prefix("event:"))
                .unwrap()
                .trim()
                .to_owned();
            let data = frame
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim_start)
                .collect::<Vec<_>>()
                .join("\n");
            (event, serde_json::from_str(&data).unwrap())
        })
        .collect()
}
fn streamed(d: &Daemon, id: &str) -> (ReadBegin, ReadRecord, ReadComplete) {
    streamed_request(d, &read(id))
}
fn streamed_request(d: &Daemon, request: &Request) -> (ReadBegin, ReadRecord, ReadComplete) {
    let operation = d.operation_context(&actor()).unwrap();
    let mut bytes = Vec::new();
    super::super::produce(request, d, &operation, &mut bytes)
        .unwrap()
        .unwrap();
    let frames = frames(&bytes);
    assert_eq!(
        frames
            .iter()
            .map(|(event, _)| event.as_str())
            .collect::<Vec<_>>(),
        ["begin", "record", "complete"]
    );
    (
        serde_json::from_value(frames[0].1.clone()).unwrap(),
        serde_json::from_value(frames[1].1.clone()).unwrap(),
        serde_json::from_value(frames[2].1.clone()).unwrap(),
    )
}
fn assert_single_object_run(d: &Daemon, reference: &str, begin: &ReadBegin) {
    let reader = d.reads.reader().unwrap();
    let count = reader
        .query(
            "SELECT count(*) AS n FROM source_query_run WHERE object_ref=?1 AND run IS NOT NULL",
            &[polylogue_store::SqlValue::Text(reference.into())],
        )
        .unwrap();
    assert_eq!(count[0]["n"], 1);
    let Reference::Run(id) = Reference::parse(&begin.snapshot.query).unwrap() else {
        panic!("object reading must cite its real Run, not a question")
    };
    let run = reader.query_run(&id, None).unwrap().unwrap().value;
    assert_eq!(run.question, None);
    assert!(
        matches!(run.binding, RunBinding::OriginalObject { execution } if execution.reference.to_string() == reference)
    );
    let outcome = reader.query_run_outcome(&id, None).unwrap().unwrap().value;
    assert_eq!(outcome.outcome, RunOutcome::Ok);
    assert_eq!(outcome.count, Some(1));
    assert!(outcome.exact);
}

#[test]
fn actual_no_effect_request_read_records_one_object_run_and_roundtrips_through_directory_import() {
    let mut d = test_daemon::start("synthetic input");
    d.pause_signals();
    native_no_effect(&d.daemon, "no-effect");
    let (begin, record, complete) = streamed(&d.daemon, "no-effect");
    assert_eq!(complete.counts.authored, 1);
    let Some(Record::Authored(original)) = record.record() else {
        panic!("whole accepted request omitted")
    };
    assert!(original.prepared.effect.assertions.is_empty());
    assert_eq!(original.prepared.outcome.id, "tag-base");
    assert_single_object_run(&d.daemon, "request:no-effect", &begin);
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("bundle");
    let expected = original.prepared.clone();
    let mut sink = crate::cli::write_out::bundle::Bundle::new(
        &destination,
        begin,
        polylogue_model::bundle::Format::Jsonl,
        false,
    )
    .unwrap();
    sink.push(record).unwrap();
    sink.finish(complete).unwrap();
    let mut receiver = test_daemon::start("independent receiving source");
    receiver.pause_signals();
    let limit = crate::daemon::execution::limit(None, Default::default());
    let imported = crate::daemon::service::handle_with_caller(
        Request::Import {
            request_id: "import-object".into(),
            paths: vec![destination],
        },
        &receiver.daemon,
        &limit,
        &actor(),
    );
    crate::test_answers::ok(imported);
    let reader = receiver.daemon.reads.reader().unwrap();
    let Some(polylogue_store::read::authored_evidence::AuthoredReadEvidence::Local {
        record, ..
    }) = reader
        .authored_record(&Reference::Request("no-effect".into()))
        .unwrap()
    else {
        panic!("import lost the accepted no-effect object")
    };
    assert_eq!(record.prepared, expected);
}

struct Interrupt<'a> {
    bytes: Vec<u8>,
    action: Option<Box<dyn FnOnce() + 'a>>,
}
impl Write for Interrupt<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(bytes);
        if let Some(action) = self.action.take() {
            action();
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[test]
fn actual_object_stream_stops_on_scope_reset_and_cancellation_before_record_body() {
    for scope_reset in [false, true] {
        let mut d = test_daemon::start("synthetic input");
        d.pause_signals();
        native_no_effect(&d.daemon, "interrupted-object");
        let operation = d.daemon.operation_context(&actor()).unwrap();
        let limit = crate::daemon::execution::limit(None, Default::default());
        let cancellation = limit.cancellation.clone();
        let archive = &d.daemon.archive;
        let mut output = Interrupt {
            bytes: vec![],
            action: Some(Box::new(move || {
                if scope_reset {
                    archive.advance_generation_for_test().unwrap();
                } else {
                    cancellation.cancel();
                }
            })),
        };
        let response = limit.scope(|| {
            super::super::produce(
                &read("interrupted-object"),
                &d.daemon,
                &operation,
                &mut output,
            )
            .unwrap()
        });
        let expected = if scope_reset {
            "scope_changed"
        } else {
            "cancelled"
        };
        assert!(
            matches!(&response, Err(failure) if failure.code() == expected),
            "{response:?}"
        );
        assert_eq!(
            frames(&output.bytes)
                .iter()
                .map(|(event, _)| event.as_str())
                .collect::<Vec<_>>(),
            ["begin"]
        );
    }
}
