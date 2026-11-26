use super::*;
use crate::test_daemon;
use polylogue_model::{
    Author,
    annotation::{AnnotationKind, NewAnnotation},
    authorship::{Authorship, Channel},
    bundle::Record,
    user::UserWrite,
};
use polylogue_model::{
    Reference,
    query_objects::{RunBinding, RunOutcome},
    read::*,
};
use std::io::Write;

fn actor() -> Authorship {
    Authorship {
        author: Author::Operator,
        via: Channel::Cli,
    }
}
fn repeat_write_with_no_effect(d: &Daemon, id: &str) {
    let invocation = d.invocation(&actor());
    let write = || UserWrite::Assert {
        annotations: vec![NewAnnotation {
            kind: AnnotationKind::Tag,
            key: Some("read-object-test".into()),
            value: None,
            valid_until: None,
            links: vec![polylogue_model::annotation::NewAnnotationLink {
                role: polylogue_model::annotation::LinkRole::About,
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
fn read(id: &str) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection": format!("request:{id}"), "steps":[{"verb":"read", "delivery": "stream", "format": "jsonl"}]}),
    )
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
/// The stream of a read as the daemon's socket delivers it: the transfer's
/// begin, its one record and its completion (docs/interfaces.md, finite
/// transfers), then the line's step report and answer.
fn streamed(d: &test_daemon::TestDaemon, id: &str) -> (ReadBegin, ReadRecord, ReadComplete) {
    let request = crate::client::request("rows", read(id).arguments, format!("stream-{id}"));
    let mut frames = Vec::new();
    let answer = crate::client::call_streaming(
        &d.daemon.archive,
        &request,
        Some(std::time::Duration::from_secs(10)),
        |frame| {
            frames.push((frame.event, serde_json::from_str(&frame.data).unwrap()));
            Ok(false)
        },
    )
    .unwrap()
    .expect("the line answers");
    frames.push(("answer".into(), serde_json::to_value(answer).unwrap()));
    let events: Vec<&str> = frames
        .iter()
        .map(|(event, _): &(String, serde_json::Value)| event.as_str())
        .collect();
    assert_eq!(events[..3], ["begin", "record", "complete"]);
    assert_eq!(events[3..], ["steps", "answer"]);
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
    let run = reader.query_run(&id).unwrap().unwrap();
    assert_eq!(run.question, None);
    assert!(
        matches!(run.binding, RunBinding::OriginalObject { execution } if execution.reference.to_string() == reference)
    );
    let outcome = reader.query_run_outcome(&id).unwrap().unwrap();
    assert_eq!(outcome.outcome, RunOutcome::Ok);
    assert_eq!(outcome.count, Some(1));
    assert!(outcome.exact);
}

#[test]
fn no_effect_request_read_records_one_object_run_and_roundtrips_through_directory_import() {
    let mut d = test_daemon::start("synthetic input");
    d.pause_signals();
    repeat_write_with_no_effect(&d.daemon, "no-effect");
    let (begin, record, complete) = streamed(&d, "no-effect");
    assert_eq!(complete.counts.authored, 1);
    let Some(Record::Authored(original)) = record.record() else {
        panic!("whole accepted request omitted")
    };
    assert!(original.prepared.effect.annotations.is_empty());
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
    let limit = polylogue_store::read::limit::ReadLimit::new(
        crate::daemon::execution::default_read_deadline(),
        Default::default(),
    );
    let imported =
        crate::test_daemon::Call::new("import", serde_json::json!({"paths": [destination]}))
            .id("import-object")
            .within(&limit.clone())
            .authorship(actor().clone())
            .answer(&receiver.daemon);
    crate::test_answers::ok(imported);
    let reader = receiver.daemon.reads.reader().unwrap();
    let Some(polylogue_store::read::authored_read::AuthoredRead::Local { record, .. }) = reader
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
fn object_stream_stops_on_scope_reset_and_cancellation_before_record_body() {
    for scope_reset in [false, true] {
        let mut d = test_daemon::start("synthetic input");
        d.pause_signals();
        repeat_write_with_no_effect(&d.daemon, "interrupted-object");
        let operation = d.daemon.operation_context(&actor()).unwrap();
        let limit = polylogue_store::read::limit::ReadLimit::new(
            crate::daemon::execution::default_read_deadline(),
            Default::default(),
        );
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
            let reader = d.daemon.reads.reader()?;
            polylogue_reading::stream(
                &reader,
                d.daemon.as_ref(),
                &crate::daemon::service::read::request_of(
                    crate::daemon::verbs::read::arguments(
                        &crate::protocol::steps::ReadStep {
                            delivery: ReadDelivery::Stream,
                            format: Some("jsonl".into()),
                            ..Default::default()
                        },
                        "request:interrupted-object".into(),
                    )
                    .unwrap(),
                    None,
                )
                .unwrap(),
                &operation,
                &mut output,
            )
        });
        let expected = if scope_reset {
            polylogue_model::Code::ScopeChanged
        } else {
            polylogue_model::Code::Cancelled
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

fn assert_object_page_run(daemon: &Daemon, page: &ReadPage) {
    assert_eq!(page.document.header.target, page.snapshot.query);
    let Reference::Run(id) = Reference::parse(&page.snapshot.query).unwrap() else {
        panic!("an object page names its committed run")
    };
    let reader = daemon.reads.reader().unwrap();
    let run = reader.query_run(&id).unwrap().unwrap();
    assert!(
        matches!(run.binding, RunBinding::OriginalObject { execution }
        if execution.reference == Reference::Request("large-object".into()))
    );
    let outcome = reader.query_run_outcome(&id).unwrap().unwrap();
    assert_eq!(outcome.outcome, RunOutcome::Ok);
    assert_eq!(outcome.count, Some(1));
    assert!(outcome.exact);
    for item in &page.document.items {
        if matches!(item.kind, ItemKind::Fragment { .. }) {
            assert_eq!(item.source, Source::record(page.snapshot.query.clone()));
        }
    }
}

#[test]
fn object_read_pages_past_its_budget() {
    let mut daemon = test_daemon::start("synthetic object pages");
    daemon.pause_signals();
    let request = daemon.daemon.invocation(&actor()).request(
        "large-object".into(),
        UserWrite::Assert {
            annotations: vec![NewAnnotation {
                kind: AnnotationKind::Note,
                key: None,
                value: Some(serde_json::json!("SYNTHETIC λ🙂 ".repeat(64))),
                valid_until: None,
                links: vec![polylogue_model::annotation::NewAnnotationLink {
                    role: polylogue_model::annotation::LinkRole::About,
                    reference: Reference::parse("session:claude-code:s1").unwrap(),
                }],
            }],
        },
    );
    daemon
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .user(request.clone())
        .unwrap();
    let reference = Reference::Request(request.id.clone());
    let expected = {
        let reader = daemon.daemon.reads.reader().unwrap();
        let Some(polylogue_store::read::authored_read::AuthoredRead::Local { record, .. }) =
            reader.authored_record(&reference).unwrap()
        else {
            panic!("stored request")
        };
        assert_eq!(record.prepared.request, request);
        let [fact] = record.prepared.effect.annotations.as_slice() else {
            panic!("one stored annotation")
        };
        assert_eq!(fact.kind, AnnotationKind::Note);
        assert_eq!(
            fact.value,
            Some(serde_json::json!("SYNTHETIC λ🙂 ".repeat(64)))
        );
        ReadItem {
            source: Source::record(reference.to_string()),
            kind: ItemKind::Other {
                type_name: "authored".into(),
                data: serde_json::json!({"record": record, "selection": AuthoredSelection {
                    selected: reference, coverage: AuthoredCoverage::WholeRequest,
                }}),
            },
        }
    };
    let mut call = test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection":"request:large-object", "steps":[{"verb":"read", "budget":64}]}),
    );
    let mut pending = String::new();
    let mut items = Vec::new();
    let mut first: Option<Vec<ReadItem>> = None;
    let mut pages = 0;
    loop {
        let answer = call.answer(&daemon.daemon);
        let page = crate::test_answers::read(answer);
        assert_object_page_run(&daemon.daemon, &page);
        if pages == 1 {
            let previous = page
                .document
                .footer
                .previous
                .as_ref()
                .expect("previous object page");
            let answer = test_daemon::Call::from_request(&previous.request).answer(&daemon.daemon);
            let before = crate::test_answers::read(answer);
            assert_object_page_run(&daemon.daemon, &before);
            let mut expected = first.clone().unwrap();
            for item in &mut expected {
                if matches!(item.kind, ItemKind::Fragment { .. }) {
                    item.source = Source::record(before.snapshot.query.clone());
                }
            }
            assert_eq!(before.document.items, expected);
        }
        if pages == 0 {
            first = Some(page.document.items.clone());
        }
        for item in &page.document.items {
            match &item.kind {
                ItemKind::Fragment {
                    text,
                    position,
                    until,
                    total,
                } => {
                    assert_eq!(position.character, pending.chars().count() as u64);
                    pending.push_str(text);
                    assert_eq!(*until, pending.chars().count() as u64);
                    if until == total {
                        assert_eq!(
                            position.item_hash,
                            polylogue_model::BlobHash::of(pending.as_bytes())
                        );
                        items.push(serde_json::from_str::<ReadItem>(&pending).unwrap());
                        pending.clear();
                    }
                }
                _ => {
                    assert!(pending.is_empty());
                    items.push(item.clone());
                }
            }
        }
        pages += 1;
        let Some(next) = page.document.footer.next else {
            break;
        };
        call = test_daemon::Call::from_request(&next.request);
    }
    assert!(pages > 1);
    assert!(pending.is_empty());
    assert_eq!(items, [expected]);
}
