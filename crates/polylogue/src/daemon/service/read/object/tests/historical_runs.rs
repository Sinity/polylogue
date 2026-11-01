use super::*;
use polylogue_model::{
    authored::ImportAttribution, historical::HistoricalRecord, query_objects::HistoricalRunReplay,
};

fn raw_read(id: &polylogue_model::RunId) -> Request {
    Request::Read {
        target: format!("run:{id}"),
        options: polylogue_model::read::ReadOptions {
            view: View::Raw,
            ..Default::default()
        },
        delivery: ReadDelivery::Stream,
        format: "jsonl".into(),
        budget: None,
        with_blobs: false,
    }
}
fn seed(d: &Daemon) -> polylogue_model::query_objects::HistoricalQueryRun {
    let original = polylogue_store::fixture::query_objects::synthetic_source3_run().unwrap();
    assert!(matches!(
        original.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
    d.writes
        .as_ref()
        .unwrap()
        .restore_history(
            HistoricalRecord::Run(Box::new(original.clone())),
            vec![],
            ImportAttribution {
                input: "synthetic-old-run".into(),
                chain: 1,
                machine: d.config.requests.machine().to_owned(),
                actor: Author::Operator,
                at: d.config.requests.now(),
            },
        )
        .unwrap();
    original
}
fn retained(
    d: &Daemon,
    id: &polylogue_model::RunId,
) -> polylogue_model::query_objects::HistoricalQueryRun {
    d.reads
        .reader()
        .unwrap()
        .historical_query_run(id)
        .unwrap()
        .unwrap()
}

#[test]
fn public_raw_historical_run_exports_imports_and_rebuilds_without_replay_authority() {
    let mut d = test_daemon::start("synthetic source");
    d.pause_signals();
    let original = seed(&d.daemon);
    let operation = d.daemon.operation_context(&actor()).unwrap();
    let mut replay = raw_read(&original.id);
    if let Request::Read { options, .. } = &mut replay {
        options.view = View::Transcript;
    }
    let refusal =
        super::super::super::produce(&replay, &d.daemon, &operation, &mut Vec::new()).unwrap();
    assert!(
        matches!(&refusal, Err(failure) if failure.code() == "not_built"
            && failure.message == "run references require their acknowledged execution binding is not built yet"),
        "{refusal:?}"
    );

    let (begin, record, complete) = streamed_request(&d.daemon, &raw_read(&original.id));
    assert_single_object_run(&d.daemon, &format!("run:{}", original.id), &begin);
    assert_eq!(complete.counts.runs, 1);
    assert_eq!(complete.counts.authored, 0);
    let Some(Record::Historical(row)) = record.record() else {
        panic!("missing historical Run")
    };
    let HistoricalRecord::Run(exported) = &row.evidence else {
        panic!("wrong historical grain")
    };
    assert_eq!(exported.encoded_run, original.encoded_run);
    assert_eq!(exported.question, original.question);
    assert_eq!(exported.declared_run_hash, None);
    assert_eq!(exported.declared_binding_hash, None);
    assert_eq!(exported.imports.observed.len(), 1);
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("run-bundle");
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
            request_id: "import-historical-run".into(),
            paths: vec![destination],
        },
        &receiver.daemon,
        &limit,
        &actor(),
    );
    assert!(
        matches!(
            imported,
            command_surface::Answer::Ok { .. }
                | command_surface::Answer::Empty { .. }
                | command_surface::Answer::Degraded { .. }
        ),
        "{imported:?}"
    );
    let restored = retained(&receiver.daemon, &original.id);
    assert_eq!(restored.encoded_run, original.encoded_run);
    assert_eq!(restored.source, original.source);
    assert_eq!(restored.question, original.question);
    assert_eq!(restored.declared_binding_hash, None);
    assert_eq!(restored.declared_run_hash, None);
    assert!(restored.native.observed.is_empty());
    assert_eq!(restored.imports.observed.len(), 1);
    assert!(
        restored
            .imports
            .declared
            .contains(&exported_import(&d.daemon, &original.id))
    );
    assert!(matches!(
        restored.replay,
        HistoricalRunReplay::Unavailable { .. }
    ));
    receiver.restart_rebuilding_index();
    receiver.pause_signals();
    assert_eq!(retained(&receiver.daemon, &original.id), restored);
    let (_, reread, done) = streamed_request(&receiver.daemon, &raw_read(&original.id));
    assert_eq!(done.counts.runs, 1);
    let Some(Record::Historical(row)) = reread.record() else {
        panic!("rebuild lost historical Run")
    };
    let HistoricalRecord::Run(run) = &row.evidence else {
        panic!("wrong historical grain")
    };
    assert_eq!(**run, restored);
}
fn exported_import(d: &Daemon, id: &polylogue_model::RunId) -> ImportAttribution {
    retained(d, id).imports.observed[0].clone()
}

#[test]
fn raw_historical_run_honors_scope_and_cancellation_before_record_body() {
    for scope_reset in [false, true] {
        let mut d = test_daemon::start("synthetic source");
        d.pause_signals();
        let original = seed(&d.daemon);
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
            super::super::super::produce(
                &raw_read(&original.id),
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

#[test]
fn raw_historical_run_page_preserves_record_and_small_budget_offers_streaming() {
    let mut d = test_daemon::start("synthetic source");
    d.pause_signals();
    let original = seed(&d.daemon);
    let mut request = raw_read(&original.id);
    if let Request::Read { delivery, .. } = &mut request {
        *delivery = ReadDelivery::Page;
    }
    let operation = d.daemon.operation_context(&actor()).unwrap();
    let reader = d.daemon.reads.reader().unwrap();
    let response = super::super::super::page(&reader, &d.daemon, &request, &operation);
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    let page: ReadPage = serde_json::from_value(data).unwrap();
    assert!(page.document.footer.position.is_none());
    let exported = page
        .document
        .items
        .iter()
        .find_map(|item| match &item.kind {
            polylogue_model::read::ItemKind::Other { data, .. } => data.get("record").cloned(),
            _ => None,
        })
        .expect("page omitted historical metadata");
    let exported: polylogue_model::bundle::HistoricalRecord =
        serde_json::from_value(exported).unwrap();
    let HistoricalRecord::Run(run) = exported.evidence else {
        panic!("wrong page grain")
    };
    assert_eq!(run.encoded_run, original.encoded_run);
    if let Request::Read { budget, .. } = &mut request {
        *budget = Some(1);
    }
    let response = super::super::super::page(&reader, &d.daemon, &request, &operation);
    assert!(
        matches!(response, command_surface::Answer::Error { ref error, .. } if error.code.as_deref() == Some("bad_request")),
        "{response:?}"
    );
    drop(reader);
    let (_, record, complete) = streamed_request(&d.daemon, &raw_read(&original.id));
    assert!(matches!(record.record(), Some(Record::Historical(_))));
    assert_eq!(complete.counts.runs, 1);
}
