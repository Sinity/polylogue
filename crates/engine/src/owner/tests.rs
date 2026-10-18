use super::*;
use std::{path::Path, time::Instant};

/// Upper bound for owner work in tests; generous because the suite runs on a
/// loaded machine.
const WAIT: Duration = Duration::from_secs(60);

fn request(path: &Path, id: &str) -> ImportRequest {
    ImportRequest {
        format: InputFormat::CodexRollout,
        context: vec![],
        path: path.to_owned(),
        observation: Observation {
            id: id.into(),
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: path.to_str().unwrap().into(),
            seen_at: 0,
        },
        metadata_allowance: 4096,
    }
}
mod cancellation;
mod changed_inputs;
mod claude_turns;
mod configured;
mod history_turns;
mod obsolete_canonical;
mod reader_shutdown;
fn owner(root: &Path) -> Owner {
    Owner::start(
        Writer::open(root).unwrap(),
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap()
}
fn import(handle: &Handle, request: ImportRequest) -> StoredInput {
    handle
        .try_import(request)
        .unwrap()
        .recv_timeout(WAIT)
        .unwrap()
        .unwrap()
}
fn progress(handle: &Handle, id: &str) -> Progress {
    let end = Instant::now() + WAIT;
    loop {
        match handle.try_progress(id) {
            Ok(reply) => {
                return reply.recv_timeout(WAIT).unwrap().unwrap();
            }
            Err(Error::Busy) if Instant::now() < end => thread::yield_now(),
            other => panic!("progress admission: {:?}", other.err()),
        }
    }
}
fn until_caught_up(handle: &Handle, id: &str) -> Progress {
    let end = Instant::now() + WAIT;
    loop {
        let p = progress(handle, id);
        assert!(p.failure.is_none(), "{:?}", p.failure);
        assert!(p.recovery_error.is_none(), "{:?}", p.recovery_error);
        if p.parser.as_ref().is_some_and(caught_up) {
            return p;
        }
        assert!(
            Instant::now() < end,
            "provider did not reach stable frontier"
        );
        thread::yield_now();
    }
}
fn raw() -> &'static [u8] {
    b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"synthetic-session\",\"source\":\"cli\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"id\":\"u\",\"role\":\"user\",\"content\":[{\"text\":\"hello\"}]}}\n"
}
#[test]
fn authored_reply_does_not_open_derived_status_after_commit() {
    use polylogue_model::authored::{NoteKind, Target, WriteNote};
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let writer = Writer::open(root).unwrap();
    let reader = writer.read_handle().unwrap();
    // Keep the open connections and make only a new derived-tier open fail.
    std::fs::rename(root.join("embeddings.db"), root.join("held-embeddings.db")).unwrap();
    std::fs::create_dir(root.join("embeddings.db")).unwrap();
    assert!(writer.status().is_err());
    let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    let request = WriteNote {
        request_id: "01900000-0000-7000-8000-000000000501".into(),
        author: "agent".into(),
        machine: "synthetic-machine".into(),
        created_at: None,
        kind: Some(NoteKind::Correction),
        value: "Correction survives a derived status outage.".into(),
        targets: Some(vec![Target {
            reference: "message:future".into(),
            seen_hash: None,
        }]),
        relations: vec![],
    };
    let committed = service
        .handle()
        .try_note(request.clone(), Instant::now() + Duration::from_secs(30))
        .unwrap()
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    assert!(committed.recovery.is_none());
    let retry = service
        .handle()
        .try_note(request, Instant::now() + Duration::from_secs(30))
        .unwrap()
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    assert_eq!(committed.revision, retry.revision);
    assert_eq!(
        reader
            .authored_snapshot()
            .unwrap()
            .family(&committed.revision.reference())
            .unwrap()
            .history
            .len(),
        1
    );
    std::fs::remove_dir(root.join("embeddings.db")).unwrap();
    std::fs::rename(root.join("held-embeddings.db"), root.join("embeddings.db")).unwrap();
    service.shutdown().unwrap();
}
#[test]
fn declared_claude_context_replays_without_provider_and_remains_pending() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("untrusted-download-name.jsonl");
    let raw=b"{\"type\":\"user\",\"sessionId\":\"declared-main\",\"uuid\":\"u\",\"message\":{\"role\":\"user\",\"content\":\"retained prompt\"}}\n";
    std::fs::write(&path, raw).unwrap();
    let mut request = request(&path, "claude-observation");
    request.format = InputFormat::ClaudeCodeJsonl;
    request.context = vec!["claude-main".into(), "declared-main".into()];
    let root = dir.path().join("archive");
    let service = owner(&root);
    let retained = import(&service.handle(), request.clone());
    let deadline = Instant::now() + WAIT;
    loop {
        let p = progress(&service.handle(), &retained.id);
        assert_eq!(p.source_rows.parsed_from.input, p.input.id);
        if let Some(head) = p.parser
            && head.parsed_upto == retained.length
        {
            assert_eq!(head.counts.used, 1);
            assert_eq!(
                head.finalization_pending,
                !matches!(
                    p.source_rows.state,
                    polylogue_model::source_rows::SourceRowState::Complete
                )
            );
            assert_ne!(head.status, Phase::Done);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Claude parser did not publish its checkpoint"
        );
        thread::yield_now();
    }
    std::fs::remove_file(&path).unwrap();
    assert_eq!(import(&service.handle(), request.clone()), retained);
    let mut wrong = request.clone();
    wrong.context[1] = "another-main".into();
    assert!(matches!(
        service
            .handle()
            .try_import(wrong)
            .unwrap()
            .recv_timeout(WAIT)
            .unwrap(),
        Err(Error::Store(StoreError::ClientRequestConflict { .. }))
    ));
    service.shutdown().unwrap();
    let service = owner(&root);
    assert_eq!(import(&service.handle(), request), retained);
    assert_eq!(
        progress(&service.handle(), &retained.id)
            .source_rows
            .parsed_from
            .input,
        retained.id
    );
    service.shutdown().unwrap();
}
#[test]
fn claude_actor_selects_each_pending_locator_and_replays_after_provider_loss() {
    use polylogue_sources::{claude_code::keyed::MemoryReadView, frame::drive_jsonl};
    let raw = concat!(
        "{\"type\":\"user\",\"sessionId\":\"declared-main\",\"uuid\":\"u\",\"promptSource\":\"typed\",\"message\":{\"role\":\"user\",\"content\":\"retained question\"}}\n",
        "{\"type\":\"assistant\",\"sessionId\":\"declared-main\",\"uuid\":\"a\",\"parentUuid\":\"u\",\"message\":{\"id\":\"answer\",\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"retained answer\"}],\"stop_reason\":\"end_turn\"}}\n"
    );
    let framed = drive_jsonl(
        claude_code::KeyedRecords::new(
            claude_code::FileCtx::main("declared-main"),
            MemoryReadView::default(),
        )
        .unwrap(),
        raw.as_bytes(),
        0,
    );
    let ids = framed
        .output
        .unwrap()
        .delta
        .pending_messages
        .into_iter()
        .map(|p| p.message)
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("opaque-download.jsonl");
    std::fs::write(&path, raw).unwrap();
    let root = dir.path().join("archive");
    let writer = Writer::open(&root).unwrap();
    let reader = writer.read_handle().unwrap();
    let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    let mut r = request(&path, "claude-selected");
    r.format = InputFormat::ClaudeCodeJsonl;
    r.context = vec!["claude-main".into(), "declared-main".into()];
    let retained = import(&service.handle(), r.clone());
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let p = progress(&service.handle(), &retained.id);
        assert!(p.failure.is_none(), "{:?}", p.failure);
        assert_eq!(p.source_rows.parsed_from.input, p.input.id);
        let selected = ids
            .iter()
            .map(|id| reader.selected_message(id).unwrap())
            .collect::<Vec<_>>();
        if selected.iter().all(Option::is_some) {
            for (view, text) in selected
                .into_iter()
                .zip(["retained question", "retained answer"])
            {
                let view = view.unwrap();
                let block = view
                    .block(polylogue_model::BlockSlot::Item(0))
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    block.text_range(0, 64).unwrap().unwrap().bytes,
                    text.as_bytes()
                );
            }
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Claude pending locators did not become selected"
        );
        thread::yield_now();
    }
    std::fs::remove_file(&path).unwrap();
    assert_eq!(import(&service.handle(), r.clone()), retained);
    service.shutdown().unwrap();
    drop(reader);
    let service = owner(&root);
    assert_eq!(import(&service.handle(), r), retained);
    assert_eq!(
        progress(&service.handle(), &retained.id)
            .source_rows
            .parsed_from
            .input,
        retained.id
    );
    service.shutdown().unwrap();
}
#[test]
fn actor_selects_a_source_message_but_history_stays_unfinished() {
    use polylogue_sources::{
        codex::keyed::{KeyedRecords, MemoryReadView, Resume},
        frame::drive_jsonl,
    };
    let framed = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        raw(),
        0,
    );
    let delta = framed.output.unwrap();
    let mut memory = MemoryReadView::default();
    memory.apply(&delta);
    let message = memory.projected_message(0).unwrap().unwrap().id;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic-canonical.jsonl");
    std::fs::write(&path, raw()).unwrap();
    let writer = Writer::open(&dir.path().join("archive")).unwrap();
    let reader = writer.read_handle().unwrap();
    let service = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    let retained = import(
        &service.handle(),
        request(&path, "synthetic-canonical-request"),
    );
    std::fs::remove_file(&path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let p = progress(&service.handle(), &retained.id);
        assert!(p.failure.is_none(), "{:?}", p.failure);
        assert_eq!(p.source_rows.parsed_from.input, p.input.id);
        if let Some(view) = reader.selected_message(&message).unwrap() {
            let block = view
                .block(polylogue_model::BlockSlot::Item(0))
                .unwrap()
                .unwrap();
            let range = block.text_range(0, 64).unwrap().unwrap();
            assert_eq!(range.bytes, b"hello");
            assert_eq!(
                p.parser.unwrap().finalization_pending,
                !matches!(
                    p.source_rows.state,
                    polylogue_model::source_rows::SourceRowState::Complete
                )
            );
            break;
        }
        assert!(
            Instant::now() < deadline,
            "canonical source message did not become readable"
        );
        thread::yield_now();
    }
    // Receipt replay works after the provider file is gone, while the actor
    // serves selected-Message readers.
    assert_eq!(
        import(
            &service.handle(),
            request(&path, "synthetic-canonical-request")
        )
        .id,
        retained.id
    );
    service.shutdown().unwrap();
    assert!(reader.selected_message(&message).unwrap().is_some());
}
#[test]
fn stored_receipt_replays_before_missing_source_and_recovers_provider_state() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let path = dir.path().join("synthetic.jsonl");
    std::fs::write(&path, raw()).unwrap();
    let service = owner(&root);
    let retained = import(&service.handle(), request(&path, "synthetic-request"));
    std::fs::remove_file(&path).unwrap();
    let replay = import(&service.handle(), request(&path, "synthetic-request"));
    assert_eq!(retained.id, replay.id);
    let mut changed = request(&path, "synthetic-request");
    changed.observation.seen_at = 1;
    assert!(matches!(
        service
            .handle()
            .try_import(changed)
            .unwrap()
            .recv_timeout(WAIT)
            .unwrap(),
        Err(Error::Store(_))
    ));
    service.shutdown().unwrap();
    let service = owner(&root);
    let p = until_caught_up(&service.handle(), &retained.id);
    let head = p.parser.unwrap();
    assert_eq!(head.counts.framed, 2);
    assert_eq!(head.parsed_upto, raw().len() as u64);
    assert!(matches!(head.status, Phase::Finalizing | Phase::Done));
    assert_eq!(
        head.finalization_pending,
        !matches!(
            p.source_rows.state,
            polylogue_model::source_rows::SourceRowState::Complete
        )
    );
    assert_eq!(p.source_rows.parsed_from.input, p.input.id);
    service.shutdown().unwrap();
}
#[test]
fn partial_tail_is_stable_across_restart_and_retry() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let path = dir.path().join("synthetic-tail.jsonl");
    let mut bytes = raw().to_vec();
    bytes.extend_from_slice(b"{\"type\":");
    std::fs::write(&path, &bytes).unwrap();
    let service = owner(&root);
    let retained = import(&service.handle(), request(&path, "synthetic-tail-request"));
    let before = until_caught_up(&service.handle(), &retained.id)
        .parser
        .unwrap();
    assert_eq!(before.status, Phase::WaitingTail);
    assert_eq!(before.parsed_upto, raw().len() as u64);
    assert_eq!(before.inspected_upto, bytes.len() as u64);
    service.shutdown().unwrap();
    let service = owner(&root);
    service
        .handle()
        .try_retry(&retained.id)
        .unwrap()
        .recv_timeout(WAIT)
        .unwrap()
        .unwrap();
    for _ in 0..8 {
        let p = progress(&service.handle(), &retained.id);
        let head = p.parser.unwrap();
        assert_eq!(head.token, before.token);
        assert_eq!(head.counts, before.counts);
        assert_eq!(
            head.finalization_pending,
            !matches!(
                p.source_rows.state,
                polylogue_model::source_rows::SourceRowState::Complete
            )
        );
        assert_eq!(p.source_rows.parsed_from.input, p.input.id);
    }
    service.shutdown().unwrap();
}
#[test]
fn oversized_logical_record_is_an_inspectable_admission_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic-large.jsonl");
    let mut bytes = b"{\"type\":\"synthetic\",\"payload\":\"".to_vec();
    bytes.resize(1024 * 1024 + 64, b'x');
    bytes.extend_from_slice(b"\"}\n");
    std::fs::write(&path, &bytes).unwrap();
    let service = owner(&dir.path().join("archive"));
    let retained = import(&service.handle(), request(&path, "synthetic-large-request"));
    let deadline = Instant::now() + WAIT;
    loop {
        let p = progress(&service.handle(), &retained.id);
        if let Some(failure) = p.failure {
            assert_eq!(
                failure,
                format!(
                    "source framing requires {} bytes (allocation failed: false)",
                    bytes.len() - 1
                )
            );
            let head = p.parser.unwrap();
            assert_eq!(head.parsed_upto, 0);
            assert_eq!(head.counts.framed, 0);
            assert_ne!(head.status, Phase::Done);
            assert_eq!(p.source_rows.parsed_from.input, p.input.id);
            break;
        }
        assert!(Instant::now() < deadline, "missing admission failure");
        thread::yield_now();
    }
    service.shutdown().unwrap();
}
#[test]
fn queue_capacity_request_bytes_and_shutdown_return_typed_errors() {
    let (sender, _receiver) = mpsc::sync_channel(1);
    let inner = Arc::new(Inner {
        sender,
        accepting: Mutex::new(true),
        cancelled: Arc::new(AtomicBool::new(false)),
    });
    let handle = Handle { inner };
    let _first = handle.try_progress("synthetic").unwrap();
    assert!(matches!(handle.try_progress("synthetic"), Err(Error::Busy)));
    assert!(matches!(
        handle.try_progress(&"x".repeat(REQUEST_BYTES + 1)),
        Err(Error::RequestTooLarge { .. })
    ));
    let mut mismatched = request(Path::new("/synthetic/path"), "synthetic");
    mismatched.observation.path = "/synthetic/other".into();
    assert!(matches!(
        handle.try_import(mismatched),
        Err(Error::InvalidRequest(_))
    ));
    *handle.inner.accepting.lock().unwrap() = false;
    assert!(matches!(handle.try_retry("synthetic"), Err(Error::Stopped)));
    let text = diagnostic("ą".repeat(DETAIL_BYTES));
    assert!(text.len() <= DETAIL_BYTES);
    assert!(text.ends_with("[diagnostic truncated]"));
}
#[test]
fn shutdown_replies_to_accepted_import_instead_of_losing_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.jsonl");
    std::fs::write(&path, raw()).unwrap();
    let service = owner(&dir.path().join("archive"));
    let reply = service
        .handle()
        .try_import(request(&path, "synthetic-stop"))
        .unwrap();
    service.shutdown().unwrap();
    assert!(matches!(
        reply.recv_timeout(WAIT),
        Ok(Ok(_)) | Ok(Err(Error::Stopped))
    ));
}
