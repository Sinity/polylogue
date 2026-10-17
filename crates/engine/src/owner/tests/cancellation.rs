use super::*;

fn fixture() -> (tempfile::TempDir, Actor, RetainedSource, Head) {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    let proof = writer
        .blob_capture()
        .unwrap()
        .capture(std::io::Cursor::new(raw()), &AtomicBool::new(false))
        .unwrap();
    let hash = writer.retain_captured_blob(proof).unwrap();
    let request = request(
        Path::new("/synthetic/input.jsonl"),
        "synthetic-cancellation",
    );
    let input = writer
        .retain_immutable_input(
            hash,
            request.format,
            &[],
            &request.observation.borrowed(),
            request.metadata_allowance,
        )
        .unwrap();
    let source = writer.retained_source(&input.id).unwrap().unwrap();
    let context = parser_context_json(&source).unwrap();
    let head = writer
        .begin_parser(&Basis {
            input_pk: source.input.input_pk,
            chain: source.input.chain,
            parser_version: parser_version(source.format),
            digest: parser_basis(&source),
            acquired_upto: source.input.length,
            companions_json: Some(&context),
            excisions_seen: 0,
        })
        .unwrap();
    let (sender, _receiver) = mpsc::sync_channel(COMMANDS);
    let inner = Arc::new(Inner {
        sender,
        accepting: Mutex::new(true),
        cancelled: Arc::new(AtomicBool::new(false)),
    });
    (temp, Actor::new(writer, inner), source, head)
}

fn source_job(actor: &mut Actor, input: RetainedSource, head: Head) {
    actor.compute = Some(ComputeJob {
        input,
        head,
        join: thread::spawn(|| {}),
    });
}

#[test]
fn stop_between_loop_observation_and_completion_does_not_poison_restart() {
    let (temp, mut actor, input, head) = fixture();
    let before = actor.writer.status().unwrap().write_counter;
    source_job(&mut actor, input.clone(), head.clone());
    // Exactly the recv_timeout window: the loop observed no stop, then the
    // worker observed cancellation and queued its completion before dispatch.
    actor.observe_stop();
    assert!(!actor.stopping);
    actor.inner.cancelled.store(true, Ordering::Release);
    actor
        .message(Message::Computed(Err(StoreError::Cancelled.into())))
        .unwrap();
    assert!(actor.stopping);
    assert!(actor.compute.is_none());
    assert!(actor.writer.parser_failure(&head.token).unwrap().is_none());
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
    drop(actor);

    let service = owner(&temp.path().join("archive"));
    let resumed = until_settled(&service.handle(), &input.input.id);
    assert!(resumed.failure.is_none());
    assert_eq!(resumed.parser.unwrap().parsed_upto, input.input.length);
    service.shutdown().unwrap();
}

#[test]
fn typed_source_cancellation_after_dispatch_is_not_a_parser_defect() {
    for error in [
        Error::Store(StoreError::Cancelled),
        Error::Source(source_turn::Error::Store(StoreError::Cancelled)),
    ] {
        let (_temp, mut actor, input, head) = fixture();
        let before = actor.writer.status().unwrap().write_counter;
        source_job(&mut actor, input, head.clone());
        assert!(!actor.inner.cancelled.load(Ordering::Acquire));
        actor.message(Message::Computed(Err(error))).unwrap();
        assert!(actor.stopping);
        assert!(actor.writer.parser_failure(&head.token).unwrap().is_none());
        assert_eq!(actor.writer.status().unwrap().write_counter, before);
    }
}

#[test]
fn typed_canonical_cancellation_is_not_a_parser_defect() {
    for error in [
        canonical_message::Error::Cancelled,
        canonical_message::Error::Store(StoreError::Cancelled),
    ] {
        let (_temp, mut actor, input, head) = fixture();
        let before = actor.writer.status().unwrap().write_counter;
        let reader = actor.take_canonical_reader().unwrap();
        actor.canonical = Some(CanonicalJob {
            input,
            head: head.clone(),
            position: CanonicalPosition::first(InputFormat::CodexRollout),
            handle: None,
            join: thread::spawn(move || reader),
        });
        actor
            .message(Message::CanonicalComputed(Err(Error::Canonical(error))))
            .unwrap();
        assert!(actor.stopping);
        assert!(actor.canonical.is_none());
        assert!(actor.writer.parser_failure(&head.token).unwrap().is_none());
        assert_eq!(actor.writer.status().unwrap().write_counter, before);
    }
}

#[test]
fn reused_canonical_reader_observes_later_value_writes_and_sealing() {
    use polylogue_store::values::{Codec, State};
    let (_temp, mut actor, _input, _head) = fixture();
    let value = actor.writer.begin_value(Codec::Utf8, 3).unwrap();
    let reader = actor.take_canonical_reader().unwrap();
    assert_eq!(reader.value_head(&value.id).unwrap().appended, 0);
    actor.canonical_reader = Some(reader);

    actor
        .writer
        .append_value_chunk(&value.id, 0, b"new", 3)
        .unwrap();
    let reader = actor.take_canonical_reader().unwrap();
    assert_eq!(reader.value_head(&value.id).unwrap().appended, 3);
    let mut proof = reader.validate_value(&value.id).unwrap();
    while !proof.progress().complete {
        proof.advance(&reader, 3).unwrap();
    }
    actor.canonical_reader = Some(reader);
    actor.writer.seal_value(&proof).unwrap();

    let reader = actor.take_canonical_reader().unwrap();
    assert_eq!(reader.value_head(&value.id).unwrap().state, State::Ready);
    assert_eq!(reader.value_range(&value.id, 0, 3).unwrap().bytes, b"new");
}

#[test]
fn cancelled_capture_replies_stopped_without_publishing() {
    let (_temp, mut actor, _input, _head) = fixture();
    let before = actor.writer.status().unwrap().write_counter;
    let request = request(
        Path::new("/synthetic/next.jsonl"),
        "synthetic-stopped-capture",
    );
    let (reply, received) = mpsc::sync_channel(1);
    actor.capture = Some(CaptureJob {
        expected: None,
        format: request.format,
        context: request.context,
        observation: request.observation,
        allowance: request.metadata_allowance,
        reply: ImportReply::Ordinary(reply),
        join: thread::spawn(|| {}),
    });
    actor
        .message(Message::Captured(Err(StoreError::Cancelled.into())))
        .unwrap();
    assert!(matches!(received.try_recv(), Ok(Err(Error::Stopped))));
    assert!(actor.stopping);
    assert!(actor.capture.is_none());
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
}

#[test]
fn genuine_worker_failure_still_records_a_parser_defect() {
    let (_temp, mut actor, input, head) = fixture();
    source_job(&mut actor, input, head.clone());
    actor
        .message(Message::Computed(Err(StoreError::Invariant(
            "synthetic corrupt source".into(),
        )
        .into())))
        .unwrap();
    assert!(!actor.stopping);
    assert_eq!(
        actor.writer.parser_failure(&head.token).unwrap().as_deref(),
        Some("parser state invariant: synthetic corrupt source")
    );
}

#[test]
fn prepared_physical_history_is_not_selected_after_owner_stop_and_restart_can_retry() {
    let (temp, mut actor, _input, _head) = fixture();
    let path = temp.path().join("archive/archive.db");
    // A synthetic owner database is a valid physical SQLite input; its schema
    // is deliberately unsupported by the conversation decoder. No engine SQL.
    let prepared = actor
        .writer
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let before = actor.writer.status().unwrap().write_counter;
    let request = request(&path, "stopped-physical-history");
    let observation = request.observation.clone();
    let (reply, received) = mpsc::sync_channel(1);
    actor.capture = Some(CaptureJob {
        expected: None,
        format: InputFormat::CodexHistorySqlite,
        context: vec![],
        observation: observation.clone(),
        allowance: 65536,
        reply: ImportReply::Ordinary(reply),
        join: thread::spawn(|| {}),
    });
    actor.inner.cancelled.store(true, Ordering::Release);
    actor
        .message(Message::Captured(Ok(PreparedCapture::History(prepared))))
        .unwrap();
    assert!(matches!(received.try_recv(), Ok(Err(Error::Stopped))));
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
    drop(actor);
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    let mut after = String::new();
    while let Some(source) = writer.next_retained_source(&after).unwrap() {
        assert_ne!(source.format, InputFormat::CodexHistorySqlite);
        after = source.input.id;
    }
    let p = writer
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let hash = p.hash();
    let n = p.length();
    let input = writer
        .retain_history_pages(p, &observation.borrowed(), 65536)
        .unwrap();
    let bytes = writer
        .read_handle()
        .unwrap()
        .read_source_range(&input.id, 1, hash, 0, n)
        .unwrap();
    assert_eq!(blake3::hash(&bytes).as_bytes(), hash.bytes());
}
