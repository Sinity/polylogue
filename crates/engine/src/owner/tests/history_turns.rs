use super::*;

fn actor(root: &Path) -> (Actor, Receiver<Message>) {
    let (sender, receiver) = mpsc::sync_channel(COMMANDS);
    let inner = Arc::new(Inner {
        sender,
        accepting: Mutex::new(true),
        cancelled: Arc::new(AtomicBool::new(false)),
    });
    let mut actor = Actor::new(Writer::open(root).unwrap(), inner);
    actor.history_budget = polylogue_store::history_pages::turns::Budget {
        bytes: 65536,
        checkpoints: 2,
    };
    (actor, receiver)
}
fn physical(path: &Path, id: &str) -> ImportRequest {
    let mut value = request(path, id);
    value.format = InputFormat::CodexHistorySqlite;
    value
}
#[test]
fn parked_history_capture_shares_turns_with_regular_capture_and_reads() {
    let temp = tempfile::tempdir().unwrap();
    let mut provider = Writer::open(&temp.path().join("provider")).unwrap();
    provider.put_file_blob(&vec![b'x'; 131072]).unwrap();
    let path = temp.path().join("provider/archive.db");
    let baseline = provider
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let expected_hash = baseline.hash();
    drop(baseline);
    let (mut actor, events) = actor(&temp.path().join("archive"));
    let reader = actor.writer.read_handle().unwrap();
    let (reply, history) = mpsc::sync_channel(1);
    actor
        .message(Message::Import(
            physical(&path, "physical-turns"),
            None,
            ImportReply::Configured(reply),
        ))
        .unwrap();
    let first = events.recv_timeout(super::WAIT).unwrap();
    assert!(matches!(&first, Message::CaptureYield(..)));
    // A real configured input, not a counter.
    let home = temp.path().join("home");
    std::fs::create_dir_all(home.join("sessions")).unwrap();
    let regular_path = home.join("sessions/rollout-between-turns.jsonl");
    let raw = b"{\"type\":\"session_meta\",\"payload\":{\"id\":\"synthetic\"}}\n{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"regular work between physical quanta\"}]}}\n";
    std::fs::write(&regular_path, raw).unwrap();
    let config = polylogue_store::configured_codex::Config {
        paginated_history: None,
        paginated_interval_seconds: 300,
        home,
        machine: "synthetic-machine".into(),
        host: "synthetic-host".into(),
        rollouts: "sessions".into(),
        catalog: "state_5.sqlite".into(),
        history: "history.jsonl".into(),
    };
    let candidate = polylogue_store::configured_codex::discover(&config, &AtomicBool::new(false))
        .unwrap()
        .candidates
        .into_iter()
        .next()
        .unwrap();
    let mut regular = request(&regular_path, "regular-between-turns");
    regular.format = candidate.format();
    let (reply, regular_reply) = mpsc::sync_channel(1);
    actor
        .message(Message::Import(
            regular,
            Some((
                candidate,
                BlobHash::from_bytes(*blake3::hash(raw).as_bytes()),
                raw.len() as u64,
            )),
            ImportReply::Configured(reply),
        ))
        .unwrap();
    assert!(actor.waiting_capture.is_some());
    actor.message(first).unwrap();
    actor.advance_capture_turn().unwrap();
    assert!(actor.capture.is_some());
    assert!(actor.turn_capture.as_ref().unwrap().grant.is_some());
    actor
        .message(events.recv_timeout(super::WAIT).unwrap())
        .unwrap();
    let regular = regular_reply
        .recv_timeout(super::WAIT)
        .unwrap()
        .unwrap();
    let source = reader.source(&regular.input.id).unwrap().unwrap();
    assert_eq!(
        reader
            .read_source_range(&regular.input.id, 1, source.binding, 0, raw.len() as u64)
            .unwrap(),
        raw
    );
    assert!(matches!(history.try_recv(), Err(TryRecvError::Empty)));
    let physical_id = polylogue_model::InputId::imported(
        &expected_hash,
        InputFormat::CodexHistorySqlite.name(),
        &[],
    )
    .to_string();
    assert!(reader.source(&physical_id).unwrap().is_none());
    // Later provider commits do not change the pinned read view.
    provider
        .put_file_blob(b"later provider generation")
        .unwrap();
    let mut turns = 1;
    actor.advance_capture_turn().unwrap();
    loop {
        let event = events.recv_timeout(super::WAIT).unwrap();
        let done = matches!(&event, Message::TurnCaptured(..));
        if let Message::CaptureYield(p, ..) = &event {
            assert!(p.charged_bytes <= 65536 && p.checkpoints <= 2);
            turns += 1;
        }
        actor.message(event).unwrap();
        if done {
            break;
        }
        actor.advance_capture_turn().unwrap();
    }
    assert!(turns > 2);
    let result = history
        .recv_timeout(super::WAIT)
        .unwrap()
        .unwrap();
    assert_eq!(result.input.id, physical_id);
    let bytes = reader
        .read_source_range(&result.input.id, 1, expected_hash, 0, result.input.length)
        .unwrap();
    assert_eq!(blake3::hash(&bytes).as_bytes(), expected_hash.bytes());
}

#[test]
fn owner_stop_during_a_parked_turn_selects_no_prefix_and_fresh_owner_can_retry() {
    let temp = tempfile::tempdir().unwrap();
    let mut provider = Writer::open(&temp.path().join("provider")).unwrap();
    let path = temp.path().join("provider/archive.db");
    let archive = temp.path().join("archive");
    let (mut actor, events) = actor(&archive);
    let before = actor.writer.status().unwrap().write_counter;
    let (reply, result) = mpsc::sync_channel(1);
    actor
        .message(Message::Import(
            physical(&path, "cancel-parked"),
            None,
            ImportReply::Configured(reply),
        ))
        .unwrap();
    let event = events.recv_timeout(super::WAIT).unwrap();
    assert!(matches!(&event, Message::CaptureYield(..)));
    actor.message(event).unwrap();
    actor.begin_stop();
    actor
        .message(events.recv_timeout(super::WAIT).unwrap())
        .unwrap();
    assert!(matches!(
        result.recv_timeout(super::WAIT).unwrap(),
        Err(Error::Stopped)
    ));
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
    drop(actor);
    // A lost transaction cannot resume from a byte cursor; the new owner
    // captures a complete new generation.
    provider
        .put_file_blob(b"new committed generation after cancellation")
        .unwrap();
    let expected = provider
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let expected_id = polylogue_model::InputId::imported(
        &expected.hash(),
        InputFormat::CodexHistorySqlite.name(),
        &[],
    )
    .to_string();
    drop(expected);
    let owner = owner(&archive);
    let retained = owner
        .handle()
        .try_history_import(physical(&path, "cancel-parked"))
        .unwrap()
        .recv_timeout(super::WAIT)
        .unwrap()
        .unwrap();
    assert!(retained.input.length > 0);
    assert_eq!(retained.input.id, expected_id);
    owner.shutdown().unwrap();
    drop(provider);
}
