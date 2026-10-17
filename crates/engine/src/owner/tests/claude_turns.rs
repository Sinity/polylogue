use super::*;

fn new_actor(root: &Path) -> (Actor, Receiver<Message>) {
    let (sender, receiver) = mpsc::sync_channel(COMMANDS);
    let inner = Arc::new(Inner {
        sender,
        accepting: Mutex::new(true),
        cancelled: Arc::new(AtomicBool::new(false)),
    });
    (Actor::new(Writer::open(root).unwrap(), inner), receiver)
}
fn start(actor: &mut Actor, path: &Path, id: &str) -> Receiver<Result<ConfiguredImport>> {
    let mut req = request(path, id);
    req.format = InputFormat::ClaudeCodeJsonl;
    req.context = vec![
        "claude-main".into(),
        path.file_stem().unwrap().to_str().unwrap().into(),
    ];
    let (reply, rx) = mpsc::sync_channel(1);
    actor.message(Message::Claude(req, 65536, reply)).unwrap();
    rx
}
fn finish(actor: &mut Actor, events: &Receiver<Message>) -> usize {
    let mut turns = 0;
    loop {
        let event = events.recv_timeout(Duration::from_secs(10)).unwrap();
        let done = matches!(&event, Message::TurnCaptured(..));
        if let Message::CaptureYield(progress, ..) = &event {
            assert!(progress.charged_bytes <= 65536);
            turns += 1;
        }
        actor.message(event).unwrap();
        if done {
            return turns;
        }
        actor.advance_capture_turn().unwrap();
    }
}
#[test]
fn claude_capture_reuses_quanta_and_serves_a_regular_capture_before_selecting_the_append() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("session.jsonl");
    let initial=b"{\"type\":\"user\",\"uuid\":\"earlier\",\"sessionId\":\"synthetic\",\"message\":{\"role\":\"user\",\"content\":\"Earlier independent text\"}}\n";
    std::fs::write(&main, initial).unwrap();
    let (mut actor, events) = new_actor(&temp.path().join("archive"));
    let source = actor
        .writer
        .acquire_claude_main(
            &main,
            "synthetic-machine",
            "synthetic-host",
            "initial",
            0,
            &AtomicBool::new(false),
        )
        .unwrap();
    let reader = actor.writer.read_handle().unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(&main)
        .unwrap()
        .write_all(&vec![b'\n'; 262144])
        .unwrap();
    let pending = start(&mut actor, &main, "append");
    let event = events.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(matches!(&event, Message::CaptureYield(..)));
    let regular = temp.path().join("rollout.jsonl");
    std::fs::write(&regular, raw()).unwrap();
    let (reply, regular_reply) = mpsc::sync_channel(1);
    actor
        .message(Message::Import(
            request(&regular, "regular"),
            None,
            ImportReply::Ordinary(reply),
        ))
        .unwrap();
    actor.message(event).unwrap();
    actor.advance_capture_turn().unwrap();
    assert!(actor.capture.is_some());
    assert!(actor.turn_capture.as_ref().unwrap().grant.is_some());
    actor
        .message(events.recv_timeout(Duration::from_secs(10)).unwrap())
        .unwrap();
    let imported = regular_reply
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let retained = reader.source(&imported.id).unwrap().unwrap();
    assert_eq!(
        reader
            .read_source_range(&imported.id, 1, retained.binding, 0, retained.input.length)
            .unwrap(),
        raw()
    );
    assert!(matches!(pending.try_recv(), Err(TryRecvError::Empty)));
    assert_eq!(
        reader.source(&source.input.id).unwrap().unwrap().revision,
        1
    );
    assert_eq!(
        reader
            .read_source_range(&source.input.id, 1, source.binding, 0, initial.len() as u64)
            .unwrap(),
        initial
    );
    actor.advance_capture_turn().unwrap();
    assert!(finish(&mut actor, &events) > 2);
    let appended = pending
        .recv_timeout(Duration::from_secs(10))
        .unwrap()
        .unwrap();
    assert!(appended.acquired);
    assert_eq!(appended.input.id, source.input.id);
    assert_eq!(appended.input.length, initial.len() as u64 + 262144);
    let before = actor.writer.status().unwrap().write_counter;
    let unchanged = start(&mut actor, &main, "unchanged");
    assert!(finish(&mut actor, &events) > 2);
    assert!(
        !unchanged
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap()
            .acquired
    );
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
}
#[test]
fn claude_parked_cancellation_keeps_acknowledged_prefix_and_restarts_without_a_lifetime_quota() {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("session.jsonl");
    std::fs::write(&main, b"\n").unwrap();
    let archive = temp.path().join("archive");
    let (mut actor, events) = new_actor(&archive);
    let source = actor
        .writer
        .acquire_claude_main(
            &main,
            "synthetic-machine",
            "synthetic-host",
            "initial",
            0,
            &AtomicBool::new(false),
        )
        .unwrap();
    std::fs::write(&main, vec![b'\n'; 196608]).unwrap();
    let pending = start(&mut actor, &main, "cancelled");
    let first = events.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(matches!(&first, Message::CaptureYield(..)));
    actor.message(first).unwrap();
    let before = actor.writer.status().unwrap().write_counter;
    actor.begin_stop();
    actor
        .message(events.recv_timeout(Duration::from_secs(10)).unwrap())
        .unwrap();
    assert!(matches!(
        pending.recv_timeout(Duration::from_secs(10)).unwrap(),
        Err(Error::Stopped)
    ));
    assert_eq!(actor.writer.status().unwrap().write_counter, before);
    assert_eq!(
        actor
            .writer
            .retained_source(&source.input.id)
            .unwrap()
            .unwrap()
            .input
            .length,
        1
    );
    drop(actor);
    let (mut resumed, events) = new_actor(&archive);
    let completed = start(&mut resumed, &main, "resumed");
    assert!(finish(&mut resumed, &events) > 2);
    assert_eq!(
        completed
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap()
            .input
            .length,
        196608
    );
}
