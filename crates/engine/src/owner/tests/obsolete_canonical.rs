use super::*;

fn fixture() -> (
    tempfile::TempDir,
    Actor,
    Receiver<Message>,
    StoredSource,
    PathBuf,
) {
    let temp = tempfile::tempdir().unwrap();
    let main = temp.path().join("session.jsonl");
    std::fs::write(&main, b"{\"type\":\"user\",\"uuid\":\"root\",\"sessionId\":\"session\",\"promptSource\":\"typed\",\"message\":{\"content\":\"Initial\"}}\n").unwrap();
    let (sender, receiver) = mpsc::sync_channel(COMMANDS);
    let inner = Arc::new(Inner {
        sender,
        accepting: Mutex::new(true),
        cancelled: Arc::new(AtomicBool::new(false)),
    });
    let mut actor = Actor::new(Writer::open(&temp.path().join("archive")).unwrap(), inner);
    let input = actor
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
    let input = actor
        .writer
        .stored_source(&input.input.id)
        .unwrap()
        .unwrap();
    actor.active = Some(Active::Begin(input.clone()));
    for _ in 0..1000 {
        if matches!(
            &actor.active,
            Some(Active::Canonical(_, _, CanonicalPosition::Audit(_), _))
        ) {
            return (temp, actor, receiver, input, main);
        }
        advance(&mut actor, &receiver);
    }
    panic!("initial source did not reach final audit");
}
fn advance(actor: &mut Actor, receiver: &Receiver<Message>) {
    actor.tick().unwrap();
    if actor.compute.is_some() || actor.canonical.is_some() {
        actor
            .message(receiver.recv_timeout(Duration::from_secs(10)).unwrap())
            .unwrap();
    }
}
fn append(actor: &mut Actor, input: &StoredSource, main: &Path) -> StoredSource {
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(main)
        .unwrap()
        .write_all(b"{\"type\":\"last-prompt\",\"sessionId\":\"session\",\"leafUuid\":\"root\"}\n")
        .unwrap();
    actor
        .writer
        .acquire_claude_main(
            main,
            "synthetic-machine",
            "synthetic-host",
            "append",
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
    let current = actor
        .writer
        .stored_source(&input.input.id)
        .unwrap()
        .unwrap();
    assert!(current.revision > input.revision);
    let context = parser_context_json(&current).unwrap();
    actor
        .writer
        .begin_parser(&Basis {
            input_pk: current.input.input_pk,
            chain: current.input.chain,
            parser_version: parser_version(current.format),
            digest: parser_identity(&current),
            acquired_upto: current.input.length,
            companions_json: Some(&context),
            excisions_seen: 0,
        })
        .unwrap();
    current
}
#[test]
fn append_during_finalization_discards_only_obsolete_audit_and_restarts_cleanly() {
    let (temp, mut actor, receiver, input, main) = fixture();
    let old_head = actor
        .writer
        .parser_head(input.input.input_pk, input.input.chain)
        .unwrap()
        .unwrap();
    let current = append(&mut actor, &input, &main);
    // The already selected audit still carries the old input, but its next
    // pinned snapshot sees contiguous acquisition growth under the same token.
    actor.tick().unwrap();
    let event = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(matches!(
        &event,
        Message::CanonicalComputed(Err(Error::Store(StoreError::StaleParserState)))
    ));
    actor.message(event).unwrap();
    assert!(matches!(&actor.active, Some(Active::Begin(v)) if v.revision == current.revision));
    assert!(
        actor
            .writer
            .parser_failure(&old_head.token)
            .unwrap()
            .is_none()
    );
    // Stop immediately after discard. Ordinary owner recovery must consume the
    // appended record and certify all accounting, without the former failure.
    drop(actor);
    let service = owner(&temp.path().join("archive"));
    let resumed = until_caught_up(&service.handle(), &input.input.id);
    assert!(resumed.failure.is_none(), "{:?}", resumed.failure);
    let head = resumed.parser.unwrap();
    assert_eq!(head.parsed_upto, current.input.length);
    assert_eq!(head.counts.used, 2);
    service.shutdown().unwrap();
}
#[test]
fn stale_parser_state_and_changed_source_corruption_are_reported() {
    for changed in [false, true] {
        let (_temp, mut actor, receiver, input, main) = fixture();
        if changed {
            append(&mut actor, &input, &main);
        }
        actor.tick().unwrap();
        let _actual = receiver.recv_timeout(Duration::from_secs(10)).unwrap();
        let error = if changed {
            StoreError::Invariant("synthetic canonical corruption".into())
        } else {
            StoreError::StaleParserState
        };
        actor
            .message(Message::CanonicalComputed(Err(error.into())))
            .unwrap();
        let head = actor
            .writer
            .parser_head(input.input.input_pk, input.input.chain)
            .unwrap()
            .unwrap();
        assert!(actor.writer.parser_failure(&head.token).unwrap().is_some());
        assert!(!matches!(actor.active, Some(Active::Begin(_))));
    }
}
