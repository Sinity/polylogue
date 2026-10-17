use super::*;

fn fixture() -> (tempfile::TempDir, Actor, Vec<RetainedSource>) {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    let mut inputs = Vec::new();
    for n in 0..4 {
        let bytes = [raw(), &vec![b'\n'; n]].concat();
        let proof = writer
            .blob_capture()
            .unwrap()
            .capture(std::io::Cursor::new(bytes), &AtomicBool::new(false))
            .unwrap();
        let hash = writer.retain_captured_blob(proof).unwrap();
        let observation = request(Path::new("/synthetic/input.jsonl"), &format!("input-{n}"));
        let input = writer
            .retain_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &observation.observation.borrowed(),
                4096,
            )
            .unwrap();
        inputs.push(writer.retained_source(&input.id).unwrap().unwrap());
    }
    inputs.sort_by(|a, b| a.input.id.cmp(&b.input.id));
    let (sender, _receiver) = mpsc::sync_channel(COMMANDS);
    let actor = Actor::new(
        writer,
        Arc::new(Inner {
            sender,
            accepting: Mutex::new(true),
            cancelled: Arc::new(AtomicBool::new(false)),
        }),
    );
    (temp, actor, inputs)
}

#[test]
fn retried_input_before_scan_cursor_runs_without_waiting_for_catalog_wrap() {
    let (_temp, mut actor, inputs) = fixture();
    actor.after = inputs[1].input.id.clone();
    actor.retry(&inputs[0].input.id).unwrap();
    actor.tick().unwrap();
    assert_eq!(actor.active_id(), Some(inputs[0].input.id.as_str()));
    assert!(
        actor
            .writer
            .parser_head(inputs[0].input.input_pk, inputs[0].input.chain)
            .unwrap()
            .is_some()
    );
    assert_eq!(actor.after, inputs[1].input.id);
}

#[test]
fn captured_input_runs_after_current_input_without_rewinding_background_scan() {
    let (_temp, mut actor, inputs) = fixture();
    actor.after = inputs[1].input.id.clone();
    actor.active = Some(Active::Begin(inputs[1].clone()));
    let bytes = [raw(), b"\n\n\n\n\n"].concat();
    let proof = actor
        .writer
        .blob_capture()
        .unwrap()
        .capture(std::io::Cursor::new(bytes), &AtomicBool::new(false))
        .unwrap();
    let req = request(Path::new("/synthetic/late.jsonl"), "captured-late");
    let (reply, receiver) = mpsc::sync_channel(1);
    actor
        .finish_capture(
            CaptureJob {
                expected: None,
                format: req.format,
                context: req.context,
                observation: req.observation,
                allowance: req.metadata_allowance,
                reply: ImportReply::Ordinary(reply),
                join: thread::spawn(|| {}),
            },
            Ok(PreparedCapture::Blob(proof)),
        )
        .unwrap();
    let captured = receiver.recv().unwrap().unwrap();
    actor.tick().unwrap();
    assert_eq!(actor.active_id(), Some(inputs[1].input.id.as_str()));
    // The current input finishes before the next selection. Its continuation
    // is never dropped merely to prioritize a newer retained acquisition.
    actor.active = None;
    actor.tick().unwrap();
    assert_eq!(actor.active_id(), Some(captured.id.as_str()));
    assert_eq!(actor.after, inputs[1].input.id);
}

#[test]
fn repeated_recent_changes_alternate_with_background_and_do_not_skip_inputs() {
    let (_temp, mut actor, inputs) = fixture();
    actor.retry(&inputs[3].input.id).unwrap();
    actor.tick().unwrap();
    assert_eq!(actor.active_id(), Some(inputs[3].input.id.as_str()));
    for background in &inputs {
        actor.active = None; // each selected input completed
        actor.retry(&inputs[3].input.id).unwrap();
        actor.tick().unwrap();
        assert_eq!(actor.active_id(), Some(background.input.id.as_str()));
        assert_eq!(actor.after, background.input.id);
        actor.active = None;
        actor.tick().unwrap();
        assert_eq!(actor.active_id(), Some(inputs[3].input.id.as_str()));
        assert_eq!(actor.after, background.input.id);
    }
}

#[test]
fn coalesced_change_hints_are_recovered_by_the_durable_scan_after_restart() {
    let (temp, mut actor, inputs) = fixture();
    for input in &inputs {
        actor.retry(&input.input.id).unwrap();
    }
    drop(actor);
    let writer = Writer::open(&temp.path().join("archive")).unwrap();
    let (sender, _receiver) = mpsc::sync_channel(COMMANDS);
    let mut restarted = Actor::new(
        writer,
        Arc::new(Inner {
            sender,
            accepting: Mutex::new(true),
            cancelled: Arc::new(AtomicBool::new(false)),
        }),
    );
    for input in &inputs {
        restarted.tick().unwrap();
        assert_eq!(restarted.active_id(), Some(input.input.id.as_str()));
        restarted.active = None;
    }
}
