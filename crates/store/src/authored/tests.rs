use super::*;
fn id(n: u32) -> String {
    format!("01900000-0000-7000-8000-{n:012}")
}
fn request(n: u32) -> WriteNote {
    WriteNote {
        request_id: id(n),
        author: "codex:synthetic-writer".into(),
        machine: "synthetic-machine".into(),
        created_at: Some(1000 - n as i64),
        kind: Some(NoteKind::Correction),
        value: format!("revision {n}"),
        targets: Some(vec![Target {
            reference: "block:claude-code:neutral-session:answer:0".into(),
            seen_hash: Some(BlobHash::of(b"abc")),
        }]),
        relations: vec![],
    }
}
fn revise(n: u32, reference: String) -> WriteNote {
    WriteNote {
        kind: None,
        targets: None,
        relations: vec![Relation {
            kind: "revises".into(),
            other: reference,
        }],
        ..request(n)
    }
}
fn withdraw(n: u32, reference: String) -> WriteNote {
    WriteNote {
        kind: Some(NoteKind::Withdrawal),
        targets: None,
        relations: vec![Relation {
            kind: "withdraws".into(),
            other: reference,
        }],
        ..request(n)
    }
}
#[test]
fn authored_history_retry_stale_revision_and_recursive_withdrawal_survive_index_loss() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let mut writer = Writer::open(&archive).unwrap();
    let r1 = writer.write_note(&request(1)).unwrap();
    let root = format!("note:{}", r1.root);
    let before = writer.read_handle().unwrap().authored_snapshot().unwrap();
    let counter = writer.status().unwrap().write_counter;
    assert_eq!(writer.write_note(&request(1)).unwrap(), r1);
    assert_eq!(writer.status().unwrap().write_counter, counter);
    let mut changed = request(1);
    changed.targets.as_mut().unwrap()[0].seen_hash = Some(BlobHash::of(b"substituted"));
    assert!(matches!(
        writer.write_note(&changed),
        Err(Error::ClientRequestConflict { .. })
    ));
    let r2 = writer.write_note(&revise(2, r1.reference())).unwrap();
    assert_eq!(r2.targets, r1.targets);
    assert!(
        r2.created_at < r1.created_at,
        "clock order opposes revision order"
    );
    let current = writer.read_handle().unwrap().authored_snapshot().unwrap();
    assert_eq!(current.family(&root).unwrap().current, vec![r2.reference()]);
    let err = writer.write_note(&revise(3, r1.reference())).unwrap_err();
    assert!(matches!(
        err,
        Error::Authored(Failure {
            kind: ErrorKind::Conflict,
            ..
        })
    ));
    assert!(err.to_string().contains(&r2.id));
    assert_eq!(
        before.family(&root).unwrap().current,
        vec![r1.reference()],
        "pinned archive read never moves to the new head"
    );
    let w2 = writer.write_note(&withdraw(4, r2.reference())).unwrap();
    let view = writer.read_handle().unwrap().authored_snapshot().unwrap();
    assert_eq!(view.family(&root).unwrap().current, vec![r1.reference()]);
    assert!(view.family(&r2.reference()).unwrap().shown[0].withdrawn);
    let restore = writer.write_note(&withdraw(5, w2.reference())).unwrap();
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .authored_snapshot()
            .unwrap()
            .family(&root)
            .unwrap()
            .current,
        vec![r2.reference()]
    );
    let wr = writer.write_note(&withdraw(6, root.clone())).unwrap();
    assert!(
        writer
            .read_handle()
            .unwrap()
            .authored_snapshot()
            .unwrap()
            .family(&root)
            .unwrap()
            .current
            .is_empty()
    );
    assert!(
        writer
            .read_handle()
            .unwrap()
            .authored_snapshot()
            .unwrap()
            .family(&r1.reference())
            .unwrap()
            .shown[0]
            .withdrawn
    );
    writer.write_note(&withdraw(7, wr.reference())).unwrap();
    let expected = writer
        .read_handle()
        .unwrap()
        .authored_snapshot()
        .unwrap()
        .family(&root)
        .unwrap();
    assert_eq!(expected.current, vec![r2.reference()]);
    assert_eq!(expected.history.len(), 2);
    assert_eq!(expected.withdrawal_trail.len(), 4);
    assert!(
        expected
            .withdrawal_trail
            .iter()
            .any(|w| w.revision.id == restore.id)
    );
    drop(before);
    drop(current);
    drop(view);
    drop(writer);
    for name in ["index.db", "index.db-wal", "index.db-shm"] {
        let p = archive.join(name);
        if p.exists() {
            std::fs::remove_file(p).unwrap();
        }
    }
    let mut writer = Writer::open(&archive).unwrap();
    let actual = writer
        .read_handle()
        .unwrap()
        .authored_snapshot()
        .unwrap()
        .family(&root)
        .unwrap();
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    assert_eq!(
        writer.write_note(&request(1)).unwrap(),
        r1,
        "old request retry returns original effect after later revisions/rebuild"
    );
}
#[test]
fn authored_replay_normalizes_sets_and_rejects_unsupported_intent_without_rows() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(temp.path()).unwrap();
    let mut a = request(20);
    a.targets.as_mut().unwrap().push(Target {
        reference: "message:claude-code:neutral-session:other".into(),
        seen_hash: None,
    });
    let first = writer.write_note(&a).unwrap();
    a.targets.as_mut().unwrap().reverse();
    assert_eq!(writer.write_note(&a).unwrap(), first);
    let mut unsupported = serde_json::to_value(request(21)).unwrap();
    unsupported["intent"] = serde_json::json!("authored_return");
    assert!(serde_json::from_value::<WriteNote>(unsupported).is_err());
    let mut conflict = a.clone();
    conflict.value = "changed final form".into();
    assert!(matches!(
        writer.write_note(&conflict),
        Err(Error::ClientRequestConflict { .. })
    ));
    let snapshot = writer.read_handle().unwrap().authored_snapshot().unwrap();
    assert_eq!(
        snapshot
            .family(&format!("note:{}", id(20)))
            .unwrap()
            .history
            .len(),
        1
    );
}

#[test]
fn authored_controls_require_committed_parents_and_refusals_are_atomic() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(temp.path()).unwrap();
    let initial = writer.status().unwrap().write_counter;
    for n in [20, 21] {
        for relation in ["revises", "withdraws"] {
            let parent = if n == 20 { id(n) } else { id(99) };
            let action = if relation == "revises" {
                revise(n, format!("note:{parent}@{parent}"))
            } else {
                withdraw(n, format!("note:{parent}@{parent}"))
            };
            assert!(matches!(
                writer.write_note(&action),
                Err(Error::Authored(Failure {
                    kind: ErrorKind::NotFound,
                    ..
                }))
            ));
            assert_eq!(writer.status().unwrap().write_counter, initial);
        }
    }
    // A refused action reserves neither the request nor a revision.
    let first = writer.write_note(&request(20)).unwrap();
    let mut second = revise(21, first.reference());
    second.targets = Some(vec![Target {
        reference: "message:other".into(),
        seen_hash: None,
    }]);
    let second = writer.write_note(&second).unwrap();
    assert!(second.targets[0].seen_hash.is_none());
    let snapshot = writer.read_handle().unwrap().authored_snapshot().unwrap();
    let links = snapshot.backlinks(&first.targets[0].reference).unwrap();
    assert_eq!(
        links[0].matching_target_revisions.as_ref().unwrap(),
        &[first.reference()]
    );
    assert!(
        links[0]
            .current_target_revisions
            .as_ref()
            .unwrap()
            .is_empty()
    );
    assert_eq!(links[0].current, &[second.reference()]);
    let mut forward = request(22);
    forward.relations.push(Relation {
        kind: "corrects".into(),
        other: format!("note:{}", id(99)),
    });
    assert!(
        writer.write_note(&forward).is_ok(),
        "non-control corrects may await collection"
    );
}

#[test]
fn authored_basis_shape_is_admitted_before_reading_its_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(temp.path()).unwrap();
    let n = writer.write_note(&request(40)).unwrap();
    writer
        .archive
        .execute(
            "UPDATE note_target SET seen_hash=?1 WHERE note_id=?2",
            params![vec![0u8; 65536], n.id],
        )
        .unwrap();
    let snapshot = writer.read_handle().unwrap().authored_snapshot().unwrap();
    let error = snapshot.family(&n.reference()).err().unwrap();
    assert!(matches!(
        error,
        Error::Authored(Failure {
            kind: ErrorKind::NotYetAvailable,
            ..
        })
    ));
    assert!(error.to_string().contains("basis length"));
}
