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
fn history_retry_stale_revision_and_withdrawals_survive_index_loss() {
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
fn replay_sorts_sets_and_rejects_unsupported_intent_without_rows() {
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
fn revise_and_withdraw_need_committed_parents_and_refusals_roll_back() {
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
    let links = snapshot
        .backlinks_page(
            &first.targets[0].reference,
            snapshot.archive_counter,
            0,
            16,
            default_until(),
            &mut || false,
        )
        .unwrap();
    assert!(links.after.is_none());
    let links = links.families;
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
fn oversized_seen_hash_is_refused_before_reading_it() {
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
    assert!(error.to_string().contains("seen_hash"), "{error}");
}

#[test]
fn long_revision_history_keeps_current_history_and_undo_readable() {
    for (count, bytes) in [(129, 5), (16, 16384)] {
        let temp = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(temp.path()).unwrap();
        let mut first = request(1000);
        first.value = "a".repeat(bytes);
        let first = writer.write_note(&first).unwrap();
        let mut latest = first.clone();
        let mut last_request = None;
        for n in 1..count {
            let mut next = revise(1000 + n, latest.reference());
            next.value = char::from(b'a' + (n % 20) as u8).to_string().repeat(bytes);
            latest = writer.write_note(&next).unwrap();
            last_request = Some(next);
        }
        let reader = writer.read_handle().unwrap();
        let snapshot = reader.authored_snapshot().unwrap();
        let cutoff = snapshot.archive_counter;
        let current = snapshot.family(&format!("note:{}", first.root)).unwrap();
        assert_eq!(current.current, [latest.reference()]);
        assert_eq!(current.shown[0].revision, latest);
        assert_eq!(
            snapshot.family(&first.reference()).unwrap().shown[0].revision,
            first
        );
        assert_eq!(snapshot.exact(&first.reference(), cutoff).unwrap(), first);
        let mut history = vec![];
        let mut after = 0;
        loop {
            let page = snapshot
                .family_page(
                    &format!("note:{}", first.root),
                    cutoff,
                    PageKind::History,
                    after,
                    default_until(),
                    &mut || false,
                )
                .unwrap();
            history.extend(page.history);
            match page.history_after {
                Some(next) => {
                    assert!(next > after);
                    after = next;
                }
                None => break,
            }
        }
        assert_eq!(history.len(), count as usize);
        assert_eq!(history[0], first.reference());
        assert_eq!(history.last(), Some(&latest.reference()));
        // Each withdrawal and undo adds history; reading it must not load the
        // old 16 KiB values or run out of room for the last undo.
        let mut control = writer
            .write_note(&withdraw(2000, latest.reference()))
            .unwrap();
        for n in 1..20 {
            let now = reader
                .authored_snapshot()
                .unwrap()
                .family(&format!("note:{}", first.root))
                .unwrap();
            assert_eq!(now.current == [latest.reference()], n % 2 == 0);
            control = writer
                .write_note(&withdraw(2000 + n, control.reference()))
                .unwrap();
        }
        let wr = writer
            .write_note(&withdraw(2100, format!("note:{}", first.root)))
            .unwrap();
        assert!(
            reader
                .authored_snapshot()
                .unwrap()
                .family(&format!("note:{}", first.root))
                .unwrap()
                .current
                .is_empty()
        );
        writer.write_note(&withdraw(2101, wr.reference())).unwrap();
        assert_eq!(
            reader
                .authored_snapshot()
                .unwrap()
                .family(&format!("note:{}", first.root))
                .unwrap()
                .current,
            [latest.reference()]
        );
        let counter = writer.status().unwrap().write_counter;
        assert_eq!(
            writer.write_note(last_request.as_ref().unwrap()).unwrap(),
            latest,
            "committed replay precedes present control state and returns the original effect"
        );
        assert_eq!(writer.status().unwrap().write_counter, counter);
        // Every page reports status at the same archive counter, even after a
        // later withdrawal and a withdrawal of that withdrawal.
        let old = reader
            .authored_snapshot()
            .unwrap()
            .family_page(
                &format!("note:{}", first.root),
                cutoff,
                PageKind::Family,
                0,
                default_until(),
                &mut || false,
            )
            .unwrap();
        assert_eq!(old.current, [latest.reference()]);
        assert!(old.withdrawal_trail.is_empty());
    }
}

#[test]
fn backlinks_pages_list_each_family_once_with_all_matches() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(temp.path()).unwrap();
    let first = writer.write_note(&request(1)).unwrap();
    let second = writer.write_note(&revise(2, first.reference())).unwrap();
    for n in 3..=18 {
        writer.write_note(&request(n)).unwrap();
    }
    let target = &first.targets[0].reference;
    let reader = writer.read_handle().unwrap();
    let cutoff = reader.authored_snapshot().unwrap().archive_counter;
    let mut roots = vec![];
    let mut after = 0;
    loop {
        let snapshot = reader.authored_snapshot().unwrap();
        let page = snapshot
            .backlinks_page(target, cutoff, after, 1, default_until(), &mut || false)
            .unwrap();
        for family in page.families {
            if family.root == first.root {
                assert_eq!(
                    family.matching_target_revisions.as_ref().unwrap(),
                    &[first.reference(), second.reference()]
                );
            }
            roots.push(family.root);
        }
        match page.after {
            Some(next) => {
                assert!(next > after);
                after = next;
            }
            None => break,
        }
    }
    assert_eq!(roots.len(), 17);
    let unique = roots.iter().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), 17);
    writer
        .write_note(&withdraw(19, second.reference()))
        .unwrap();
    let old = reader
        .authored_snapshot()
        .unwrap()
        .backlinks_page(target, cutoff, 0, 1, default_until(), &mut || false)
        .unwrap();
    assert_eq!(old.families[0].current, [second.reference()]);
    let now = reader
        .authored_snapshot()
        .unwrap()
        .backlinks_page(
            target,
            writer.status().unwrap().write_counter,
            0,
            1,
            default_until(),
            &mut || false,
        )
        .unwrap();
    assert_eq!(now.families[0].current, [first.reference()]);
}

#[test]
fn deadline_refusal_rolls_back_write_but_replay_still_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(dir.path()).unwrap();
    let first = writer.write_note(&request(1)).unwrap();
    let before = writer.status().unwrap().write_counter;
    let next = withdraw(2, first.reference());
    assert!(matches!(
        writer.write_note_until(&next, Instant::now()),
        Err(Error::Cancelled)
    ));
    assert_eq!(writer.status().unwrap().write_counter, before);
    assert_eq!(
        writer
            .archive
            .query_row(
                "SELECT count(*) FROM client_request WHERE id=?1",
                [id(2)],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    assert_eq!(
        writer
            .archive
            .query_row("SELECT count(*) FROM note WHERE id=?1", [id(2)], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    let committed = writer.write_note(&next).unwrap();
    assert_eq!(
        writer.write_note_until(&next, Instant::now()).unwrap(),
        committed
    );
}

#[test]
fn malformed_controls_refuse_status_but_pinned_value_reads_work() {
    for cyclic in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(dir.path()).unwrap();
        for n in [1, 2] {
            writer.archive.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,local_seq) VALUES(?1,?2,'note','malformed import','agent','machine',0,?3)",params![id(n),id(1),n]).unwrap();
        }
        writer
            .archive
            .execute(
                "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,'revises',?2)",
                params![id(2), format!("note:{}@{}", id(1), id(1))],
            )
            .unwrap();
        writer
            .archive
            .execute(
                "INSERT INTO note_relation(note_id,kind,other) VALUES(?1,'withdraws',?2)",
                params![
                    id(1),
                    if cyclic {
                        format!("note:{}@{}", id(1), id(2))
                    } else {
                        format!("note:{}@{}", id(99), id(99))
                    }
                ],
            )
            .unwrap();
        writer
            .archive
            .execute("UPDATE meta SET value='2' WHERE key='write_counter'", [])
            .unwrap();
        let snapshot = writer.read_handle().unwrap().authored_snapshot().unwrap();
        assert!(matches!(
            snapshot.family(&format!("note:{}", id(1))),
            Err(Error::Authored(Failure {
                kind: ErrorKind::NotYetAvailable,
                ..
            }))
        ));
        assert_eq!(
            snapshot
                .exact(&format!("note:{}@{}", id(1), id(1)), 2)
                .unwrap()
                .value,
            "malformed import"
        );
        let before = writer.status().unwrap().write_counter;
        assert!(matches!(
            writer.write_note(&withdraw(3, format!("note:{}@{}", id(1), id(1)))),
            Err(Error::Authored(Failure {
                kind: ErrorKind::NotYetAvailable,
                ..
            }))
        ));
        assert_eq!(writer.status().unwrap().write_counter, before);
        assert_eq!(
            writer
                .archive
                .query_row(
                    "SELECT count(*) FROM client_request WHERE id=?1",
                    [id(3)],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            0
        );
    }
}
