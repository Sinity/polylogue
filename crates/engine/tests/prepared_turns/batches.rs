use super::*;
fn batch_budget() -> Budget {
    Budget {
        cells: 64,
        encoded_bytes: 2 * 1024 * 1024,
    }
}
fn many() -> Vec<CellChange> {
    (0u16..257)
        .map(|n| CellChange {
            namespace: "synthetic.batch".into(),
            key: n.to_be_bytes().to_vec(),
            value: (n % 7 != 0).then(|| n.to_be_bytes().repeat(32)),
        })
        .collect()
}
#[test]
fn bounded_batches_reduce_turns_and_install_every_exact_cell_without_early_progress() {
    let (_dir, mut writer, head) = fixture();
    let pinned = writer.parser_snapshot(&head.token).unwrap();
    let changes = many();
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), many()).unwrap();
    let mut turns = 0usize;
    let published = loop {
        let prior = writer.parser_work(&head.token).unwrap().unwrap();
        let result = pending.turn(&mut writer, batch_budget()).unwrap();
        turns += 1;
        // Peer writes/readback remain possible between every admitted transaction.
        let blob = writer.put_file_blob(&[turns as u8]).unwrap();
        assert_eq!(
            writer
                .read_handle()
                .unwrap()
                .read_blob_range(blob, 0, 1)
                .unwrap(),
            [turns as u8]
        );
        assert!(
            pinned
                .cell("synthetic.batch", &[0, 1], 4096)
                .unwrap()
                .is_none()
        );
        if let Turn::Published(h) = result {
            break h;
        }
        let next = writer.parser_work(&head.token).unwrap().unwrap();
        assert!(next.token.staged - prior.token.staged <= 64);
        assert!(next.token.applied - prior.token.applied <= 64);
        assert_eq!(
            writer
                .parser_head(head.token.input_pk, head.token.chain)
                .unwrap()
                .unwrap()
                .parsed_upto,
            0
        );
    };
    // 5 bounded staging + seal + 5 bounded application + publication, rather
    // than 257 staging + seal + 129 two-cell applications + publication.
    assert_eq!(turns, 12);
    assert_eq!(published.parsed_upto, 100);
    assert!(published.finalization_pending);
    assert_eq!(published.status, "finalizing");
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    for c in changes {
        assert_eq!(snapshot.cell(&c.namespace, &c.key, 4096).unwrap(), c.value);
    }
}
#[test]
fn staging_prefix_keeps_later_byte_obligation_and_unfit_first_member_is_atomic() {
    let (_dir, mut writer, head) = fixture();
    let changes = vec![
        CellChange {
            namespace: "synthetic".into(),
            key: vec![0],
            value: Some(vec![1; 8]),
        },
        CellChange {
            namespace: "synthetic".into(),
            key: vec![1],
            value: Some(vec![2; 200]),
        },
        CellChange {
            namespace: "synthetic".into(),
            key: vec![2],
            value: Some(vec![3; 8]),
        },
    ];
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), changes).unwrap();
    let budget = Budget {
        cells: 64,
        encoded_bytes: 128,
    };
    assert!(matches!(
        pending.turn(&mut writer, budget).unwrap(),
        Turn::Yielded
    ));
    let before = writer.parser_work(&head.token).unwrap().unwrap().token;
    assert_eq!(before.staged, 1);
    assert!(matches!(
        pending.turn(&mut writer, budget),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(
        writer.parser_work(&head.token).unwrap().unwrap().token,
        before
    );
    let published = loop {
        if let Turn::Published(h) = pending.turn(&mut writer, batch_budget()).unwrap() {
            break h;
        }
    };
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    for (n, value) in [vec![1; 8], vec![2; 200], vec![3; 8]]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            snapshot.cell("synthetic", &[n as u8], 4096).unwrap(),
            Some(value)
        );
    }
}
#[test]
fn malformed_batch_and_stale_token_never_install_a_partial_transaction() {
    let (_dir, mut writer, head) = fixture();
    let mut changes = many();
    changes[1].namespace = "@work:reserved".into();
    let mut pending = PendingInstall::begin(&mut writer, &head, progress(), changes).unwrap();
    let original = writer.parser_work(&head.token).unwrap().unwrap().token;
    assert!(matches!(
        pending.turn(&mut writer, batch_budget()),
        Err(Error::Invariant(_))
    ));
    assert_eq!(
        writer.parser_work(&head.token).unwrap().unwrap().token,
        original
    );
    let c = many();
    writer
        .stage_parser(&original, &c[..64], batch_budget())
        .unwrap();
    assert!(matches!(
        writer.stage_parser(&original, &c[64..128], batch_budget()),
        Err(Error::StaleParserState)
    ));
    assert_eq!(
        writer
            .parser_work(&head.token)
            .unwrap()
            .unwrap()
            .token
            .staged,
        64
    );
}
#[test]
fn interrupted_batches_discard_unsealed_and_finish_sealed_without_replay() {
    for sealed in [false, true] {
        let (_dir, mut writer, head) = fixture();
        let changes = many();
        let mut pending = PendingInstall::begin(&mut writer, &head, progress(), many()).unwrap();
        let count = if sealed { 7 } else { 1 }; // five stage + seal + one apply.
        for _ in 0..count {
            assert!(matches!(
                pending.turn(&mut writer, batch_budget()).unwrap(),
                Turn::Yielded
            ));
        }
        let work = writer.parser_work(&head.token).unwrap().unwrap();
        assert_eq!(work.applying, sealed);
        assert_eq!(work.token.applied, if sealed { 64 } else { 0 });
        drop(pending); // Cancellation loses the in-memory delta, not durable obligations.
        let mut turns = 0usize;
        loop {
            turns += 1;
            assert!(turns <= 5);
            match recover_turn(&mut writer, &head.token, batch_budget()).unwrap() {
                Recovery::Yielded => assert_eq!(
                    writer
                        .parser_head(head.token.input_pk, head.token.chain)
                        .unwrap()
                        .unwrap()
                        .parsed_upto,
                    0
                ),
                Recovery::Reparse => {
                    assert!(!sealed);
                    assert_eq!(
                        writer
                            .parser_snapshot(&head.token)
                            .unwrap()
                            .head
                            .parsed_upto,
                        0
                    );
                    break;
                }
                Recovery::Published(h) => {
                    assert!(sealed);
                    assert_eq!(turns, 5);
                    assert_eq!(h.parsed_upto, 100);
                    assert!(h.finalization_pending);
                    let snapshot = writer.parser_snapshot(&h.token).unwrap();
                    for c in &changes {
                        assert_eq!(snapshot.cell(&c.namespace, &c.key, 4096).unwrap(), c.value);
                    }
                    break;
                }
            }
        }
    }
}

#[test]
fn staging_prefix_accounts_for_exact_encoded_framing_before_mutation() {
    let (_dir, mut writer, head) = fixture();
    let work = writer.prepare_parser(&head, progress()).unwrap();
    let changes = changes();
    // Prepared framing is 25 bytes plus the work namespace and original member.
    let one = 25
        + format!("@work:{}", work.token.work_id).len()
        + changes[0].namespace.len()
        + changes[0].key.len()
        + changes[0].value.as_ref().unwrap().len();
    let below_two = Budget {
        cells: 64,
        encoded_bytes: one * 2 - 1,
    };
    assert_eq!(work.token.staging_prefix(&changes, below_two).unwrap(), 1);
    assert!(matches!(
        writer.stage_parser(&work.token, &changes[..2], below_two),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(
        writer.parser_work(&head.token).unwrap().unwrap().token,
        work.token
    );
    let exact_two = Budget {
        cells: 64,
        encoded_bytes: one * 2,
    };
    assert_eq!(work.token.staging_prefix(&changes, exact_two).unwrap(), 2);
    let staged = writer
        .stage_parser(&work.token, &changes[..2], exact_two)
        .unwrap();
    assert_eq!(staged.token.staged, 2);
    assert_eq!(staged.token.applied, 0);
    assert_eq!(staged.token.predecessor, head.token);
}
