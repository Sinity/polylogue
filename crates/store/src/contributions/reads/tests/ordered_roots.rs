use super::*;
use std::time::{Duration, Instant};
fn until() -> Instant {
    Instant::now() + Duration::from_secs(120)
}
fn temp_count(view: &MessageView) -> i64 {
    locked(&view.read).unwrap().connection.query_row("SELECT count(*) FROM sqlite_temp_master WHERE type='table' AND name LIKE 'polylogue_extra_order_%'",[],|r|r.get(0)).unwrap()
}
fn install(f: &mut Fixture, keys: &[String], block: bool) {
    let prior = f
        .writer
        .current_source_projection(&f.head.token, &[7; 32], 65536)
        .unwrap();
    let build = f.prepare("ordered", 0, 7).build;
    let mut sequence = 0;
    f.writer
        .add_message_source(
            &build,
            SourceRange { start: 0, end: 1 },
            &ProjectionStep {
                expected_sequence: sequence,
                next_cursor: b"source",
            },
            65536,
        )
        .unwrap();
    sequence += 1;
    if block {
        f.writer
            .begin_block(
                &build,
                0,
                0,
                BlockKind::ToolResult {
                    outcome: &polylogue_model::ToolOutcome::Ok,
                    exit_code: None,
                    duration_ms: None,
                },
                BlockPresence {
                    source_block: true,
                    source_text: false,
                },
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"block",
                },
                65536,
            )
            .unwrap();
        sequence += 1;
    }
    for (i, key) in keys.iter().enumerate() {
        f.json(&build, &format!("{i}").into_bytes());
        let slot = Slot::Extra {
            block: block.then_some(BlockSlot::Item(0)),
            key: key.clone(),
        };
        if block {
            f.writer
                .write_block_field(
                    &build,
                    &slot,
                    None,
                    true,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"extra",
                    },
                    65536,
                )
                .unwrap();
        } else {
            f.writer
                .write_message_field(
                    &build,
                    &slot,
                    None,
                    true,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"extra",
                    },
                    65536,
                )
                .unwrap();
        }
        sequence += 1;
    }
    if block {
        f.writer
            .install_block(
                &build,
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"installed",
                },
                65536,
            )
            .unwrap();
        sequence += 1;
    }
    let prepared = f
        .writer
        .finish_message_projection(
            &build,
            &ProjectionStep {
                expected_sequence: sequence,
                next_cursor: b"done",
            },
            BlobHash::from_bytes([f.head.token.step as u8; 32]),
            FinishState::Finished,
            65536,
        )
        .unwrap();
    let source = f
        .writer
        .publish_source_projection(&prepared, prior.as_ref(), b"synthetic-reader-proof", 65536)
        .unwrap();
    f.writer
        .select_compatible_source_message_for_test(&source, 65536)
        .unwrap();
}
fn view(f: &Fixture) -> MessageView {
    f.writer
        .read_handle()
        .unwrap()
        .selected_message(&MessageId::new(f.session.clone(), "ordered").unwrap())
        .unwrap()
        .unwrap()
}
#[test]
fn ordered_extras_selected_message_and_block_use_decoded_keys_and_exact_handles() {
    for block in [false, true] {
        let mut f = Fixture::new();
        let keys: Vec<String> = ["z", "\u{10000}", "\u{e000}", "\"", "\n", "\0", "", "a"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        install(&mut f, &keys, block);
        let view = view(&f);
        let selected = if block {
            Some(view.block_at(0).unwrap().unwrap())
        } else {
            None
        };
        let mut order = match &selected {
            Some(b) => b.ordered_extras(65536, until(), &mut || false),
            None => view.ordered_extras(65536, until(), &mut || false),
        }
        .unwrap();
        assert_eq!(temp_count(&view), 1);
        let mut actual = vec![];
        while let Some(entry) = order.next(until(), &mut || false).unwrap() {
            let FieldRead::Value(value) = entry.value else {
                panic!("unexpected conflict")
            };
            let number: usize =
                serde_json::from_slice(&value.range(0, 65536).unwrap().bytes).unwrap();
            assert_eq!(entry.key, keys[number]);
            actual.push(entry.key);
        }
        let mut expected = keys;
        expected.sort();
        assert_eq!(actual, expected);
        assert_eq!(temp_count(&view), 0);
    }
}
#[test]
fn ordered_extras_accepted_keyset_beyond_four_mib_has_no_aggregate_cap() {
    let mut f = Fixture::new();
    let keys: Vec<String> = (0..80)
        .rev()
        .map(|i| format!("{i:03}{}", "x".repeat(60000)))
        .collect();
    assert!(keys.iter().map(String::len).sum::<usize>() > 4 * 1048576);
    install(&mut f, &keys, false);
    let view = view(&f);
    let mut order = view.ordered_extras(65536, until(), &mut || false).unwrap();
    let mut count = 0;
    let mut last = None;
    while let Some(entry) = order.next(until(), &mut || false).unwrap() {
        if let Some(last) = last {
            assert!(last < entry.key);
        }
        last = Some(entry.key);
        count += 1;
    }
    assert_eq!(count, 80);
    assert_eq!(temp_count(&view), 0);
}
#[test]
fn ordered_extras_drop_cancel_deadline_and_small_key_allowance_clean_owned_temp_only() {
    let mut f = Fixture::new();
    install(&mut f, &["z".into(), "a".into()], false);
    let view = view(&f);
    let survivor = view.ordered_extras(65536, until(), &mut || false).unwrap();
    assert_eq!(temp_count(&view), 1);
    let mut attempts = 0;
    assert!(matches!(
        view.ordered_extras(65536, until(), &mut || {
            attempts += 1;
            attempts >= 4
        }),
        Err(Error::Cancelled)
    ));
    assert_eq!(temp_count(&view), 1);
    assert!(view.ordered_extras(1, until(), &mut || false).is_err());
    assert_eq!(temp_count(&view), 1);
    let mut cancelled = view.ordered_extras(65536, until(), &mut || false).unwrap();
    assert_eq!(temp_count(&view), 2);
    assert!(matches!(
        cancelled.next(Instant::now(), &mut || false),
        Err(Error::Cancelled)
    ));
    assert_eq!(temp_count(&view), 1);
    assert!(cancelled.next(until(), &mut || false).is_err());
    drop(survivor);
    assert_eq!(temp_count(&view), 0);
}
#[test]
fn ordered_extras_old_selected_root_stays_pinned_and_new_selection_does_not_add_keys() {
    let mut f = Fixture::new();
    install(&mut f, &["z".into()], false);
    let old = view(&f);
    let mut ordered = old.ordered_extras(65536, until(), &mut || false).unwrap();
    f.advance();
    install(&mut f, &["a".into()], false);
    assert_eq!(
        ordered.next(until(), &mut || false).unwrap().unwrap().key,
        "z"
    );
    assert!(ordered.next(until(), &mut || false).unwrap().is_none());
    let current = view(&f);
    let mut ordered = current
        .ordered_extras(65536, until(), &mut || false)
        .unwrap();
    assert_eq!(
        ordered.next(until(), &mut || false).unwrap().unwrap().key,
        "a"
    );
}

#[test]
fn ordered_extras_current_removal_epoch_revokes_and_cleans_cursor() {
    let mut f = Fixture::new();
    install(&mut f, &["z".into()], false);
    let view = view(&f);
    let mut order = view.ordered_extras(65536, until(), &mut || false).unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE message SET removal_epoch=removal_epoch+1 WHERE pk=?1",
            [view.owner.message],
        )
        .unwrap();
    assert!(matches!(
        order.next(until(), &mut || false),
        Err(Error::MessagePredecessorChanged)
    ));
    assert_eq!(temp_count(&view), 0);
}
#[test]
fn ordered_extras_mismatched_handle_and_duplicate_key_refuse_without_wrong_value() {
    let mut f = Fixture::new();
    install(&mut f, &["z".into(), "a".into()], false);
    let view = view(&f);
    let mut order = view.ordered_extras(65536, until(), &mut || false).unwrap();
    {
        let read = locked(&view.read).unwrap();
        let table=read.connection.query_row("SELECT name FROM sqlite_temp_master WHERE type='table' AND name LIKE 'polylogue_extra_order_%'",[],|r|r.get::<_,String>(0)).unwrap();
        assert!(
            read.connection
                .execute(
                    &format!("INSERT INTO temp.{table} SELECT * FROM temp.{table}"),
                    []
                )
                .is_err()
        );
        read.connection.execute(&format!("UPDATE temp.{table} SET semantic_key=(SELECT semantic_key FROM temp.{table} WHERE sort_key=x'7a') WHERE sort_key=x'61'"),[]).unwrap();
    }
    assert!(matches!(
        order.next(until(), &mut || false),
        Err(Error::Invariant(_))
    ));
    assert_eq!(temp_count(&view), 0);
}

#[test]
fn ordered_extras_sink_error_drops_private_state_before_any_next_value() {
    let mut f = Fixture::new();
    install(&mut f, &["z".into(), "a".into()], false);
    let view = view(&f);
    let mut acquired = 0;
    let result = (|| -> std::io::Result<()> {
        let mut ordered = view.ordered_extras(65536, until(), &mut || false).unwrap();
        while let Some(entry) = ordered.next(until(), &mut || false).unwrap() {
            acquired += 1;
            let FieldRead::Value(value) = entry.value else {
                panic!("conflict")
            };
            let _ = value.range(0, 65536).unwrap();
            return Err(std::io::ErrorKind::WouldBlock.into());
        }
        Ok(())
    })();
    assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::WouldBlock);
    assert_eq!(acquired, 1);
    assert_eq!(temp_count(&view), 0);
}
#[test]
fn ordered_extras_zero_count_conflict_is_occupied_and_not_dropped() {
    let mut f = Fixture::new();
    install(&mut f, &["a".into()], false);
    let (body,message,session,root,slot,fact):(i64,i64,i64,i64,i64,i64)=f.writer._index.query_row("SELECT b.pk,b.message_pk,b.session_pk,b.field_root_pk,s.pk,f.pk FROM message_body b JOIN message m ON m.selected_body_pk=b.pk JOIN message_field_slot s ON s.message_pk=m.pk JOIN message_field_fact f ON f.slot_pk=s.pk WHERE m.id=?1",[MessageId::new(f.session.clone(),"ordered").unwrap().as_str()],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?))).unwrap();
    let tx = f.writer._index.transaction().unwrap();
    let owner = d::Owner { message, session };
    let mut work = d::Work::default();
    let path = f.writer.root.join("index.db");
    let witness = d::replace(
        &tx,
        &path,
        owner,
        d::Kind::Witnesses,
        None,
        fact as u64,
        Some(d::Edge {
            link: d::Link::Field(fact),
            counts: d::Counts {
                source_fields: 1,
                ..Default::default()
            },
        }),
        &mut work,
    )
    .unwrap()
    .0;
    let changed = d::replace(
        &tx,
        &path,
        owner,
        d::Kind::Fields,
        Some(root),
        slot as u64,
        Some(d::Edge {
            link: d::Link::Conflict {
                slot,
                root: witness,
                height: 15,
            },
            counts: d::Counts::default(),
        }),
        &mut work,
    )
    .unwrap()
    .0;
    tx.execute(
        "UPDATE message_body SET field_root_pk=?1 WHERE pk=?2",
        params![changed, body],
    )
    .unwrap();
    tx.commit().unwrap();
    let view = view(&f);
    let mut ordered = view.ordered_extras(65536, until(), &mut || false).unwrap();
    let entry = ordered.next(until(), &mut || false).unwrap().unwrap();
    assert_eq!(entry.key, "a");
    assert!(matches!(entry.value, FieldRead::Conflict));
    assert!(ordered.next(until(), &mut || false).unwrap().is_none());
}
