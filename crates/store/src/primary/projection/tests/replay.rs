use super::*;
use crate::primary::*;

fn control(control: ReplayControl) -> Confirmed {
    Confirmed {
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(control)),
        native: None,
    }
}
fn delivered(sequence: u64) -> Confirmed {
    control(ReplayControl::Delivered {
        delivery: NativeDeliveryCursor {
            stream_instance: "instance".into(),
            stream_sequence: sequence,
        },
    })
}
fn input_product(id: &str) -> Product {
    match input(1, id).change {
        ConfirmedChange::Live(product) => *product,
        _ => unreachable!("input builds a live fact"),
    }
}
fn rows(index: &Connection, sql: &str) -> i64 {
    index.query_row(sql, [], |row| row.get(0)).unwrap()
}

#[test]
fn a_staged_delivery_survives_reopen_without_advancing_the_applied_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[input(3, "source"), delivered(7)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), None);
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, None);
    assert_eq!(
        state.staged_delivery,
        Some(NativeDeliveryCursor {
            stream_instance: "instance".into(),
            stream_sequence: 7,
        })
    );
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 0);
    apply(&mut index, &resolver, &[barrier(3), delivered(8)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(3));
    assert_eq!(replay_state(&index).unwrap().applied_position, Some(3));
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 1);
}

#[test]
fn a_restarted_consumer_redelivers_applied_positions_without_changing_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[input(2, "source"), barrier(2), delivered(9)],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
    // Sinex provisioned the consumer again: it starts over from sequence 1.
    apply(
        &mut index,
        &resolver,
        &[input(2, "source"), barrier(2), delivered(4)],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 1);
    assert_eq!(rows(&index, "SELECT count(*) FROM primary_pending"), 0);
    assert_eq!(
        replay_state(&index)
            .unwrap()
            .staged_delivery
            .map(|delivery| delivery.stream_sequence),
        Some(4)
    );
}

/// law: lifecycle-reaches-index
#[test]
fn an_event_erased_before_this_index_read_it_is_recorded_as_tombstoned() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                native: None,
                position: 2,
                output_index: 0,
                event: "event-erased".into(),
                change: ConfirmedChange::Tombstoned,
            },
            barrier(2),
        ],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
    let (state, product): (String, Option<Vec<u8>>) = index
        .query_row(
            "SELECT state,product FROM primary_fact WHERE event='event-erased'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(state, "tombstoned");
    assert_eq!(product, None);
    // A tombstoned event never becomes live again.
    let refused = apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                native: None,
                position: 3,
                output_index: 0,
                event: "event-erased".into(),
                change: ConfirmedChange::Live(Box::new(input_product("x"))),
            },
            barrier(3),
        ],
    )
    .unwrap_err();
    assert!(refused.to_string().contains("tombstoned"), "{refused}");
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
}

#[test]
fn a_scope_reset_cannot_share_a_page_with_facts() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                input(1, "source"),
                control(ReplayControl::Reset {
                    scope: "other".into()
                })
            ],
        )
        .is_err()
    );
    assert_eq!(rows(&index, "SELECT count(*) FROM primary_pending"), 0);
}

#[test]
fn an_unreadable_event_in_the_middle_of_a_page_is_refused_and_later_events_apply() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            input(1, "before"),
            barrier(1),
            Confirmed {
                native: None,
                position: 2,
                output_index: 0,
                event: "event-unknown".into(),
                change: ConfirmedChange::Refused {
                    kind: "polylogue.future@7".into(),
                    reason: "unsupported Polylogue kind polylogue.future@7".into(),
                    unreadable_input: true,
                },
            },
            barrier(2),
            input(3, "after"),
            barrier(3),
        ],
    )
    .unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(3));
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 2);
    let refused: (i64, String, String) = index
        .query_row(
            "SELECT position,kind,reason FROM primary_refused WHERE event='event-unknown'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        refused,
        (
            2,
            "polylogue.future@7".into(),
            "unsupported Polylogue kind polylogue.future@7".into()
        )
    );
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_fact WHERE event='event-unknown' AND state='live' AND product IS NULL"
        ),
        1
    );
}

#[test]
fn primary_source_cut_requires_completed_scope_and_survives_new_admissions() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) =
        crate::Archive::open_primary(directory.path(), &directory.path().join("bytes")).unwrap();
    let mut index =
        crate::archive::open_write(&archive.index_db(), crate::archive::cache::SETUP).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-a").unwrap();
    let boundary =
        |index: &Connection| crate::primary::SourceBoundary::read(index, index, &archive).unwrap();
    assert!(boundary(&index).is_none());
    let observed = |position, id: &str| {
        let mut fact = input(position, "first");
        if let ConfirmedChange::Live(product) = &mut fact.change
            && let Product::Input { rows, .. } = product.as_mut()
        {
            rows.observation = Some(serde_json::from_value(serde_json::json!({
                    "id":id,"machine":"machine","host":"host","path":"/home/op/source",
                    "path_bytes":b"/home/op/source".to_vec(),"class":"seen","polylogue_version":"test","seen_at":position
            })).unwrap());
        }
        fact
    };
    apply(
        &mut index,
        &resolver,
        &[
            observed(2, "first-head"),
            control(ReplayControl::Head { position: 9 }),
            delivered(1),
        ],
    )
    .unwrap();
    assert!(
        boundary(&index).is_none(),
        "a staged page and stream head do not complete the scope"
    );
    drop(index);
    let mut index =
        crate::archive::open_write(&archive.index_db(), crate::archive::cache::SETUP).unwrap();
    assert!(boundary(&index).is_none());
    apply(&mut index, &resolver, &[barrier(2)]).unwrap();
    let cut = boundary(&index).unwrap();
    assert_eq!(cut.through(&index, &archive).unwrap(), 2);
    let mut append = observed(4, "later-head");
    append.event = "event-first-append".into();
    apply(
        &mut index,
        &resolver,
        &[append, input(5, "next"), barrier(5)],
    )
    .unwrap();
    assert_eq!(cut.through(&index, &archive).unwrap(), 2);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 2);
    let reads = crate::ReadPool::open(&archive).unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .source_subject_at(&cut, "first")
            .unwrap()
            .unwrap()
            .observation,
        "first-head"
    );
    let later = boundary(&index).unwrap();
    for (position, change) in [
        (7, ConfirmedChange::Archived),
        (8, ConfirmedChange::Restored),
    ] {
        let previous = boundary(&index).unwrap();
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    event: "event-first".into(),
                    output_index: 0,
                    native: None,
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert!(matches!(
            previous.through(&index, &archive),
            Err(StoreError::ScopeChanged)
        ));
    }
    assert!(matches!(
        later.through(&index, &archive),
        Err(StoreError::ScopeChanged)
    ));
    let previous = boundary(&index).unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-b").unwrap();
    assert!(boundary(&index).is_none());
    assert!(matches!(
        previous.through(&index, &archive),
        Err(StoreError::ScopeChanged)
    ));
}

#[test]
fn unreadable_input_blocks_absence_until_its_real_lifecycle_or_readable_replacement() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) =
        crate::Archive::open_primary(directory.path(), &directory.path().join("bytes")).unwrap();
    let mut index =
        crate::archive::open_write(&archive.index_db(), crate::archive::cache::SETUP).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope").unwrap();
    let boundary =
        |index: &Connection| crate::primary::SourceBoundary::read(index, index, &archive).unwrap();
    let refused = |position, event: &str, unreadable_input| Confirmed {
        position,
        event: event.into(),
        output_index: 0,
        native: None,
        change: ConfirmedChange::Refused {
            kind: "opaque-native-kind".into(),
            reason: "synthetic unavailable body".into(),
            unreadable_input,
        },
    };
    apply(
        &mut index,
        &resolver,
        &[
            input(1, "known"),
            barrier(1),
            refused(2, "non-input", false),
            barrier(2),
        ],
    )
    .unwrap();
    let previous = boundary(&index).unwrap();
    apply(
        &mut index,
        &resolver,
        &[refused(3, "possible-input", true), barrier(3)],
    )
    .unwrap();
    assert!(boundary(&index).is_none());
    assert_eq!(
        rows(&index, "SELECT count(*) FROM source_input WHERE id='known'"),
        1
    );
    assert!(matches!(
        previous.through(&index, &archive),
        Err(StoreError::ScopeChanged)
    ));
    let lifecycle = |position, change| Confirmed {
        position,
        event: "possible-input".into(),
        output_index: 0,
        native: None,
        change,
    };
    apply(
        &mut index,
        &resolver,
        &[lifecycle(4, ConfirmedChange::Archived), barrier(4)],
    )
    .unwrap();
    assert!(boundary(&index).is_some());
    apply(
        &mut index,
        &resolver,
        &[lifecycle(5, ConfirmedChange::Restored), barrier(5)],
    )
    .unwrap();
    assert!(boundary(&index).is_none());
    let mut replacement = input(6, "recovered");
    replacement.event = "possible-input".into();
    apply(&mut index, &resolver, &[replacement, barrier(6)]).unwrap();
    assert!(boundary(&index).is_some());
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_refused WHERE event='possible-input'"
        ),
        0
    );
    assert_eq!(
        rows(
            &index,
            "SELECT admitted FROM primary_fact WHERE event='possible-input'"
        ),
        3
    );
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM source_input WHERE id='recovered'"
        ),
        1
    );
    apply(
        &mut index,
        &resolver,
        &[refused(7, "erased-input", true), barrier(7)],
    )
    .unwrap();
    assert!(boundary(&index).is_none());
    let mut erased = lifecycle(8, ConfirmedChange::Tombstoned);
    erased.event = "erased-input".into();
    apply(&mut index, &resolver, &[erased, barrier(8)]).unwrap();
    assert!(boundary(&index).is_some());
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_refused WHERE event='erased-input'"
        ),
        0
    );
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_fact WHERE event='erased-input' AND state='tombstoned' AND product IS NULL AND native_context IS NULL"
        ),
        1
    );
}
