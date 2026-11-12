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
fn projection_pages_resume_hidden_and_install_the_certificate_with_the_complete_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope", false).unwrap();
    apply(&mut index, &resolver, &[input(1, "obsolete"), barrier(1)]).unwrap();
    // An equal scope can still need replacement at a new stream origin.
    assert!(crate::primary::scope::reset(&mut index, &archive, &resolver, "scope", true).unwrap());
    let mut state = replay_state(&index).unwrap();
    state.projection = Some(ProjectionReplay {
        origin: 4,
        stream_instance: "new-stream".into(),
        position: None,
        cursor: None,
    });
    save_replay(&index, &state).unwrap();
    let page = vec![
        input(2, "retained"),
        control(ReplayControl::ProjectionPage {
            position: 8,
            cursor: "next-page".into(),
        }),
    ];
    apply(&mut index, &resolver, &page).unwrap();
    // A lost reply can repeat the same already-staged page.
    apply(&mut index, &resolver, &page).unwrap();
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 0);
    assert_eq!(checkpoint(&index).unwrap(), None);
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(
        state.projection.as_ref().unwrap().cursor.as_deref(),
        Some("next-page")
    );
    assert_eq!(state.basis, None);
    assert!(apply(&mut index, &resolver, &[barrier(8)]).is_err());
    let basis = ProjectionBasis {
        stamp: "owner-signature".into(),
        position: 8,
        scope: "scope".into(),
        catalog: "catalog".into(),
    };
    index.execute_batch("CREATE TRIGGER stop_snapshot BEFORE INSERT ON meta WHEN new.key='confirmed_position' BEGIN SELECT raise(ABORT,'stop snapshot'); END").unwrap();
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                input(3, "archived"),
                Confirmed {
                    position: 6,
                    event: input(3, "archived").event,
                    output_index: 0,
                    native: None,
                    change: ConfirmedChange::Archived
                },
                control(ReplayControl::ProjectionComplete {
                    basis: basis.clone()
                })
            ]
        )
        .is_err()
    );
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(replay_state(&index).unwrap().basis, None);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 0);
    index.execute_batch("DROP TRIGGER stop_snapshot").unwrap();
    apply(&mut index, &resolver, &[]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(8));
    let state = replay_state(&index).unwrap();
    assert_eq!(state.basis, Some(basis));
    assert_eq!(state.projection, None);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 1);
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_fact WHERE state='archived'"
        ),
        1
    );
    assert_eq!(rows(&index, "SELECT count(*) FROM primary_pending"), 0);
    assert!(!has_fact(&index, &input(1, "obsolete").event).unwrap());
    assert_eq!(
        state.completed_delivery, None,
        "a projection certificate is not a native delivery"
    );
    apply(&mut index, &resolver, &[completed(8, 20)]).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, Some(8));
    assert_eq!(
        state.completed_delivery.unwrap().stream_sequence,
        20,
        "a real marker at the unchanged head supplies completion evidence"
    );
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
    // A native retry may repeat an already applied delivery.
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
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-a", false).unwrap();
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
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope-b", false).unwrap();
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
    crate::primary::scope::reset(&mut index, &archive, &resolver, "scope", false).unwrap();
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
    assert_eq!(previous.through(&index, &archive).unwrap(), 2);
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

fn completed(position: i64, sequence: u64) -> Confirmed {
    control(ReplayControl::Completion {
        position,
        delivery: Some(CompletedDelivery {
            stream_instance: "instance".into(),
            stream_sequence: sequence,
            fingerprint: "a".repeat(64),
        }),
    })
}

#[test]
fn completed_receipt_survives_partial_pages_and_rolls_back_with_visible_progress() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[input(2, "first"), completed(2, 5), delivered(5)],
    )
    .unwrap();
    apply(&mut index, &resolver, &[input(3, "later"), delivered(7)]).unwrap();
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, Some(2));
    assert_eq!(state.completed_delivery.unwrap().stream_sequence, 5);
    assert_eq!(state.staged_delivery.unwrap().stream_sequence, 7);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 1);
    // A repeated old barrier may appear after the newer partial data.
    apply(&mut index, &resolver, &[completed(2, 8), delivered(8)]).unwrap();
    assert_eq!(
        replay_state(&index)
            .unwrap()
            .completed_delivery
            .unwrap()
            .stream_sequence,
        5
    );
    index.execute_batch("CREATE TRIGGER stop_progress BEFORE UPDATE ON meta WHEN new.key='confirmed_replay' AND json_extract(new.value,'$.applied_position')=3 BEGIN SELECT raise(ABORT,'stop progress'); END").unwrap();
    assert!(apply(&mut index, &resolver, &[completed(3, 9), delivered(9)]).is_err());
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, Some(2));
    assert_eq!(state.completed_delivery.unwrap().stream_sequence, 5);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 1);
    index.execute_batch("DROP TRIGGER stop_progress").unwrap();
    apply(&mut index, &resolver, &[]).unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, Some(3));
    assert_eq!(state.completed_delivery.unwrap().stream_sequence, 9);
    assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 2);
}

#[test]
fn a_new_completion_after_higher_pending_data_cannot_become_a_restart_boundary() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            input(2, "first"),
            input(3, "later"),
            completed(2, 8),
            delivered(8),
        ],
    )
    .unwrap();
    let state = replay_state(&index).unwrap();
    assert_eq!(state.applied_position, Some(2));
    assert_eq!(state.completed_delivery, None);
    assert_eq!(
        rows(
            &index,
            "SELECT count(*) FROM primary_pending WHERE position=3"
        ),
        1
    );
    apply(&mut index, &resolver, &[completed(3, 9)]).unwrap();
    assert_eq!(
        replay_state(&index)
            .unwrap()
            .completed_delivery
            .unwrap()
            .stream_sequence,
        9
    );
}

#[test]
fn malformed_and_wrong_kind_pending_evidence_refuse_visible_progress() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[input(2, "first"), delivered(5)]).unwrap();
    let projection =
        serde_json::json!({"stamp":"certificate","position":2,"scope":"scope","catalog":"catalog"});
    for evidence in [
        serde_json::json!({"stream_instance":"instance","stream_sequence":5,"fingerprint":"bad"}),
        serde_json::json!({"stream_instance":"instance","stream_sequence":0,"fingerprint":"a".repeat(64)}),
        serde_json::json!({"stream_instance":"instance","stream_sequence":5,"fingerprint":"a".repeat(64),"stamp":"ambiguous","position":2,"scope":"scope","catalog":"catalog"}),
        projection,
    ] {
        index
            .execute(
                "INSERT OR REPLACE INTO primary_complete(position,evidence) VALUES (2,?1)",
                [packed(&evidence).unwrap()],
            )
            .unwrap();
        assert!(
            apply(&mut index, &resolver, &[]).is_err(),
            "invalid evidence must refuse: {evidence}"
        );
        assert_eq!(checkpoint(&index).unwrap(), None);
        assert_eq!(replay_state(&index).unwrap().completed_delivery, None);
        assert_eq!(rows(&index, "SELECT count(*) FROM source_input"), 0);
        assert_eq!(rows(&index, "SELECT count(*) FROM primary_pending"), 1);
    }
    index.execute("DELETE FROM primary_complete", []).unwrap();
    apply(&mut index, &resolver, &[completed(2, 5)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
    assert_eq!(
        replay_state(&index)
            .unwrap()
            .completed_delivery
            .unwrap()
            .stream_sequence,
        5
    );
}
