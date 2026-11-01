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
    apply(&mut index, &resolver, &[input(2, "source"), barrier(2), delivered(9)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(2));
    // Sinex provisioned the consumer again: it starts over from sequence 1.
    apply(&mut index, &resolver, &[input(2, "source"), barrier(2), delivered(4)]).unwrap();
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
