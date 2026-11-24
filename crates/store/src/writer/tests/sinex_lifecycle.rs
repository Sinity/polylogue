//! Local rows come from confirmed original bytes and respect the lifecycle
//! of their matching Sinex facts.
use super::*;
use crate::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};

fn reading(text: &str) -> Batch {
    let mut batch = batch(
        "original",
        Route::Live,
        0,
        vec![row(
            12,
            text_message(
                "message",
                "session",
                Role::Assistant,
                text,
                Some("end_turn"),
            ),
        )],
        &["session"],
    );
    batch.change = Change::Reparse { chain: 1 };
    batch.observation = None;
    batch.parse[0].parse_version = "synthetic=1".into();
    batch
}

/// Writes the local rows of a re-parse of a confirmed chain, as the serial
/// writer does after the parsed events are in the spool.
fn write_local(index: &mut Connection, archive: &Archive, mut batch: Batch) {
    batch.parse[0].prepare();
    let resolver =
        resolve::Resolver::new(archive, index, resolve::IndexUpkeep::PerTransaction).unwrap();
    resolver.batches(&mut [&mut batch]).unwrap();
    let texts = crate::text_rows::TextIds::new(
        index,
        false,
        crate::archive::MemoryBudget::default().keys() / 3,
    )
    .unwrap();
    let text_transaction = texts.transaction();
    let tx = index.transaction().unwrap();
    let counter = bump_counter(&tx).unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    let step = &batch.parse[0];
    write_rows(
        &tx,
        &batch,
        step,
        rows::ReadingWrite {
            texts: &texts,
            resolution: step.resolution.as_ref().unwrap(),
            counter,
            upkeep: &resolve::IndexUpkeep::PerTransaction,
            source_commit: Some(&fixture_commit()),
            changes: &changes,
            retraction: false,
            replacement: false,
        },
    )
    .unwrap();
    rows::finish(
        &tx,
        &changes,
        &resolve::IndexUpkeep::PerTransaction,
        &texts,
        counter,
    )
    .unwrap();
    tx.commit().unwrap();
    text_transaction.commit();
}

/// Applies one confirmed position holding `changes` for the event `event`.
fn apply_position(
    index: &mut Connection,
    archive: &Archive,
    position: i64,
    changes: Vec<(String, ConfirmedChange)>,
) {
    let resolver =
        resolve::Resolver::new(archive, index, resolve::IndexUpkeep::PerTransaction).unwrap();
    let mut events = changes
        .into_iter()
        .enumerate()
        .map(|(ordinal, (event, change))| Confirmed {
            sinex_context: None,
            position,
            output_index: ordinal as u64,
            event,
            change,
        })
        .collect::<Vec<_>>();
    events.push(Confirmed {
        sinex_context: None,
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
            position,
            delivery: None,
        })),
    });
    crate::primary::confirmed::apply(
        &crate::text_rows::TextIds::new(
            index,
            false,
            crate::archive::MemoryBudget::default().keys() / 3,
        )
        .unwrap(),
        index,
        &resolver,
        &events,
    )
    .unwrap();
}

fn shown_text(archive: &Archive, message: &polylogue_model::MessageId) -> Option<String> {
    let pool = ReadPool::open(archive).unwrap();
    let reader = pool.reader().unwrap();
    reader
        .message_with_block_selection(message)
        .unwrap()
        .map(|(value, _)| value.text())
}

#[test]
fn confirmed_lifecycle_controls_visibility_when_original_bytes_are_reparsed() {
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let mut index = Connection::open(archive.index_db()).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let original = reading("locally parsed original");
    let message = original.parse[0].output.messages[0].message.id.clone();
    write_local(&mut index, &archive, original.clone());

    // Sinex confirms the parsed events of the same step at positions after the
    // material acknowledgment the local rows carry.
    let mut prepared = original.clone();
    prepared.parse[0].prepare();
    let facts = crate::primary::products::parse(
        &prepared.input,
        1,
        Some(prepared.length),
        &prepared.parse[0],
    )
    .unwrap()
    .into_iter()
    .enumerate()
    .map(|(ordinal, product)| {
        (
            format!("event-{ordinal}"),
            ConfirmedChange::Live(Box::new(Product::Interpreted {
                reading: product.reading().unwrap().clone(),
                record: crate::primary::SourceRecord::of(&product).unwrap().unwrap(),
            })),
        )
    })
    .collect::<Vec<_>>();
    let message_event = facts
        .iter()
        .position(|(_, change)| {
            matches!(change, ConfirmedChange::Live(product)
                if matches!(product.as_ref(), Product::Interpreted { record, .. } if record.family == "message"))
        })
        .map(|ordinal| format!("event-{ordinal}"))
        .unwrap();
    apply_position(&mut index, &archive, 10, facts);
    let committed = || {
        ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .message_commit(&message)
            .unwrap()
    };
    assert_eq!(committed(), Some(fixture_commit().sequence as u64));

    for (position, change) in [
        (11, ConfirmedChange::Archived),
        (12, ConfirmedChange::Restored),
        (13, ConfirmedChange::Archived),
        (14, ConfirmedChange::Tombstoned),
    ] {
        let visible = matches!(change, ConfirmedChange::Restored);
        apply_position(
            &mut index,
            &archive,
            position,
            vec![(message_event.clone(), change)],
        );
        write_local(&mut index, &archive, original.clone());
        assert_eq!(
            shown_text(&archive, &message).as_deref(),
            visible.then_some("locally parsed original"),
            "position {position}"
        );
        assert_eq!(
            committed(),
            visible.then_some(fixture_commit().sequence as u64)
        );
    }
}
