use super::*;
use crate::primary::{ParsedSource, PublicationReceipt};

fn publication(frame: &str, number: i64) -> PublicationReceipt {
    PublicationReceipt {
        producer: "fixture-producer".into(),
        frame: frame.into(),
        number,
        commit_seq: Some(number + 100),
        root_material: "00000000-0000-0000-0000-000000000001".into(),
    }
}

fn interpretation(text: &str, version: &str) -> Batch {
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
    batch.parse[0].reading_version = version.into();
    batch.parse[0].prepare();
    batch
}

fn apply(index: &mut Connection, mut batch: Batch, publication: &PublicationReceipt) -> Result<()> {
    super::super::source_eligibility::retain_source_records(
        index,
        &batch.input,
        1,
        batch.length,
        &mut batch.parse[0],
    )?;
    resolve::prepare_index(index, &mut batch, 1, false)?;
    let tx = index.transaction()?;
    let counter = bump_counter(&tx)?;
    let changes = crate::change_summary::ChangeSummary::start(&tx)?;
    let step = &batch.parse[0];
    rows::write_published_rows(
        &tx,
        &batch,
        step,
        step.resolution.as_ref().unwrap(),
        counter,
        None,
        &fixture_receipt(),
        publication,
        true,
        &changes,
    )?;
    rows::finish(&tx, &changes, false)?;
    tx.commit()?;
    Ok(())
}

fn source(batch: &Batch) -> ParsedSource {
    ParsedSource {
        root_material: publication("unused", 1).root_material,
        reading: crate::primary::Reading::from_step(
            &batch.input,
            1,
            Some(batch.length),
            &batch.parse[0],
        ),
    }
}

fn text(index: &Connection) -> String {
    let hash: Vec<u8> = index
        .query_row("SELECT version_hash FROM message_source", [], |row| {
            row.get(0)
        })
        .unwrap();
    crate::versions::read_message(index, &hash)
        .unwrap()
        .unwrap()
        .0
        .text()
}

#[test]
fn native_ack_does_not_cover_a_rolled_back_interpretation_at_an_old_cursor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.db");
    let mut index = Connection::open(&path).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let original = interpretation("original interpretation", "synthetic=1");
    let first = publication("first-frame", 11);
    apply(&mut index, original.clone(), &first).unwrap();
    assert_eq!(text(&index), "original interpretation");
    let replacement = interpretation("new interpretation", "synthetic=2");
    let second = publication("second-frame", 12);
    let processed = |index: &Connection, batch: &Batch, receipt: &PublicationReceipt| {
        crate::primary::projection::processed_source(index, &source(batch), receipt).unwrap()
    };
    assert!(processed(&index, &original, &first));
    assert!(!processed(&index, &replacement, &second));
    index.execute_batch("CREATE TRIGGER reject_checkpoint BEFORE INSERT ON parse_state WHEN new.reading_version='synthetic=2' BEGIN SELECT raise(ABORT,'fixture interruption'); END").unwrap();
    assert!(apply(&mut index, replacement.clone(), &second).is_err());
    assert_eq!(text(&index), "original interpretation");
    assert!(!processed(&index, &replacement, &second));
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    index
        .execute_batch("DROP TRIGGER reject_checkpoint")
        .unwrap();
    assert!(!processed(&index, &replacement, &second));
    apply(&mut index, replacement.clone(), &second).unwrap();
    assert_eq!(text(&index), "new interpretation");
    assert!(processed(&index, &replacement, &second));
    assert!(processed(&index, &original, &first));
    let mut wrong = second.clone();
    wrong.frame = "another-frame".into();
    assert!(!processed(&index, &replacement, &wrong));
    wrong = second.clone();
    wrong.root_material = "00000000-0000-0000-0000-000000000002".into();
    assert!(!processed(&index, &replacement, &wrong));
    wrong = second.clone();
    wrong.producer = "another-producer".into();
    assert!(!processed(&index, &replacement, &wrong));
}

#[test]
fn native_parser_diagnostic_lifecycle_preserves_the_local_model_and_processed_source() {
    use crate::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let mut index = Connection::open(directory.path().join("index.db")).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let original = interpretation("locally parsed original", "synthetic=1");
    let receipt = publication("original-frame", 11);
    apply(&mut index, original.clone(), &receipt).unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let diagnostic = Product::ParseProgress {
        reading: source(&original).reading,
        counts: Default::default(),
        reset: true,
        comparison: None,
        checkpoint: None,
        stopped_attempt: None,
        semantic_event_ids: None,
    };
    for (position, change) in [
        (1, ConfirmedChange::Live(Box::new(diagnostic))),
        (2, ConfirmedChange::Archived),
        (3, ConfirmedChange::Restored),
        (4, ConfirmedChange::Tombstoned),
    ] {
        crate::primary::projection::apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "diagnostic".into(),
                    change,
                },
                Confirmed {
                    native: None,
                    position: 0,
                    output_index: 0,
                    event: String::new(),
                    change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
                        position,
                        delivery: None,
                    })),
                },
            ],
        )
        .unwrap();
        assert_eq!(text(&index), "locally parsed original");
        assert!(
            crate::primary::projection::processed_source(&index, &source(&original), &receipt)
                .unwrap()
        );
    }
}

#[test]
fn an_archived_value_does_not_turn_a_processed_incomplete_tail_into_an_unknown_prefix() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let mut index = Connection::open(directory.path().join("index.db")).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let mut original = interpretation("archived original", "synthetic=1");
    original.parse[0].parsed_upto = 90;
    let message = crate::primary::products::parse(
        &original.input,
        1,
        Some(original.length),
        &original.parse[0],
    )
    .unwrap()
    .into_iter()
    .find(|product| matches!(product, crate::primary::Product::Message { .. }))
    .unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    for (position, change) in [
        (1, ConfirmedChange::Live(Box::new(message))),
        (2, ConfirmedChange::Archived),
    ] {
        crate::primary::projection::apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "hidden-message".into(),
                    change,
                },
                Confirmed {
                    native: None,
                    position: 0,
                    output_index: 0,
                    event: String::new(),
                    change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
                        position,
                        delivery: None,
                    })),
                },
            ],
        )
        .unwrap();
    }
    let receipt = publication("processed-hidden-source", 12);
    apply(&mut index, original.clone(), &receipt).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .query_row(
                "SELECT status,parsed_upto,at_length FROM parse_state",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                }
            )
            .unwrap(),
        ("waiting".into(), 90, 100)
    );
    assert!(
        crate::primary::projection::processed_source(&index, &source(&original), &receipt).unwrap()
    );
}
