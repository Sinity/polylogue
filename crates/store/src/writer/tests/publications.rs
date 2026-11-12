use super::*;
use crate::primary::{ParsedSource, PublicationReceipt};

fn publication(frame: &str, number: i64) -> PublicationReceipt {
    PublicationReceipt {
        producer: "fixture-producer".into(),
        frame: frame.into(),
        number,
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
