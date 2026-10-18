use super::*;
use crate::contributions::{ProjectionStep, SmallField};
mod construction;
fn payload() -> SmallField {
    SmallField::prepare(b"null".to_vec(), b"null".to_vec(), 65536).unwrap()
}
fn slot(key: &str) -> Slot {
    Slot::Extra {
        block: None,
        key: key.into(),
    }
}
fn step(sequence: u64) -> ProjectionStep<'static> {
    ProjectionStep {
        expected_sequence: sequence,
        next_cursor: b"small-field-cursor",
    }
}
#[test]
fn small_field_same_value_origin_dedup_and_cursor_commit_survive_reopen() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            4096,
        )
        .unwrap();
    assert_eq!(
        f.scalar("SELECT count(*) FROM body_value WHERE state='ready'"),
        1
    );
    assert_eq!(
        f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
        2
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_field_fact WHERE value_pk=origin_value_pk"),
        1
    );
    let mut f = f.reopen();
    assert_eq!(
        f.writer.message_projection(&build, 4096).unwrap().sequence,
        1
    );
    let before = f.scalar("SELECT count(*) FROM body_value");
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("two"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            4096
        ),
        Err(Error::MessagePredecessorChanged)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), before);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
}
#[test]
fn small_field_late_cursor_failure_rolls_back_chunks_leases_and_field_tree() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer._index.execute_batch("CREATE TRIGGER refuse_small_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic crash boundary'); END;").unwrap();
    assert!(
        f.writer
            .write_small_field(
                &build,
                &slot("one"),
                None,
                true,
                false,
                &payload(),
                &step(0),
                4096
            )
            .is_err()
    );
    f.writer
        ._index
        .execute_batch("DROP TRIGGER refuse_small_cursor;")
        .unwrap();
    let f = f.reopen();
    for table in [
        "body_value",
        "body_value_chunk",
        "message_field_fact",
        "message_body_node",
    ] {
        assert_eq!(f.scalar(&format!("SELECT count(*) FROM {table}")), 0);
    }
    assert_eq!(
        f.writer.message_projection(&build, 4096).unwrap().sequence,
        0
    );
}
#[test]
fn small_field_admission_and_existing_producer_refuse_before_mutation() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            1
        ),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    f.writer
        .begin_message_value(&build, values::Codec::Json, 4)
        .unwrap();
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            4096
        ),
        Err(Error::MessagePreparationPending)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert!(SmallField::prepare(vec![0xff], b"null".to_vec(), 65536).is_err());
    assert!(SmallField::prepare(vec![b' '; 9], b"null".to_vec(), 8).is_err());
}
#[test]
fn small_field_stale_source_cannot_allocate_values() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .begin_parser(&parser_state::Basis {
            input_pk: f.head.token.input_pk,
            chain: f.head.token.chain,
            parser_version: 3,
            digest: [8; 32],
            acquired_upto: f.head.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("one"),
            None,
            true,
            false,
            &payload(),
            &step(0),
            4096
        ),
        Err(Error::MessagePredecessorChanged)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
}

mod runs;
