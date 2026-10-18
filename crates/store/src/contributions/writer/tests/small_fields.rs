use super::*;
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
#[test]
fn small_field_dedups_value_and_origin_and_survives_reopen() {
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
            &step(0, b"small-field-cursor"),
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
    assert_eq!(f.sequence(&build), 1);
    let before = f.scalar("SELECT count(*) FROM body_value");
    assert!(matches!(
        f.writer.write_small_field(
            &build,
            &slot("two"),
            None,
            true,
            false,
            &payload(),
            &step(0, b"small-field-cursor"),
            4096
        ),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), before);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
}
#[test]
fn small_field_cursor_failure_rolls_back_everything() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.exec("CREATE TRIGGER refuse_small_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic crash boundary'); END;");
    assert!(
        f.writer
            .write_small_field(
                &build,
                &slot("one"),
                None,
                true,
                false,
                &payload(),
                &step(0, b"small-field-cursor"),
                4096
            )
            .is_err()
    );
    f.exec("DROP TRIGGER refuse_small_cursor;");
    let f = f.reopen();
    for table in [
        "body_value",
        "body_value_chunk",
        "message_field_fact",
        "message_body_node",
    ] {
        assert_eq!(f.scalar(&format!("SELECT count(*) FROM {table}")), 0);
    }
    assert_eq!(f.sequence(&build), 0);
}
#[test]
fn small_field_rejects_oversize_or_active_value_before_writing() {
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
            &step(0, b"small-field-cursor"),
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
            &step(0, b"small-field-cursor"),
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
            &step(0, b"small-field-cursor"),
            4096
        ),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
}

mod runs;
