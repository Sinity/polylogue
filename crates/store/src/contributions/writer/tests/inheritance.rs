use super::*;

#[test]
fn inheritance_shares_roots_and_rejects_replay_or_wrong_state() {
    let mut f = Fixture::new();
    let old = f.prepare().unwrap().build;
    let slot = Slot::Message(MessageField::At);
    f.json(&old, ValuePurpose::Normalized, b"1");
    f.writer
        .write_message_field(&old, &slot, None, true, &step(0, b"field"), 4096)
        .unwrap();
    f.writer
        .finish_message_projection(
            &old,
            &step(1, b"finished"),
            BlobHash::of(b"synthetic-evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
    f.select_prepared_body();
    f.exec("DELETE FROM message_build");
    let reader = f.writer.read_handle().unwrap();
    let pinned = reader.selected_message(&f.message).unwrap().unwrap();
    let first = f.head.clone();
    f.advance_parser();
    let new = f
        .prepare_with(|_, s| s.evidence_revision = 2)
        .unwrap()
        .build;
    let facts = f.scalar("SELECT count(*) FROM message_field_fact");
    let mut wrong = first.token.clone();
    wrong.generation += 1;
    let inherited = step(0, b"inherited");
    assert!(matches!(
        f.writer
            .inherit_message_projection(&new, &old.body_id(), &wrong, &inherited, 4096),
        Err(Error::StaleMessage)
    ));
    f.exec("CREATE TRIGGER fail_cursor BEFORE UPDATE OF source_cursor ON message_build BEGIN SELECT RAISE(ABORT,'synthetic cursor failure'); END;");
    assert!(
        f.writer
            .inherit_message_projection(&new, &old.body_id(), &first.token, &inherited, 4096)
            .is_err()
    );
    assert_eq!(f.sequence(&new), 0);
    f.exec("DROP TRIGGER fail_cursor");
    let result = f
        .writer
        .inherit_message_projection(&new, &old.body_id(), &first.token, &inherited, 4096)
        .unwrap();
    assert_eq!(result.evidence_revision, 1);
    assert_eq!(result.version_hash, BlobHash::of(b"synthetic-evidence"));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), facts);
    assert!(
        f.writer
            .inherit_message_projection(&new, &old.body_id(), &first.token, &inherited, 4096)
            .is_err()
    );
    f.json(&new, ValuePurpose::Normalized, b"2");
    f.writer
        .set_message_field(&new, &slot, None, true, &step(1, b"changed"), 4096)
        .unwrap();
    match pinned.field(&slot, 4096).unwrap().unwrap() {
        FieldRead::Value(field) => assert_eq!(field.range(0, 64).unwrap().bytes, b"1"),
        _ => panic!("unexpected conflict"),
    }
    // Retiring the new preparation must not affect the selected old body.
    f.writer.retire_message_preparation(&new).unwrap();
    f.retire_until_complete(&new);
    assert!(pinned.field(&slot, 4096).unwrap().is_some());
}
