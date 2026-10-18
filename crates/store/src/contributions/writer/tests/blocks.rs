use super::*;
#[test]
fn failed_text_block_checkpoint_rolls_back_and_resumes() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.value(&build, b"hello");
    f.exec("CREATE TRIGGER refuse_projection BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic checkpoint failure'); END;");
    let step = step(0, b"next");
    assert!(matches!(
        f.writer.write_block(
            &build,
            0,
            0,
            BlockKind::Text,
            presence(true, true),
            &step,
            4096
        ),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_body_node"), 0);
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    assert_eq!(f.sequence(&build), 0);
    f.exec("DROP TRIGGER refuse_projection");
    let mut f = f.reopen();
    f.writer
        .write_block(
            &build,
            0,
            0,
            BlockKind::Text,
            presence(true, true),
            &step,
            4096,
        )
        .unwrap();
    let state = f.writer.message_projection(&build, 4096).unwrap();
    assert_eq!(state.sequence, 1);
    assert_eq!(state.cursor.as_deref(), Some(b"next".as_slice()));
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    assert_eq!(f.scalar("SELECT ready FROM message_body"), 0);
    assert!(matches!(
        f.writer.advance_message_projection(&build, &step, 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.writer.message_projection(&build, 4096).unwrap(), state);
    f.writer.retire_message_preparation(&build).unwrap();
    for turn in 0..200 {
        let nodes = f.scalar("SELECT count(*) FROM message_body_node");
        let result = f.writer.retire_message_turn(&build).unwrap();
        if result == Retirement::Node {
            assert_eq!(
                f.scalar("SELECT count(*) FROM message_body_node"),
                nodes - 1
            );
        }
        if result == Retirement::Complete {
            break;
        }
        assert!(turn < 199);
    }
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
    assert_eq!(
        f.scalar("SELECT count(*) FROM parser_cell WHERE namespace='message.projection.v1'"),
        0
    );
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 0);
}
#[test]
fn replacing_text_blocks_collects_only_unreferenced_leaves() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    for n in 0..3 {
        f.value(&build, b"shared");
        f.writer
            .write_block(
                &build,
                0,
                n % 2,
                BlockKind::Text,
                presence(true, true),
                &step(n, b"cursor"),
                4096,
            )
            .unwrap();
    }
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 3);
    for turn in 0..200 {
        let nodes = f.scalar("SELECT count(*) FROM message_body_node");
        if !f.writer.collect_message_nodes(&f.message).unwrap() {
            break;
        }
        assert_eq!(
            f.scalar("SELECT count(*) FROM message_body_node"),
            nodes - 1
        );
        assert!(turn < 199);
    }
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    f.value(&build, b"");
    assert!(matches!(
        f.writer.write_block(
            &build,
            1,
            0,
            BlockKind::Text,
            presence(true, false),
            &step(3, b"invalid"),
            4096
        ),
        Err(Error::Invariant(_))
    ));
    assert_eq!(f.sequence(&build), 3);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
}
#[test]
fn staged_thinking_block_survives_restart_and_counts_source_fields() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .begin_block(
            &build,
            4,
            91,
            BlockKind::Thinking { encrypted: None },
            presence(true, false),
            &step(0, b"staged"),
            4096,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
    let mut f = f.reopen();
    assert_eq!(f.prepare().unwrap().build, build);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_build WHERE active_block_pk IS NOT NULL"),
        1
    );
    assert!(!f.writer.collect_message_nodes(&f.message).unwrap());
    f.value(&build, b"false");
    let slot = Slot::Block {
        block: BlockSlot::Item(42),
        field: BlockField::ThinkingData,
    };
    assert!(matches!(
        f.writer
            .write_block_field(&build, &slot, None, true, &step(0, b"bad"), 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert!(f.prepare().unwrap().active_value.is_some());
    assert!(matches!(
        f.writer.install_block(&build, &step(1, b"bad"), 4096),
        Err(Error::ValueNotReady)
    ));
    let descriptor = b"{\"codec\":\"synthetic-unvalidated\"}";
    f.json(&build, ValuePurpose::Origin, descriptor);
    f.writer
        .write_block_field(&build, &slot, None, true, &step(1, b"field"), 4096)
        .unwrap();
    assert!(matches!(
        f.writer
            .write_block_field(&build, &slot, None, true, &step(1, b"field"), 4096),
        Err(Error::ValueNotReady)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    let signature = Slot::Block {
        block: BlockSlot::Item(0),
        field: BlockField::ThinkingSignature,
    };
    f.json(&build, ValuePurpose::Normalized, b"\"opaque-signature\"");
    let signature_step = step(2, b"signature");
    assert!(matches!(
        f.writer
            .write_block_field(&build, &signature, None, true, &signature_step, 4096),
        Err(Error::Invariant(_))
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    f.writer
        .write_block_field(&build, &signature, None, false, &signature_step, 4096)
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 2);
    f.writer
        .install_block(&build, &step(3, b"installed"), 4096)
        .unwrap();
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_build WHERE active_block_pk IS NOT NULL"),
        0
    );
    f.writer
        .finish_message_projection(
            &build,
            &step(4, b"done"),
            polylogue_model::BlobHash::of(b"evidence"),
            polylogue_model::merge::FinishState::Unknown,
            4096,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT source_fields FROM message_candidate"), 1);
    f.select_prepared_body();
    let reader = f.writer.read_handle().unwrap();
    let view = reader.selected_message(&f.message).unwrap().unwrap();
    let block = view.block(BlockSlot::Item(0)).unwrap().unwrap();
    assert_eq!(block.encrypted, None);
    assert!(block.text_range(0, 10).unwrap().is_none());
    let actual = Slot::Block {
        block: BlockSlot::Item(0),
        field: BlockField::ThinkingData,
    };
    match block.field(&actual, 4096).unwrap().unwrap() {
        FieldRead::Value(field) => {
            assert_eq!(field.range(0, 100).unwrap().bytes, b"false");
            assert_eq!(
                field.origin_range(0, 100).unwrap().unwrap().bytes,
                descriptor
            );
        }
        _ => panic!("unexpected block field conflict"),
    }
    assert!(matches!(
        view.field(&actual, 4096).unwrap(),
        Some(FieldRead::Value(_))
    ));
    assert!(block.field(&slot, 4096).is_err());
    match block.field(&signature, 4096).unwrap().unwrap() {
        FieldRead::Value(field) => {
            assert!(!field.supplied);
            assert_eq!(field.range(0, 100).unwrap().bytes, b"\"opaque-signature\"");
        }
        _ => panic!("unexpected signature conflict"),
    }
}
#[test]
fn blocks_share_field_keys_and_abort_releases_only_the_staged_block() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    for n in 0..3 {
        f.value(&build, b"shared text");
        f.writer
            .begin_block(
                &build,
                0,
                n,
                BlockKind::Thinking {
                    encrypted: Some(false),
                },
                presence(true, true),
                &step(n * 3, b"begin"),
                4096,
            )
            .unwrap();
        f.value(&build, if n == 0 { b"field zero" } else { b"field one!" });
        f.writer
            .write_block_field(
                &build,
                &Slot::Block {
                    block: BlockSlot::Item(n as u32),
                    field: BlockField::ThinkingData,
                },
                None,
                true,
                &step(n * 3 + 1, b"field"),
                4096,
            )
            .unwrap();
        if n < 2 {
            f.writer
                .install_block(&build, &step(n * 3 + 2, b"install"), 4096)
                .unwrap();
        }
    }
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 3);
    assert_eq!(
        f.scalar("SELECT count(DISTINCT value_pk) FROM message_field_fact"),
        2
    );
    assert_eq!(f.scalar("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.block_root_pk"),2);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 3);
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 2);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
    assert_eq!(
        f.scalar("SELECT count(*) FROM body_value WHERE state='ready'"),
        3
    );
    f.retire_until_complete(&build);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
}
#[test]
fn empty_text_block_keeps_fields_but_counts_nothing() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.value(&build, b"");
    f.writer
        .begin_block(
            &build,
            0,
            0,
            BlockKind::Text,
            BlockPresence::default(),
            &step(0, b"empty"),
            4096,
        )
        .unwrap();
    assert!(matches!(
        f.writer.finish_message_projection(
            &build,
            &step(1, b"premature"),
            polylogue_model::BlobHash::of(b"evidence"),
            polylogue_model::merge::FinishState::Unknown,
            4096
        ),
        Err(Error::ValueNotReady)
    ));
    f.value(&build, b"false");
    let slot = Slot::Extra {
        block: Some(BlockSlot::Item(0)),
        key: String::new(),
    };
    assert!(matches!(
        f.writer
            .write_message_field(&build, &slot, None, false, &step(1, b"wrong owner"), 4096),
        Err(Error::Invariant(_))
    ));
    assert!(matches!(
        f.writer
            .write_block_field(&build, &slot, None, true, &step(1, b"wrong presence"), 4096),
        Err(Error::Invariant(_))
    ));
    f.writer
        .write_block_field(&build, &slot, None, false, &step(1, b"metadata"), 4096)
        .unwrap();
    f.writer
        .install_block(&build, &step(2, b"installed"), 4096)
        .unwrap();
    f.writer
        .finish_message_projection(
            &build,
            &step(3, b"done"),
            polylogue_model::BlobHash::of(b"evidence"),
            polylogue_model::merge::FinishState::Unknown,
            4096,
        )
        .unwrap();
    assert_eq!(
        f.scalar("SELECT source_blocks+source_text_blocks+source_fields FROM message_candidate"),
        0
    );
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
}
#[test]
fn pending_origin_value_blocks_block_install() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.writer
        .begin_block(
            &build,
            0,
            0,
            BlockKind::Thinking { encrypted: None },
            BlockPresence::default(),
            &step(0, b"block"),
            4096,
        )
        .unwrap();
    let origin = f.json(&build, ValuePurpose::Origin, b"null");
    let step = step(1, b"install");
    assert!(matches!(
        f.writer.install_block(&build, &step, 4096),
        Err(Error::MessagePreparationPending)
    ));
    assert!(matches!(
        f.writer.write_block(
            &build,
            1,
            0,
            BlockKind::Thinking { encrypted: None },
            BlockPresence::default(),
            &step,
            4096
        ),
        Err(Error::MessagePreparationPending)
    ));
    let resumed = f.prepare().unwrap();
    assert_eq!(f.sequence(&build), 1);
    assert_eq!(resumed.active_origin_value.unwrap().id, origin.id);
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.value_head(&origin.id).unwrap().state,
        values::State::Retiring
    );
}
