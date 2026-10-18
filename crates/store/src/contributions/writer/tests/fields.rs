use super::*;
#[test]
fn value_and_field_survive_restart_unpublished() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_candidate WHERE ready=1 OR eligible=1"),
        0
    );
    let value = f
        .writer
        .begin_message_value(&build, values::Codec::Utf8, 5)
        .unwrap();
    f.writer
        .append_value_chunk(&value.id, 0, b"hel", 3)
        .unwrap();
    let mut f = f.reopen();
    let resumed = f.prepare().unwrap();
    assert_eq!(resumed.build, build);
    let active = resumed.active_value.unwrap();
    assert_eq!(active.appended, 3);
    assert_eq!(active.id, value.id);
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    f.writer
        .append_value_chunk(&active.id, 3, b"lo", 2)
        .unwrap();
    let reader = f.writer.read_handle().unwrap();
    let mut proof = f.writer.validate_value(&active.id).unwrap();
    while !proof.progress().complete {
        proof.advance(&reader, 1).unwrap();
    }
    drop(reader);
    f.writer.seal_message_value(&build, &proof).unwrap();
    f.writer
        .write_message_field(
            &build,
            &Slot::Message(MessageField::Model),
            None,
            true,
            &step(f.sequence(&build), b"field"),
            4096,
        )
        .unwrap();
    assert!(f.prepare().unwrap().active_value.is_none());
    assert_eq!(f.scalar("SELECT incoming FROM body_value"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message_body WHERE ready=1"),
        0
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    // The first value now belongs to its fact, so a new one may start.
    f.writer
        .begin_message_value(&build, values::Codec::Json, 2)
        .unwrap();
    assert_eq!(f.scalar("SELECT sum(incoming) FROM body_value"), 2);
}
#[test]
fn duplicate_value_moves_only_its_own_lease() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let seal = f.value(&build, b"same");
    f.writer
        .write_message_field(
            &build,
            &Slot::Message(MessageField::Model),
            None,
            true,
            &step(f.sequence(&build), b"field"),
            4096,
        )
        .unwrap();
    let duplicate = f.value(&build, b"same");
    assert_eq!(duplicate.ready, seal.ready);
    assert!(duplicate.retired_duplicate.is_some());
    assert_eq!(
        f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
        2
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM body_value WHERE state='retiring' AND incoming=0"),
        1
    );
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
        1
    );
    assert_eq!(
        f.scalar(
            "SELECT count(*) FROM message_build WHERE phase='retiring' AND active_value_pk IS NULL"
        ),
        1
    );
    assert!(matches!(
        f.writer.begin_message_value(&build, values::Codec::Utf8, 1),
        Err(Error::StaleMessage)
    ));
    // Retrying invalidation releases no second lease.
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.scalar("SELECT incoming FROM body_value WHERE state='ready'"),
        1
    );
}
#[test]
fn oversized_or_duplicate_field_keeps_the_active_value() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.value(&build, b"model");
    let slot = Slot::Extra {
        block: None,
        key: "x".repeat(1000),
    };
    assert!(matches!(
        f.writer.write_message_field(
            &build,
            &slot,
            None,
            true,
            &step(f.sequence(&build), b"field"),
            8
        ),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert!(f.prepare().unwrap().active_value.is_some());
    let slot = Slot::Message(MessageField::Model);
    f.writer
        .write_message_field(
            &build,
            &slot,
            None,
            true,
            &step(f.sequence(&build), b"field"),
            4096,
        )
        .unwrap();
    f.value(&build, b"other-model");
    assert!(matches!(
        f.writer.write_message_field(
            &build,
            &slot,
            None,
            true,
            &step(f.sequence(&build), b"field"),
            4096
        ),
        Err(Error::Invariant(_))
    ));
    assert!(f.prepare().unwrap().active_value.is_some());
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
}
/// Selects a ready copy of the body that shares its field root, through a
/// separate ready candidate, so the preparation itself stays writable.
fn select_copy(f: &mut Fixture, build: &Build) {
    let path = f.index_path();
    let owner = d::Owner {
        message: build.message,
        session: build.session,
    };
    let tx = f.writer._index.transaction().unwrap();
    let source_root: Option<i64> = tx
        .query_row(
            "SELECT field_root_pk FROM message_body WHERE pk=?1",
            [build.body],
            |r| r.get(0),
        )
        .unwrap();
    let source_fields = d::summary(&tx, &path, owner, d::Kind::Fields, source_root)
        .unwrap()
        .source_fields;
    tx.execute("INSERT INTO message_candidate(message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,source_blocks,source_text_blocks,source_fields,version_hash,facts_ready,ready) SELECT message_pk,session_pk,input_pk,chain,revision,revision_length,evidence_revision,logical_key,parser_version,source_state_generation,source_state_step,source_basis_digest,basis_digest,finish,route,0,0,?2,zeroblob(32),1,1 FROM message_candidate WHERE pk=?1",params![build.candidate,source_fields]).unwrap();
    let certificate = tx.last_insert_rowid();
    tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,1,zeroblob(32),contributor_revision,removal_epoch,?2,field_root_pk,layout,role,written_by,input_pk,zeroblob(32),parser_version FROM message_body WHERE pk=?1",params![build.body,certificate]).unwrap();
    let selected = tx.last_insert_rowid();
    let root: i64 = tx
        .query_row(
            "SELECT field_root_pk FROM message_body WHERE pk=?1",
            [selected],
            |r| r.get(0),
        )
        .unwrap();
    d::acquire(&tx, &path, owner, root).unwrap();
    tx.execute("UPDATE message SET selected_body_pk=?1", [selected])
        .unwrap();
    tx.commit().unwrap();
}
#[test]
fn set_and_remove_field_keep_the_pinned_old_value() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let slot = Slot::Message(MessageField::Model);
    f.value(&build, b"old");
    f.writer
        .write_message_field(&build, &slot, None, true, &step(0, b"insert"), 4096)
        .unwrap();
    select_copy(&mut f, &build);
    let reader = f.writer.read_handle().unwrap();
    let old = reader.selected_message(&f.message).unwrap().unwrap();
    f.value(&build, b"new");
    f.exec("CREATE TRIGGER refuse_field_cursor BEFORE UPDATE ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic checkpoint failure'); END;");
    assert!(matches!(
        f.writer
            .set_message_field(&build, &slot, None, false, &step(1, b"replace"), 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    assert!(f.prepare().unwrap().active_value.is_some());
    f.exec("DROP TRIGGER refuse_field_cursor");
    assert!(matches!(
        f.writer
            .set_message_field(&build, &slot, None, false, &step(0, b"stale"), 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    f.writer
        .set_message_field(&build, &slot, None, false, &step(1, b"replace"), 4096)
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 2);
    assert_eq!(f.scalar(&format!("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.field_root_pk WHERE b.pk={}",build.body)),0);
    match old.field(&slot, 4096).unwrap().unwrap() {
        FieldRead::Value(v) => assert_eq!(v.range(0, 100).unwrap().bytes, b"old"),
        _ => panic!("unexpected conflict"),
    }
    select_copy(&mut f, &build);
    let newer = reader.selected_message(&f.message).unwrap().unwrap();
    match newer.field(&slot, 4096).unwrap().unwrap() {
        FieldRead::Value(v) => {
            assert!(!v.supplied);
            assert_eq!(v.range(0, 100).unwrap().bytes, b"new")
        }
        _ => panic!("unexpected conflict"),
    }
    f.writer
        .remove_message_field(&build, &slot, &step(2, b"removed"), 4096)
        .unwrap();
    assert_eq!(f.scalar(&format!("SELECT n.source_fields FROM message_body b JOIN message_body_node n ON n.pk=b.field_root_pk WHERE b.pk={}",build.body)),0);
    // Both pinned bodies still read their own value.
    assert!(old.field(&slot, 4096).unwrap().is_some());
    assert!(newer.field(&slot, 4096).unwrap().is_some());
    let slots = f.scalar("SELECT count(*) FROM message_field_slot");
    f.writer
        .remove_message_field(
            &build,
            &Slot::Extra {
                block: None,
                key: "never supplied".into(),
            },
            &step(3, b"absent"),
            4096,
        )
        .unwrap();
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), slots);
    assert!(matches!(
        f.writer
            .remove_message_field(&build, &slot, &step(2, b"replay"), 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.sequence(&build), 4);
}
#[test]
fn set_inserts_and_remove_refuses_while_a_value_is_active() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let slot = Slot::Message(MessageField::Model);
    f.value(&build, b"model");
    assert!(matches!(
        f.writer
            .remove_message_field(&build, &slot, &step(0, b"lost"), 4096),
        Err(Error::MessagePreparationPending)
    ));
    assert!(f.prepare().unwrap().active_value.is_some());
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 0);
    f.writer
        .set_message_field(&build, &slot, None, true, &step(0, b"new"), 4096)
        .unwrap();
    f.writer
        .remove_message_field(&build, &slot, &step(1, b"gone"), 4096)
        .unwrap();
    f.writer
        .remove_message_field(&build, &slot, &step(2, b"still gone"), 4096)
        .unwrap();
    assert_eq!(f.sequence(&build), 3);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
}
#[test]
fn value_and_origin_resume_and_commit_in_one_cursor_step() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let normalized = f.json(&build, ValuePurpose::Normalized, b"false");
    let origin = f.writer.begin_message_origin_value(&build, 2).unwrap();
    f.writer.append_value_chunk(&origin.id, 0, b"{", 1).unwrap();
    let mut f = f.reopen();
    let resumed = f.prepare().unwrap();
    assert_eq!(resumed.active_value.unwrap().id, normalized.id);
    let resumed_origin = resumed.active_origin_value.unwrap();
    assert_eq!(resumed_origin.id, origin.id);
    assert_eq!(resumed_origin.appended, 1);
    let slot = Slot::Message(MessageField::Model);
    let first = step(0, b"field");
    assert!(matches!(
        f.writer
            .write_message_field(&build, &slot, None, true, &first, 4096),
        Err(Error::ValueNotReady)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    f.writer.append_value_chunk(&origin.id, 1, b"}", 1).unwrap();
    let descriptor = f.finish_json(&build, ValuePurpose::Origin, &origin.id);
    f.exec("CREATE TRIGGER refuse_origin_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'origin checkpoint failure'); END;");
    assert!(matches!(
        f.writer
            .write_message_field(&build, &slot, None, true, &first, 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    let resumed = f.prepare().unwrap();
    assert_eq!(resumed.active_value.unwrap().id, normalized.id);
    assert_eq!(resumed.active_origin_value.unwrap().id, descriptor.id);
    f.exec("DROP TRIGGER refuse_origin_cursor");
    assert!(matches!(
        f.writer
            .write_message_field(&build, &slot, None, true, &step(1, b"stale"), 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    f.writer
        .write_message_field(&build, &slot, None, true, &first, 4096)
        .unwrap();
    let resumed = f.prepare().unwrap();
    assert!(resumed.active_value.is_none());
    assert!(resumed.active_origin_value.is_none());
    assert_eq!(
        f.scalar("SELECT origin_value_pk FROM message_field_fact"),
        descriptor.id.pk()
    );
    assert_eq!(
        f.scalar("SELECT value_pk FROM message_field_fact"),
        normalized.id.pk()
    );
    assert_eq!(f.sequence(&build), 1);
}
#[test]
fn identical_value_and_origin_hold_two_leases() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let normalized = f.json(&build, ValuePurpose::Normalized, b"null");
    let origin = f.json(&build, ValuePurpose::Origin, b"null");
    assert_eq!(normalized.id, origin.id);
    assert_eq!(
        f.scalar(&format!(
            "SELECT incoming FROM body_value WHERE pk={}",
            origin.id.pk()
        )),
        2
    );
    f.writer
        .write_message_field(
            &build,
            &Slot::Message(MessageField::Model),
            None,
            true,
            &step(0, b"pair"),
            4096,
        )
        .unwrap();
    assert_eq!(
        f.scalar("SELECT value_pk=origin_value_pk FROM message_field_fact"),
        1
    );
    assert_eq!(
        f.scalar(&format!(
            "SELECT incoming FROM body_value WHERE pk={}",
            origin.id.pk()
        )),
        2
    );
    f.writer.retire_message_preparation(&build).unwrap();
    f.retire_until_complete(&build);
    assert_eq!(
        f.scalar(&format!(
            "SELECT incoming FROM body_value WHERE pk={}",
            origin.id.pk()
        )),
        0
    );
    assert_eq!(
        f.writer.value_head(&origin.id).unwrap().state,
        values::State::Retiring
    );
    f.collect_values();
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), 0);
}
#[test]
fn pending_origin_value_blocks_other_projection_steps() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let origin = f.json(&build, ValuePurpose::Origin, b"null");
    let step = step(0, b"skip");
    assert!(matches!(
        f.writer.advance_message_projection(&build, &step, 4096),
        Err(Error::MessagePreparationPending)
    ));
    assert!(matches!(
        f.writer
            .set_message_layout(&build, MessageLayout::Flat, &step, 4096),
        Err(Error::MessagePreparationPending)
    ));
    assert!(matches!(
        f.writer.begin_block(
            &build,
            0,
            0,
            BlockKind::Thinking { encrypted: None },
            BlockPresence::default(),
            &step,
            4096
        ),
        Err(Error::MessagePreparationPending)
    ));
    assert!(matches!(
        f.writer
            .remove_message_field(&build, &Slot::Message(MessageField::Model), &step, 4096),
        Err(Error::MessagePreparationPending)
    ));
    assert!(matches!(
        f.writer.finish_message_projection(
            &build,
            &step,
            polylogue_model::BlobHash::of(b"synthetic"),
            polylogue_model::merge::FinishState::Unknown,
            4096
        ),
        Err(Error::ValueNotReady)
    ));
    assert_eq!(f.sequence(&build), 0);
    let normal = f.json(&build, ValuePurpose::Normalized, b"null");
    assert_eq!(normal.id, origin.id);
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.scalar(&format!(
            "SELECT incoming FROM body_value WHERE pk={}",
            origin.id.pk()
        )),
        0
    );
    assert_eq!(
        f.writer.value_head(&origin.id).unwrap().state,
        values::State::Retiring
    );
}
#[test]
fn replaced_field_keeps_shared_origin_until_last_fact_drains() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let slot = Slot::Message(MessageField::Model);
    f.json(&build, ValuePurpose::Normalized, b"\"first\"");
    let origin = f.json(
        &build,
        ValuePurpose::Origin,
        b"{\"codec\":\"synthetic-unvalidated\"}",
    );
    f.writer
        .write_message_field(&build, &slot, None, true, &step(0, b"first"), 4096)
        .unwrap();
    f.json(&build, ValuePurpose::Normalized, b"\"second\"");
    let again = f.json(
        &build,
        ValuePurpose::Origin,
        b"{\"codec\":\"synthetic-unvalidated\"}",
    );
    assert_eq!(origin.id, again.id);
    f.writer
        .set_message_field(&build, &slot, None, true, &step(1, b"second"), 4096)
        .unwrap();
    assert_eq!(
        f.scalar(&format!(
            "SELECT incoming FROM body_value WHERE pk={}",
            origin.id.pk()
        )),
        2
    );
    f.writer.retire_message_preparation(&build).unwrap();
    let mut facts = 0;
    loop {
        let result = f.writer.retire_message_turn(&build).unwrap();
        if result == Retirement::Fact {
            facts += 1;
            let head = f.writer.value_head(&origin.id).unwrap();
            if facts == 1 {
                assert_eq!(head.state, values::State::Ready);
                assert_eq!(
                    f.scalar(&format!(
                        "SELECT incoming FROM body_value WHERE pk={}",
                        origin.id.pk()
                    )),
                    1
                );
            } else {
                assert_eq!(head.state, values::State::Retiring);
            }
        }
        if result == Retirement::Complete {
            break;
        }
    }
    assert_eq!(facts, 2);
}
