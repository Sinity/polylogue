use super::*;
#[test]
fn selection_carries_operation_root_into_a_body_copy() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let path = f.index_path();
    let owner = d::Owner {
        message: build.message,
        session: build.session,
    };
    let tx = f.writer._index.transaction().unwrap();
    // Setup: mark the body ready and give it an empty operations directory.
    tx.execute_batch("UPDATE message_body SET version_hash=zeroblob(32),ready=1; UPDATE message_candidate SET version_hash=zeroblob(32),facts_ready=1,source_blocks=0,source_text_blocks=0,source_fields=0,ready=1,body_ready=1,eligible=1;").unwrap();
    let (root, _) = d::replace(
        &tx,
        &path,
        owner,
        d::Kind::Operations,
        None,
        0,
        None,
        &mut d::Work::default(),
    )
    .unwrap();
    d::acquire(&tx, &path, owner, root).unwrap();
    tx.execute(
        "UPDATE message_body SET operation_root_pk=?2 WHERE pk=?1",
        params![build.body, root],
    )
    .unwrap();
    tx.execute("UPDATE message SET selected_body_pk=?1", [build.body])
        .unwrap();
    tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,layout,role,written_by,input_pk,version_hash,parser_version FROM message_body WHERE pk=?1",[build.body]).unwrap();
    let source = tx.last_insert_rowid();
    let derived =
        crate::contributions::tool_associations::carry_selected(&tx, &path, owner, source).unwrap();
    assert_ne!(derived, source);
    assert_ne!(derived, build.body);
    let roots: Vec<Option<i64>> = tx
        .prepare("SELECT operation_root_pk FROM message_body ORDER BY pk")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(roots, vec![Some(root), None, Some(root)]);
    assert_eq!(
        tx.query_row(
            "SELECT count(DISTINCT version_hash) FROM message_body",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        tx.query_row(
            "SELECT incoming FROM message_body_node WHERE pk=?1",
            [root],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    tx.execute("UPDATE message SET removal_epoch=removal_epoch+1", [])
        .unwrap();
    assert_eq!(
        crate::contributions::tool_associations::carry_selected(&tx, &path, owner, source).unwrap(),
        source
    );
    tx.commit().unwrap();
}
#[test]
fn selected_read_keeps_its_snapshot_until_the_message_changes() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let reader = f.writer.read_handle().unwrap();
    assert!(reader.selected_message(&f.message).unwrap().is_none());
    for (sequence, group, local, text) in [
        (0, 4096, 900, b"later".as_slice()),
        (1, 1, 17, b"earlier".as_slice()),
    ] {
        f.value(&build, text);
        f.writer
            .write_block(
                &build,
                group,
                local,
                BlockKind::Text,
                presence(true, true),
                &step(sequence, b"selected-read-test"),
                4096,
            )
            .unwrap();
    }
    let certificate = f
        .writer
        .finish_message_projection(
            &build,
            &step(2, b"projection-complete"),
            polylogue_model::BlobHash::of(b"synthetic-source-evidence"),
            polylogue_model::merge::FinishState::Unknown,
            4096,
        )
        .unwrap();
    assert_eq!(certificate.body_pk(), build.body_pk());
    drop(reader);
    let mut f = f.reopen();
    let reader = f.writer.read_handle().unwrap();
    assert_eq!(
        f.prepare().unwrap().projection.unwrap().version_hash(),
        certificate.version_hash()
    );
    assert_eq!(f.scalar("SELECT eligible FROM message_candidate"), 0);
    f.select_prepared_body();
    let view = reader.selected_message(&f.message).unwrap().unwrap();
    assert_eq!(view.id(), &f.message);
    let first = view.block(BlockSlot::Item(0)).unwrap().unwrap();
    assert_eq!((first.group, first.local), (1, 17));
    assert_eq!(first.text_range(0, 3).unwrap().unwrap().bytes, b"ear");
    let second = view.block(BlockSlot::Item(1)).unwrap().unwrap();
    assert_eq!((second.group, second.local), (4096, 900));
    assert_eq!(second.text_range(0, 100).unwrap().unwrap().bytes, b"later");
    assert!(view.block(BlockSlot::Item(2)).unwrap().is_none());
    assert!(view.block(BlockSlot::Reasoning(0)).unwrap().is_none());
    // Delete the body rows directly and release the root, then collect.
    let root = f.scalar("SELECT block_root_pk FROM message_body");
    let path = f.index_path();
    let tx = f.writer._index.transaction().unwrap();
    tx.execute_batch("PRAGMA defer_foreign_keys=ON; UPDATE message SET selected_body_pk=NULL; DELETE FROM message_build; DELETE FROM message_candidate; DELETE FROM message_body;").unwrap();
    crate::contributions::blocks::discard_cursor(&tx, &path, &build).unwrap();
    crate::contributions::directory::release(
        &tx,
        &path,
        crate::contributions::directory::Owner {
            message: build.message,
            session: build.session,
        },
        root,
    )
    .unwrap();
    tx.commit().unwrap();
    f.collect_nodes();
    f.collect_values();
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    assert!(reader.selected_message(&f.message).unwrap().is_none());
    assert_eq!(first.text_range(3, 100).unwrap().unwrap().bytes, b"lier");
    f.writer
        ._index
        .execute(
            "UPDATE message SET hidden=1,removal_epoch=removal_epoch+1",
            [],
        )
        .unwrap();
    assert!(matches!(first.text_range(0, 100), Err(Error::StaleMessage)));
    assert!(matches!(
        view.block(BlockSlot::Item(0)),
        Err(Error::StaleMessage)
    ));
    f.writer
        ._index
        .execute("UPDATE message SET hidden=0", [])
        .unwrap();
    assert!(matches!(
        second.text_range(0, 100),
        Err(Error::StaleMessage)
    ));
    drop(reader);
    drop(f.writer);
    assert!(matches!(
        second.text_range(0, 100),
        Err(Error::StaleMessage)
    ));
    assert!(matches!(
        crate::ReadHandle::open(&f.root),
        Err(Error::WriterHeld { .. })
    ));
    drop(first);
    drop(second);
    drop(view);
    assert!(crate::ReadHandle::open(&f.root).is_ok());
}
#[test]
fn published_source_projection_is_inherited_and_replaced_atomically() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let slot = Slot::Message(MessageField::At);
    let old_value = f.json(
        &build,
        ValuePurpose::Normalized,
        br#"{"at":1,"precision":"instant","time_quality":"intrinsic","until":null}"#,
    );
    f.writer
        .write_message_field(&build, &slot, Some(1), true, &step(0, b"at"), 4096)
        .unwrap();
    let prepared = f.finish(&build);
    let old = f
        .writer
        .publish_source_projection(&prepared, None, b"old source head", 4096)
        .unwrap();
    assert_eq!(old.message(), &f.message);
    assert_eq!(old.evidence_revision(), 1);
    assert_eq!(old.evidence_guard(), b"old source head");
    assert_eq!(old.predecessor(), &f.head.token);
    assert_eq!(old.body().pk, build.body);
    assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    let mut f = f.reopen();
    let old = f
        .writer
        .current_source_projection(&f.head.token, &[3; 32], 4096)
        .unwrap()
        .unwrap();
    assert_eq!(old.version_hash(), prepared.version_hash());
    assert!(
        f.writer
            .read_handle()
            .unwrap()
            .selected_message(&f.message)
            .unwrap()
            .is_none()
    );
    f.advance_parser();
    let new = f.prepare().unwrap().build;
    f.writer
        .inherit_message_projection(
            &new,
            old.body(),
            old.predecessor(),
            &step(0, b"inherited"),
            4096,
        )
        .unwrap();
    let new_value = f.json(
        &new,
        ValuePurpose::Normalized,
        br#"{"at":2,"precision":"instant","time_quality":"intrinsic","until":null}"#,
    );
    f.writer
        .set_message_field(&new, &slot, Some(2), true, &step(1, b"dirty at"), 4096)
        .unwrap();
    let completed = f.finish(&new);
    assert!(matches!(
        f.writer
            .publish_source_projection(&completed, Some(&old), b"new source head", 1),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(
        f.scalar(&format!(
            "SELECT ready FROM message_body WHERE pk={}",
            new.body
        )),
        0
    );
    f.exec("CREATE TRIGGER refuse_source_delete BEFORE DELETE ON message_build BEGIN SELECT RAISE(ABORT,'late source publish failure'); END;");
    assert!(matches!(
        f.writer
            .publish_source_projection(&completed, Some(&old), b"new source head", 4096),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(
        f.scalar(&format!(
            "SELECT eligible FROM message_candidate WHERE pk={}",
            build.candidate
        )),
        1
    );
    assert_eq!(
        f.scalar(&format!(
            "SELECT ready FROM message_body WHERE pk={}",
            new.body
        )),
        0
    );
    let current = f
        .writer
        .current_source_projection(&f.head.token, &[3; 32], 4096)
        .unwrap()
        .unwrap();
    assert_eq!(current.body(), old.body());
    assert_eq!(current.evidence_guard(), b"old source head");
    f.exec("DROP TRIGGER refuse_source_delete");
    let published = f
        .writer
        .publish_source_projection(&completed, Some(&old), b"new source head", 4096)
        .unwrap();
    assert_eq!(published.version_hash(), old.version_hash());
    assert_eq!(published.body().pk, new.body);
    assert_eq!(published.evidence_guard(), b"new source head");
    assert_eq!(
        f.scalar(&format!(
            "SELECT eligible FROM message_candidate WHERE pk={}",
            build.candidate
        )),
        0
    );
    assert_eq!(
        f.scalar(&format!(
            "SELECT eligible FROM message_candidate WHERE pk={}",
            new.candidate
        )),
        1
    );
    assert_eq!(
        f.scalar("SELECT count(*) FROM message WHERE selected_body_pk IS NOT NULL"),
        0
    );
    assert!(matches!(
        f.writer
            .publish_source_projection(&completed, Some(&old), b"replay", 4096),
        Err(Error::StaleMessage)
    ));
    let reader = f.writer.read_handle().unwrap();
    assert_eq!(
        reader.value_range(&old_value.id, 0, 256).unwrap().bytes,
        br#"{"at":1,"precision":"instant","time_quality":"intrinsic","until":null}"#
    );
    assert_eq!(
        reader.value_range(&new_value.id, 0, 256).unwrap().bytes,
        br#"{"at":2,"precision":"instant","time_quality":"intrinsic","until":null}"#
    );
    drop(reader);
    let mut f = f.reopen();
    assert_eq!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .unwrap()
            .body(),
        published.body()
    );
    f.advance_parser();
    let third = f.prepare().unwrap().build;
    let completed = f.finish(&third);
    assert!(matches!(
        f.writer
            .publish_source_projection(&completed, Some(&old), b"stale previous", 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 4096)
            .unwrap()
            .unwrap()
            .body(),
        published.body()
    );
}
#[test]
fn current_source_projection_rejects_wrong_or_corrupt_input() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let prepared = f.finish(&build);
    let prior = f
        .writer
        .publish_source_projection(&prepared, None, b"head", 4096)
        .unwrap();
    assert!(matches!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 1),
        Err(Error::ParserAdmission { .. })
    ));
    let mut wrong = f.head.token.clone();
    wrong.basis[0] ^= 1;
    assert!(matches!(
        f.writer.current_source_projection(&wrong, &[3; 32], 4096),
        Err(Error::StaleMessage)
    ));
    wrong = f.head.token.clone();
    wrong.generation += 1;
    assert!(matches!(
        f.writer.current_source_projection(&wrong, &[3; 32], 4096),
        Err(Error::StaleMessage)
    ));
    wrong = f.head.token.clone();
    wrong.index_generation.push('x');
    assert!(matches!(
        f.writer.current_source_projection(&wrong, &[3; 32], 4096),
        Err(Error::StaleMessage)
    ));
    f.advance_parser();
    f.message = MessageId::new(f.session.clone(), "other message").unwrap();
    let other = f.prepare().unwrap().build;
    let completed = f.finish(&other);
    assert!(matches!(
        f.writer
            .publish_source_projection(&completed, Some(&prior), b"wrong identity", 4096),
        Err(Error::StaleMessage)
    ));
    assert_eq!(
        f.scalar(&format!(
            "SELECT ready FROM message_body WHERE pk={}",
            other.body
        )),
        0
    );
    f.writer
        ._index
        .execute(
            "UPDATE parser_cell SET value=?1 WHERE namespace='message.source.v1' AND key=?2",
            params![b"corrupt reference".as_slice(), [3_u8; 32].as_slice()],
        )
        .unwrap();
    assert!(matches!(
        f.writer
            .current_source_projection(&f.head.token, &[3; 32], 4096),
        Err(Error::Invariant(_))
    ));
    assert!(
        f.writer
            .current_source_projection(&f.head.token, &[9; 32], 4096)
            .unwrap()
            .is_none()
    );
}
#[test]
fn uncontested_selection_is_atomic_and_contested_stays_unselected() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    f.value(&build, b"selected text");
    f.writer
        .write_block(
            &build,
            0,
            0,
            BlockKind::Text,
            presence(true, true),
            &step(0, b"text"),
            4096,
        )
        .unwrap();
    let done = f
        .writer
        .finish_message_projection(
            &build,
            &step(1, b"complete"),
            BlobHash::of(b"producer evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
    let source = f
        .writer
        .publish_source_projection(&done, None, b"evidence guard", 4096)
        .unwrap();
    let reader = f.writer.read_handle().unwrap();
    assert!(reader.selected_message(&f.message).unwrap().is_none());
    let before = f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'");
    f.exec("CREATE TRIGGER refuse_selected_session BEFORE UPDATE OF changed ON session BEGIN SELECT RAISE(ABORT,'synthetic selected session failure'); END;");
    assert!(
        f.writer
            .select_compatible_source_message_for_test(&source, 4096)
            .is_err()
    );
    assert!(reader.selected_message(&f.message).unwrap().is_none());
    assert_eq!(
        f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
        before
    );
    f.exec("DROP TRIGGER refuse_selected_session");
    assert_eq!(
        f.writer
            .select_compatible_source_message_for_test(&source, 4096)
            .unwrap(),
        SourceSelection::Selected
    );
    assert_eq!(
        f.writer
            .select_compatible_source_message_for_test(&source, 4096)
            .unwrap(),
        SourceSelection::AlreadySelected
    );
    assert_eq!(
        f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
        before + 1
    );
    let view = reader.selected_message(&f.message).unwrap().unwrap();
    assert_eq!(
        view.block(BlockSlot::Item(0))
            .unwrap()
            .unwrap()
            .text_range(0, 64)
            .unwrap()
            .unwrap()
            .bytes,
        b"selected text"
    );
    // A second input supplies a competing candidate. Selection must not
    // pick either one.
    let blob = f
        .writer
        .put_file_blob(b"different immutable input\n")
        .unwrap();
    let input = f
        .writer
        .store_immutable_input(
            blob,
            InputFormat::CodexRollout,
            &[],
            &ImportObservation {
                id: "second observation",
                machine: "synthetic",
                host: "synthetic",
                path: "/synthetic/second.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    f.head = f
        .writer
        .begin_parser(&parser_state::Basis {
            input_pk: input.input_pk,
            chain: input.chain,
            parser_version: 3,
            digest: [7; 32],
            acquired_upto: input.length,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    let other = f.prepare().unwrap().build;
    let done = f
        .writer
        .finish_message_projection(
            &other,
            &step(0, b"complete"),
            BlobHash::of(b"other evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
    let other = f
        .writer
        .publish_source_projection(&done, None, b"other evidence guard", 4096)
        .unwrap();
    assert_eq!(
        f.writer
            .select_compatible_source_message_for_test(&other, 4096)
            .unwrap(),
        SourceSelection::Contested
    );
    assert_eq!(
        f.writer
            .select_compatible_source_message_for_test(&source, 4096)
            .unwrap(),
        SourceSelection::Contested
    );
    assert_eq!(
        f.scalar("SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'"),
        before + 1
    );
    assert_eq!(
        view.block(BlockSlot::Item(0))
            .unwrap()
            .unwrap()
            .text_range(0, 64)
            .unwrap()
            .unwrap()
            .bytes,
        b"selected text"
    );
    let selected: i64 = f
        .writer
        ._index
        .query_row(
            "SELECT selected_body_pk FROM message WHERE id=?1",
            [f.message.as_str()],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(selected, build.body);
}
#[test]
fn message_reference_reports_pending_and_keeps_reasoning_order() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let reader = f.writer.read_handle().unwrap();
    assert!(matches!(
        reader.message_reference(f.message.as_str(), 4096).unwrap(),
        MessageLookup::Pending
    ));
    assert!(matches!(
        reader.message_reference("absent", 4096).unwrap(),
        MessageLookup::Absent
    ));
    assert!(matches!(
        reader.message_reference(f.message.as_str(), 1),
        Err(Error::ParserAdmission { .. })
    ));
    f.writer
        .set_message_layout(
            &build,
            MessageLayout::SeparateReasoning,
            &step(0, b"layout"),
            4096,
        )
        .unwrap();
    for (group, local, kind, text) in [
        (
            7,
            100,
            BlockKind::Thinking { encrypted: None },
            "reason one",
        ),
        (7, 500, BlockKind::Text, "answer one"),
        (
            900,
            0,
            BlockKind::Thinking {
                encrypted: Some(false),
            },
            "reason two",
        ),
        (900, 2, BlockKind::Text, "answer two"),
    ] {
        f.value(&build, text.as_bytes());
        let sequence = f.sequence(&build);
        f.writer
            .write_block(
                &build,
                group,
                local,
                kind,
                presence(true, true),
                &step(sequence, b"block"),
                4096,
            )
            .unwrap();
    }
    assert!(matches!(
        reader.message_reference(f.message.as_str(), 4096).unwrap(),
        MessageLookup::Pending
    ));
    let prepared = f.finish(&build);
    let source = f
        .writer
        .publish_source_projection(&prepared, None, b"synthetic selected evidence", 4096)
        .unwrap();
    f.writer
        .select_compatible_source_message_for_test(&source, 4096)
        .unwrap();
    let MessageLookup::Ready(view) = reader.message_reference(f.message.as_str(), 4096).unwrap()
    else {
        panic!("ready selected reference")
    };
    assert_eq!(view.id(), &f.message);
    assert_eq!(view.header().unwrap().role, Role::Assistant);
    assert_eq!(view.header().unwrap().written_by, WrittenBy::Model);
    for (index, slot, text) in [
        (0, BlockSlot::Reasoning(0), "reason one"),
        (1, BlockSlot::Item(0), "answer one"),
        (2, BlockSlot::Reasoning(1), "reason two"),
        (3, BlockSlot::Item(1), "answer two"),
    ] {
        let block = view.block_at(index).unwrap().unwrap();
        assert_eq!(block.slot, slot);
        assert_eq!(
            block.text_range(0, 100).unwrap().unwrap().bytes,
            text.as_bytes()
        );
        let same = view.block(slot).unwrap().unwrap();
        assert_eq!((same.group, same.local), (block.group, block.local));
    }
    assert!(view.block_at(4).unwrap().is_none());
}
