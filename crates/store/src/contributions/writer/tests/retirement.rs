use super::*;
#[test]
fn retiring_a_ready_body_keeps_roots_shared_with_other_bodies() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let source = build.body_id();
    let candidate = build.candidate_id();
    f.value(&build, b"shared text");
    f.writer
        .write_block(
            &build,
            3,
            7,
            BlockKind::Text,
            presence(true, true),
            &step(0, b"field"),
            4096,
        )
        .unwrap();
    f.value(&build, b"synthetic-model");
    f.writer
        .write_message_field(
            &build,
            &Slot::Message(MessageField::Model),
            None,
            true,
            &step(1, b"finish"),
            4096,
        )
        .unwrap();
    f.writer
        .finish_message_projection(
            &build,
            &step(2, b"done"),
            BlobHash::of(b"synthetic-ready-evidence"),
            FinishState::Unknown,
            4096,
        )
        .unwrap();
    let blocks = f.scalar("SELECT block_root_pk FROM message_body");
    let fields = f.scalar("SELECT field_root_pk FROM message_body");
    let path = f.index_path();
    let owner = d::Owner {
        message: build.message,
        session: build.session,
    };
    // Mark the body ready and add a copy that shares its roots, standing in
    // for election.
    let tx = f.writer._index.transaction().unwrap();
    tx.execute_batch("UPDATE message_body SET ready=1,version_hash=(SELECT version_hash FROM message_candidate WHERE pk=message_body.preferred_candidate_pk); UPDATE message_candidate SET ready=1,body_ready=1,eligible=1;").unwrap();
    tx.execute("INSERT INTO message_body(message_pk,session_pk,ready,basis_digest,contributor_revision,removal_epoch,preferred_candidate_pk,block_root_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version) SELECT message_pk,session_pk,1,zeroblob(32),0,0,preferred_candidate_pk,block_root_pk,field_root_pk,layout,role,written_by,input_pk,version_hash,parser_version FROM message_body WHERE pk=?1",[build.body]).unwrap();
    let canonical_pk = tx.last_insert_rowid();
    d::acquire(&tx, &path, owner, blocks).unwrap();
    d::acquire(&tx, &path, owner, fields).unwrap();
    tx.execute("UPDATE message SET selected_body_pk=?1", [build.body])
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::Selected
    );
    f.writer
        ._index
        .execute("UPDATE message SET selected_body_pk=?1", [canonical_pk])
        .unwrap();
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::ActiveBuild
    );
    let tx = f.writer._index.transaction().unwrap();
    crate::contributions::blocks::discard_cursor(&tx, &path, &build).unwrap();
    tx.execute("DELETE FROM message_build WHERE pk=?1", [build.pk])
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::Eligible
    );
    f.writer
        ._index
        .execute(
            "UPDATE message_candidate SET eligible=0 WHERE pk=?1",
            [build.candidate],
        )
        .unwrap();
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::DetachedCandidate
    );
    let mut f = f.reopen(); // restart after committed pointer detach
    f.exec("CREATE TRIGGER refuse_body_release BEFORE UPDATE OF incoming ON message_body_node BEGIN SELECT RAISE(ABORT,'synthetic body release failure'); END;");
    assert!(matches!(
        f.writer.retire_message_body(&source),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 2);
    assert_eq!(f.scalar("SELECT incoming FROM message_body_node WHERE pk=(SELECT block_root_pk FROM message_body LIMIT 1)"),2);
    f.exec("DROP TRIGGER refuse_body_release");
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::Body
    );
    assert_eq!(
        f.writer.retire_message_body(&source).unwrap(),
        BodyRetirement::Missing
    );
    assert!(!f.writer.collect_message_nodes(&f.message).unwrap());
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 1);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    assert_eq!(f.scalar("SELECT ready FROM message_candidate"), 1);
    assert_eq!(f.scalar("SELECT body_pk IS NULL FROM message_candidate"), 1);
    assert_eq!(
        f.writer.collect_message_candidate(&candidate).unwrap(),
        CandidateCollection::InUse
    );
    let mut f = f.reopen();
    let reader = f.writer.read_handle().unwrap();
    let view = reader.selected_message(&f.message).unwrap().unwrap();
    let canonical = view.body_id();
    match view
        .field(&Slot::Message(MessageField::Model), 4096)
        .unwrap()
        .unwrap()
    {
        FieldRead::Value(field) => {
            assert!(field.supplied);
            assert_eq!(field.range(0, 100).unwrap().bytes, b"synthetic-model");
        }
        _ => panic!("unexpected field conflict"),
    }
    assert_eq!(
        view.block(BlockSlot::Item(0))
            .unwrap()
            .unwrap()
            .text_range(0, 100)
            .unwrap()
            .unwrap()
            .bytes,
        b"shared text"
    );
    assert_eq!(
        f.writer.retire_message_body(&canonical).unwrap(),
        BodyRetirement::Selected
    );
    drop(view);
    drop(reader);
    // Another build holds a frontier lease on the shared field root.
    let next = f.prepare().unwrap().build;
    assert!(next.body > canonical_pk);
    let tx = f.writer._index.transaction().unwrap();
    d::acquire(&tx, &path, owner, fields).unwrap();
    tx.execute("INSERT INTO message_build_frontier(build_pk,message_pk,session_pk,depth,node_pk,next_slot) VALUES(?1,?2,?3,0,?4,0)",params![next.pk,next.message,next.session,fields]).unwrap();
    tx.execute("UPDATE message SET selected_body_pk=NULL", [])
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(
        f.writer.retire_message_body(&canonical).unwrap(),
        BodyRetirement::Body
    );
    f.collect_nodes();
    assert_eq!(f.scalar("SELECT count(*) FROM block"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM text"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
    assert_eq!(
        f.writer.collect_message_candidate(&candidate).unwrap(),
        CandidateCollection::InUse
    );
    f.writer.retire_message_preparation(&next).unwrap();
    f.exec("CREATE TRIGGER refuse_fact_release BEFORE DELETE ON message_field_fact BEGIN SELECT RAISE(ABORT,'synthetic fact release failure'); END;");
    let mut failed = false;
    for n in 0..100 {
        let nodes = f.scalar("SELECT count(*) FROM message_body_node");
        let edges = f.scalar("SELECT count(*) FROM message_body_edge");
        match f.writer.retire_message_turn(&next) {
            Err(Error::Sqlite { .. }) => {
                assert_eq!(f.scalar("SELECT count(*) FROM message_body_node"), nodes);
                assert_eq!(f.scalar("SELECT count(*) FROM message_body_edge"), edges);
                assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 1);
                failed = true;
                break;
            }
            Ok(Retirement::Complete) => panic!("fact failure was bypassed"),
            Ok(_) => {}
            Err(error) => panic!("unexpected {error}"),
        }
        assert!(n < 99);
    }
    assert!(failed);
    f.exec("DROP TRIGGER refuse_fact_release");
    f.retire_until_complete(&next);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_slot"), 0);
    assert_eq!(
        f.writer.collect_message_candidate(&candidate).unwrap(),
        CandidateCollection::Candidate
    );
    assert_eq!(
        f.writer.collect_message_candidate(&candidate).unwrap(),
        CandidateCollection::Missing
    );
    f.collect_values();
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    let mut other = Fixture::new();
    assert!(matches!(
        other.writer.retire_message_body(&canonical),
        Err(Error::StaleMessage)
    ));
    assert!(matches!(
        other.writer.collect_message_candidate(&candidate),
        Err(Error::StaleMessage)
    ));
    let generation: String = f
        .writer
        ._index
        .query_row("SELECT value FROM meta WHERE key='generation'", [], |r| {
            r.get(0)
        })
        .unwrap();
    f.writer
        ._index
        .execute(
            "UPDATE meta SET value='synthetic-new-index' WHERE key='generation'",
            [],
        )
        .unwrap();
    assert!(matches!(
        f.writer.retire_message_body(&source),
        Err(Error::StaleMessage)
    ));
    assert!(matches!(
        f.writer.collect_message_candidate(&candidate),
        Err(Error::StaleMessage)
    ));
    f.writer
        ._index
        .execute(
            "UPDATE meta SET value=?1 WHERE key='generation'",
            [generation],
        )
        .unwrap();
    assert_eq!(
        f.writer
            .archive
            .query_row("SELECT count(*) FROM revision", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
}
#[test]
fn retired_preparation_drains_across_restarts() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    for i in 0..8 {
        f.value(&build, format!("value-{i}").as_bytes());
        f.writer
            .write_message_field(
                &build,
                &Slot::Extra {
                    block: None,
                    key: format!("field-{i}"),
                },
                None,
                true,
                &step(f.sequence(&build), b"field"),
                4096,
            )
            .unwrap();
    }
    assert!(matches!(
        f.writer.retire_message_turn(&build),
        Err(Error::MessagePreparationPending)
    ));
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Root
    );
    f.head = f
        .writer
        .begin_parser(&parser_state::ParserInput {
            input_pk: f.head.token.input_pk,
            chain: 1,
            parser_version: 4,
            digest: [19; 32],
            acquired_upto: f.head.acquired_upto,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    let mut f = f.reopen();
    let before = (
        f.scalar("SELECT count(*) FROM message_body_node"),
        f.scalar("SELECT coalesce(sum(incoming),0) FROM message_body_node"),
    );
    f.exec("CREATE TRIGGER refuse_node_drain BEFORE DELETE ON message_body_edge BEGIN SELECT RAISE(ABORT,'synthetic node drain failure'); END;");
    assert!(matches!(
        f.writer.retire_message_turn(&build),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(
        (
            f.scalar("SELECT count(*) FROM message_body_node"),
            f.scalar("SELECT coalesce(sum(incoming),0) FROM message_body_node")
        ),
        before
    );
    f.exec("DROP TRIGGER refuse_node_drain");
    let mut turns = 0;
    loop {
        let before_nodes = f.scalar("SELECT count(*) FROM message_body_node");
        let before_facts = f.scalar("SELECT count(*) FROM message_field_fact");
        let step = f.writer.retire_message_turn(&build).unwrap();
        turns += 1;
        match step {
            Retirement::Node => assert_eq!(
                f.scalar("SELECT count(*) FROM message_body_node"),
                before_nodes - 1
            ),
            Retirement::Fact => assert_eq!(
                f.scalar("SELECT count(*) FROM message_field_fact"),
                before_facts - 1
            ),
            Retirement::Complete => break,
            _ => {}
        }
        if turns == 5 || turns == 20 {
            f = f.reopen();
        }
        assert!(turns < 200);
    }
    assert!(turns > 16);
    assert_eq!(f.scalar("SELECT count(*) FROM message_build"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_body"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_candidate"), 0);
    assert_eq!(f.scalar("SELECT count(*) FROM message_field_fact"), 0);
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Complete
    );
    let new = f.prepare().unwrap().build;
    assert_ne!(new, build);
    assert!(new.pk > build.pk);
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Complete
    );
    assert_eq!(f.prepare().unwrap().build, new);
    let mut chunks = 0;
    loop {
        let n = f.scalar("SELECT count(*) FROM body_value_chunk");
        match f.writer.collect_retiring_value().unwrap() {
            ValueCollection::Chunk => {
                chunks += 1;
                assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), n - 1);
            }
            ValueCollection::Value => {
                assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), n)
            }
            ValueCollection::Idle => break,
        }
    }
    assert_eq!(chunks, 8);
    assert_eq!(f.scalar("SELECT count(*) FROM body_value"), 0);
    let revisions: i64 = f
        .writer
        .archive
        .query_row("SELECT count(*) FROM revision", [], |r| r.get(0))
        .unwrap();
    assert_eq!(revisions, 1);
}
#[test]
fn value_collection_deletes_one_chunk_per_turn() {
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    let h = f
        .writer
        .begin_message_value(&build, values::Codec::Utf8, 65537)
        .unwrap();
    f.writer
        .append_value_chunk(&h.id, 0, &vec![b'x'; 65536], 65536)
        .unwrap();
    f.writer.append_value_chunk(&h.id, 65536, b"y", 1).unwrap();
    let generic = f.writer.begin_value(values::Codec::Utf8, 1).unwrap();
    f.writer.retire_message_preparation(&build).unwrap();
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Complete
    );
    assert_eq!(
        f.writer.collect_retiring_value().unwrap(),
        ValueCollection::Chunk
    );
    let mut f = f.reopen();
    assert_eq!(f.scalar("SELECT count(*) FROM body_value_chunk"), 1);
    assert_eq!(
        f.writer.collect_retiring_value().unwrap(),
        ValueCollection::Chunk
    );
    assert_eq!(
        f.writer.collect_retiring_value().unwrap(),
        ValueCollection::Value
    );
    assert_eq!(
        f.writer.collect_retiring_value().unwrap(),
        ValueCollection::Idle
    );
    assert_eq!(
        f.writer.value_head(&generic.id).unwrap().state,
        values::State::Building
    );
    assert!(matches!(f.writer.value_head(&h.id), Err(Error::StaleValue)));
}
#[test]
fn source_ranges_are_removed_one_per_turn() {
    fn ranges(f: &mut Fixture, build: &Build) {
        for (start, end) in [(0, 1), (3, 5), (10, 12)] {
            f.writer._index.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,?6,?7)",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain,start,end]).unwrap();
        }
        assert!(f.writer._index.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,0,1)",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain]).is_err());
    }
    let mut f = Fixture::new();
    let build = f.prepare().unwrap().build;
    ranges(&mut f, &build);
    f.writer.retire_message_preparation(&build).unwrap();
    f.exec("CREATE TRIGGER refuse_locator_delete BEFORE DELETE ON message_source BEGIN SELECT RAISE(ABORT,'synthetic locator delete failure'); END;");
    assert!(matches!(
        f.writer.retire_message_turn(&build),
        Err(Error::Sqlite { .. })
    ));
    assert_eq!(f.scalar("SELECT count(*) FROM message_source"), 3);
    f.exec("DROP TRIGGER refuse_locator_delete");
    for remaining in (0..3).rev() {
        assert_eq!(
            f.writer.retire_message_turn(&build).unwrap(),
            Retirement::Source
        );
        assert_eq!(f.scalar("SELECT count(*) FROM message_source"), remaining);
    }
    assert_eq!(
        f.writer.retire_message_turn(&build).unwrap(),
        Retirement::Complete
    );

    let mut f = Fixture::new();
    let old = f.prepare().unwrap().build;
    ranges(&mut f, &old);
    let candidate = old.candidate_id();
    let completed = f.finish(&old);
    let previous = f
        .writer
        .publish_source_projection(&completed, None, b"old source guard", 4096)
        .unwrap();
    f.advance_parser();
    let new = f.prepare().unwrap().build;
    let completed = f.finish(&new);
    f.writer
        .publish_source_projection(&completed, Some(&previous), b"new source guard", 4096)
        .unwrap();
    assert_eq!(
        f.writer.retire_message_body(previous.body()).unwrap(),
        BodyRetirement::DetachedCandidate
    );
    assert_eq!(
        f.writer.retire_message_body(previous.body()).unwrap(),
        BodyRetirement::Body
    );
    for remaining in (0..3).rev() {
        assert_eq!(
            f.writer.collect_message_candidate(&candidate).unwrap(),
            CandidateCollection::Source
        );
        assert_eq!(f.scalar("SELECT count(*) FROM message_source"), remaining);
    }
    assert_eq!(
        f.writer.collect_message_candidate(&candidate).unwrap(),
        CandidateCollection::Candidate
    );
}
