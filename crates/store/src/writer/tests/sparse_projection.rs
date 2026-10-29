use super::*;
use polylogue_model::bundle::{BundleError, MessageProjection};

fn original() -> Message {
    let mut message = text_message("sparse", "s", Role::Assistant, "source", Some("end_turn"));
    message.blocks = [
        serde_json::json!({"type":"text","text":"first source text"}),
        serde_json::json!({"type":"tool_call","call_id":"call","tool":"shell","input":{"opaque":[null,false,42]},"command":"true"}),
        serde_json::json!({"type":"tool_result","call_id":"call","outcome":{"kind":"ok"},"output":"source output","data":{"opaque":[{},null]}}),
        serde_json::json!({"type":"other","type_name":"synthetic","data":{"exact":null}}),
    ].into_iter().map(|value| serde_json::from_value(value).unwrap()).collect();
    message
}

fn selected(message: &Message, ordinals: &[u64]) -> (Message, MessageProjection) {
    let mut part = message.clone();
    part.blocks = ordinals
        .iter()
        .map(|&ordinal| message.blocks[ordinal as usize].clone())
        .collect();
    (
        part,
        MessageProjection {
            source_version: message.content_hash().unwrap(),
            block_count: message.blocks.len() as u64,
            block_ordinals: ordinals.to_vec(),
        },
    )
}

fn projected_batch(input: &str, message: Message, projection: MessageProjection) -> Batch {
    let mut batch = batch(input, 0, 0, vec![row(0, message)], &["s"]);
    batch.parse[0]
        .output
        .message_projections
        .insert(0, projection);
    batch.parse[0].prepare();
    batch
}

fn write(w: &TestArchive, batch: Batch) {
    w.writer.send(batch).unwrap();
    let report = w.writer.flush_rows().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
}

fn current(w: &TestArchive, id: &MessageId) -> (Message, MessageProjection) {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .message_with_projection(id)
        .unwrap()
        .unwrap()
}

#[test]
fn sparse_original_positions_survive_reopen_and_streaming_reads() {
    let w = test_archive();
    let message = original();
    let part = selected(&message, &[1, 3]);
    let expected = part.clone();
    write(&w, projected_batch("partial", part.0, part.1));
    assert_eq!(current(&w, &message.id), expected);
    let TestArchive {
        writer,
        _lock,
        _dir,
        ..
    } = w;
    drop(writer);
    drop(_lock);
    let (archive, _lock) = Archive::open_or_create(&_dir.path().join("archive")).unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.message_with_projection(&message.id).unwrap(),
        Some(expected.clone())
    );
    let mut streamed = Vec::new();
    reader
        .select_messages(&sid("s"), false)
        .unwrap()
        .visit_with_projection(|message, projection| {
            streamed.push((message, projection));
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(streamed, vec![expected]);
    assert_eq!(
        reader
            .query("SELECT idx,ordinal FROM block ORDER BY idx", &[])
            .unwrap(),
        [
            serde_json::json!({"idx":0,"ordinal":"1"}),
            serde_json::json!({"idx":1,"ordinal":"3"}),
        ]
    );
}

#[test]
fn compatible_partials_union_without_downgrading_full_or_pinned_readings() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[2]);
    write(
        &w,
        projected_batch("first", first.0.clone(), first.1.clone()),
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let pinned = pool.reader().unwrap();
    assert_eq!(
        pinned.message_with_projection(&message.id).unwrap(),
        Some(first.clone())
    );
    let second = selected(&message, &[0, 1, 3]);
    write(
        &w,
        projected_batch("second", second.0.clone(), second.1.clone()),
    );
    assert_eq!(
        pinned.message_with_projection(&message.id).unwrap(),
        Some(first)
    );
    drop(pinned);
    let full = selected(&message, &[0, 1, 2, 3]);
    assert_eq!(current(&w, &message.id), full);
    let mut replay = projected_batch("second", second.0, second.1);
    replay.change = Change::Reparse { chain: 1 };
    write(&w, replay);
    assert_eq!(current(&w, &message.id), full);
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM message_version", &[])
            .unwrap()[0]["n"],
        1
    );
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM message_version_text", &[])
            .unwrap()[0]["n"],
        2
    );
}

#[test]
fn sparse_overlap_conflict_is_typed_and_keeps_the_prior_value() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[0]);
    write(
        &w,
        projected_batch("first", first.0.clone(), first.1.clone()),
    );
    let mut conflicting = first.0.clone();
    conflicting.blocks[0] = Block::Text {
        text: "different claimed bytes".into(),
    };
    let mut header = first.0.clone();
    header.attrs.insert("different-header".into(), true.into());
    let mut count = first.1.clone();
    count.block_count += 1;
    for (message, projection) in [
        (conflicting, first.1.clone()),
        (header, first.1.clone()),
        (first.0.clone(), count),
    ] {
        let mut batch = projected_batch("first", message, projection);
        batch.change = Change::Reparse { chain: 1 };
        let index = crate::archive::open_write(&w.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
        let error = resolve::prepare_index(&index, &mut batch, 1, false).unwrap_err();
        assert!(
            matches!(
                error,
                StoreError::Projection(BundleError::ProjectionConflict)
            ),
            "{error:?}"
        );
        assert_eq!(current(&w, &first.0.id), first);
    }
}

#[test]
fn sparse_observation_cannot_replace_a_complete_native_value() {
    let w = test_archive();
    let message = original();
    write(
        &w,
        batch("native", 0, 0, vec![row(0, message.clone())], &["s"]),
    );
    let part = selected(&message, &[1]);
    write(&w, projected_batch("import", part.0, part.1));
    assert_eq!(current(&w, &message.id), selected(&message, &[0, 1, 2, 3]));
}

#[test]
fn sparse_streamed_confirmed_readings_preserve_every_known_block() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    crate::schema::create_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let message = original();
    let mut readings = Vec::new();
    for (n, ordinals) in [&[0][..], &[1][..], &[2, 3][..]].into_iter().enumerate() {
        let part = selected(&message, ordinals);
        let mut input = batch(
            &format!("confirmed-{n}"),
            0,
            0,
            vec![row(0, part.0)],
            &["s"],
        );
        input.parse[0].output.message_projections.insert(0, part.1);
        let mut reading = ChainReading::new().unwrap();
        reading.push(&mut input.parse.remove(0)).unwrap();
        readings.push((input, reading));
    }
    let mut work: Vec<_> = readings
        .into_iter()
        .map(|(input, reading)| (input.input, 1, input.length, reading))
        .collect();
    resolver
        .confirmed_readings(&mut work, &HashMap::new())
        .unwrap();
    let tx = index.transaction().unwrap();
    for (input, chain, length, reading) in &mut work {
        let input = Batch {
            input: input.clone(),
            change: Change::Reparse { chain: *chain },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: vec![],
            length: *length,
            parse: vec![],
            release: InFlight::default(),
        };
        reading
            .visit(|step| rows::write_confirmed_rows(&tx, &input, step, *chain, 1, None).map(drop))
            .unwrap();
    }
    tx.commit().unwrap();
    assert_eq!(
        crate::versions::read_message(&index, &message.content_hash().unwrap().0).unwrap(),
        Some(selected(&message, &[0, 1, 2, 3]))
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message_source", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        3
    );
    assert!(
        !archive.archive_db().exists(),
        "confirmed materialization does not consult standalone source state"
    );
}

#[test]
fn sparse_distinct_source_versions_keep_the_existing_selection_law() {
    let w = test_archive();
    let message = original();
    let first = selected(&message, &[0]);
    write(
        &w,
        projected_batch("preferred", first.0.clone(), first.1.clone()),
    );
    let mut different = message.clone();
    different.blocks[3] = Block::Text {
        text: "another source reading".into(),
    };
    let second = selected(&different, &[3]);
    let mut batch = projected_batch("other", second.0, second.1);
    batch.input.arrival_rank = 2;
    write(&w, batch);
    assert_eq!(current(&w, &message.id), first);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .query("SELECT count(*) AS n FROM message_version", &[])
            .unwrap()[0]["n"],
        2
    );
}

#[test]
fn sparse_unsigned_ordinals_are_not_renumbered_by_attachment_or_tool_reads() {
    let w = test_archive();
    let mut message = original();
    message.blocks = vec![
        serde_json::from_value(serde_json::json!({"type":"image","direction":"input","url":"https://example.invalid/image.png"})).unwrap(),
        message.blocks[1].clone(),
    ];
    let ordinals = vec![1_u64 << 63, u64::MAX - 1];
    let projection = MessageProjection {
        source_version: BlobHash::of(b"synthetic unknown complete source"),
        block_count: u64::MAX,
        block_ordinals: ordinals.clone(),
    };
    write(
        &w,
        projected_batch("large-positions", message.clone(), projection.clone()),
    );
    assert_eq!(current(&w, &message.id), (message.clone(), projection));
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let asset = reader
        .attachment(&message.id, ordinals[0] as usize, 0)
        .unwrap();
    assert_eq!(asset.block as u64, ordinals[0]);
    assert!(matches!(
        reader.attachment(&message.id, 0, 0),
        Err(StoreError::NotFound(_))
    ));
    assert_eq!(
        reader.tool_outcomes(&message.id).unwrap()[0].0 as u64,
        ordinals[1]
    );
}

#[test]
fn sparse_index_migration_keeps_native_versions_and_infers_dense_positions() {
    let w = test_archive();
    let message = original();
    write(
        &w,
        batch("native", 0, 0, vec![row(0, message.clone())], &["s"]),
    );
    let index = crate::archive::open_write(&w.archive.index_db(), 1).unwrap();
    // These are the only differences from the prior index format. The
    // existing writer supplied the opaque payloads and shared text rows.
    index.execute_batch("DROP TABLE message_projection; ALTER TABLE block DROP COLUMN ordinal; PRAGMA user_version=1;").unwrap();
    crate::schema::ensure_index(&index).unwrap();
    assert_eq!(crate::schema::version(&index).unwrap(), 2);
    assert_eq!(current(&w, &message.id), selected(&message, &[0, 1, 2, 3]));
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message_version", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
}
