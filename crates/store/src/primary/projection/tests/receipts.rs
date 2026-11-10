use super::*;
use polylogue_model::{
    Block, Message, MessageId, MessageRow, Role, Session, SessionId, SessionKind,
};

fn fact(position: i64, event: &str, product: Product) -> Confirmed {
    Confirmed {
        native: None,
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(product)),
    }
}
fn lifecycle(position: i64, event: &str, change: ConfirmedChange) -> Confirmed {
    Confirmed {
        native: None,
        position,
        output_index: 0,
        event: event.into(),
        change,
    }
}
fn original() -> Message {
    Message {
        id: MessageId::parse("codex:message").unwrap(),
        session: SessionId::parse("codex:session").unwrap(),
        parent: None,
        role: Role::Assistant,
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind: None,
        blocks: vec![
            Block::Text {
                text: "first".into(),
            },
            Block::Text {
                text: "second".into(),
            },
        ],
        usage: Default::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    }
}
fn source(input: &str, message: &Message, ordinal: usize, position: i64) -> Vec<Confirmed> {
    let hash = polylogue_model::BlobHash::of(b"ab");
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 2,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    let reading = crate::primary::Reading {
        source_length: None,
        delivery: None,
        input: crate::InputKey {
            id: input.into(),
            format: "synthetic".into(),
            route: polylogue_model::Route::Export,
        },
        chain: 1,
        reading_version: "synthetic-v1".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 2,
        progress: polylogue_model::batch::StepProgress::AtEnd,
    };
    let mut subset = message.clone();
    subset.blocks = vec![message.blocks[ordinal].clone()];
    let mut counts = polylogue_model::RecordCounts {
        records: 1,
        ..Default::default()
    };
    counts.add("message", &polylogue_model::RecordOutcome::Used);
    vec![
        fact(
            ordinal as i64 + 1,
            &format!("input-{input}"),
            Product::Input {
                rows: Box::new(crate::input_rows::InputRows {
                    acquisition: None,
                    input: input.into(),
                    format: "synthetic".into(),
                    revision: Some(crate::input_rows::Revision {
                        number: 1,
                        produced: "first".into(),
                        chain: 1,
                        length: 2,
                    }),
                    extent: Some(crate::input_rows::Extent {
                        chain: 1,
                        offset: 0,
                        length: 2,
                        blob: hash,
                        blob_offset: 0,
                    }),
                    observation: None,
                }),
                material: Some(Material {
                    id: format!("raw-{input}"),
                    hash,
                    length: 2,
                    extents: vec![extent.encode()],
                }),
            },
        ),
        fact(
            position - 1,
            &format!("session-{input}"),
            Product::Session {
                reading: reading.clone(),
                facts: Session::new(message.session.clone(), SessionKind::Main),
            },
        ),
        fact(
            position,
            &format!("message-{input}"),
            Product::Message {
                reading: reading.clone(),
                facts: Box::new(MessageRow {
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Native,
                    offset: 0,
                    end: 2,
                    message: subset,
                }),
                native_fields: None,
                declared_injections: None,
                declared_attachment_bindings: None,
                order: None,
                projection: Some(polylogue_model::bundle::MessageProjection {
                    source_version: message.content_hash().unwrap(),
                    block_count: 2,
                    block_ordinals: vec![ordinal as u64],
                }),
            },
        ),
        fact(
            position + 1,
            &format!("progress-{input}"),
            Product::ParseProgress {
                reading,
                counts,
                reset: true,
                comparison: None,
                checkpoint: None,
                stopped_attempt: None,
                semantic_event_ids: None,
            },
        ),
    ]
}
fn assert_message(archive: &crate::Archive, message: &Message) {
    let pool = crate::ReadPool::open(archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader.message_with_projection(&message.id).unwrap(),
        Some((
            message.clone(),
            polylogue_model::bundle::MessageProjection {
                source_version: message.content_hash().unwrap(),
                block_count: 2,
                block_ordinals: vec![0, 1]
            }
        ))
    );
    assert_eq!(reader.message_admission(&message.id).unwrap(), Some(30));
    let mut streamed = Vec::new();
    reader
        .select_messages(&message.session, false)
        .unwrap()
        .visit_with_evidence(|value, projection, admission| {
            streamed.push((value, projection, admission));
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(streamed.len(), 1);
    assert_eq!(streamed[0].0, *message);
    assert_eq!(streamed[0].2, Some(30));
}
#[test]
fn sparse_canonical_receipts_survive_raw_lifecycle_message_restoration_and_reopen() {
    let directory = tempfile::tempdir().unwrap();
    let bytes = directory.path().join("unavailable-bytes");
    let root = directory.path().join("cache");
    let (archive, _lock) = crate::Archive::open_primary(&root, &bytes).unwrap();
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let message = original();
    let mut changes = source("a", &message, 0, 10);
    changes.extend(source("b", &message, 1, 30));
    changes.push(barrier(31));
    apply(&mut index, &resolver, &changes).unwrap();
    assert_eq!(
        crate::source::Source::admitted_through(&index, "a", 1, 2).unwrap(),
        Some(1)
    );
    assert_message(&archive, &message);
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(40, "input-a", ConfirmedChange::Archived),
            lifecycle(40, "input-b", ConfirmedChange::Archived),
            barrier(40),
        ],
    )
    .unwrap();
    assert_eq!(
        crate::source::Source::admitted_through(&index, "a", 1, 2).unwrap(),
        None
    );
    assert_message(&archive, &message);
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(50, "input-a", ConfirmedChange::Restored),
            lifecycle(50, "input-b", ConfirmedChange::Restored),
            barrier(50),
        ],
    )
    .unwrap();
    assert_eq!(
        crate::source::Source::admitted_through(&index, "a", 1, 2).unwrap(),
        Some(1)
    );
    assert_message(&archive, &message);
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(60, "message-a", ConfirmedChange::Archived),
            lifecycle(60, "message-b", ConfirmedChange::Archived),
            barrier(60),
        ],
    )
    .unwrap();
    assert!(
        crate::ReadPool::open(&archive)
            .unwrap()
            .reader()
            .unwrap()
            .message_with_projection(&message.id)
            .unwrap()
            .is_none()
    );
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(70, "message-a", ConfirmedChange::Restored),
            lifecycle(70, "message-b", ConfirmedChange::Restored),
            barrier(70),
        ],
    )
    .unwrap();
    assert_message(&archive, &message);
    drop(resolver);
    drop(index);
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[]).unwrap();
    assert_message(&archive, &message);
    assert!(!bytes.exists());
}

#[test]
fn sealed_export_callbacks_rebuild_from_confirmed_facts_without_recounting_or_receipt_changes() {
    use polylogue_model::batch::{ExportDelivery, ExportDeliverySource};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("cache");
    let (archive, _lock) =
        crate::Archive::open_primary(&root, &directory.path().join("bytes")).unwrap();
    let mut index = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let message = original();
    let input = crate::InputKey {
        id: "export".into(),
        format: "synthetic".into(),
        route: polylogue_model::Route::Live,
    };
    let mut changes = vec![source("export", &message, 0, 10).remove(0)];
    let mut prefix = ExportDelivery::seed();
    for (ordinal, footprint) in [
        ExportDeliverySource::ContainerHeader,
        ExportDeliverySource::Item { start: 0, end: 2 },
        ExportDeliverySource::Complete {
            length: 2,
            item_count: 1,
            delivery_count: 2,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let mut output = polylogue_model::Output::default();
        let mut counts = polylogue_model::RecordCounts::default();
        if ordinal == 1 {
            output
                .sessions
                .push(Session::new(message.session.clone(), SessionKind::Main));
            output.messages.push(MessageRow {
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset: 0,
                end: 2,
                message: message.clone(),
            });
            counts.records = 1;
            counts.add("message", &polylogue_model::RecordOutcome::Used);
            counts.wire_shapes.insert(
                "message".into(),
                polylogue_model::WireShape {
                    record_kind: "message".into(),
                    path: "/text".into(),
                    types: std::collections::BTreeMap::from([("string".into(), 1)]),
                    ..Default::default()
                },
            );
        }
        let complete = ordinal == 2;
        let mut step =
            crate::ParseStep::new(output, counts, 0..if complete { 2 } else { 0 }, complete);
        step.reading_version = "crate::export::VERSION=2".into();
        step.reset = ordinal == 0;
        prefix = step
            .seal_export(ordinal as u64, footprint, prefix)
            .unwrap()
            .prefix_hash;
        step.prepare();
        for product in crate::primary::products::parse(&input, 1, Some(2), &step).unwrap() {
            let position = changes.len() as i64 + 2;
            changes.push(fact(position, &format!("export-{position}"), product));
        }
    }
    changes.push(barrier(20));
    apply(&mut index, &resolver, &changes).unwrap();
    let retained: String = index
        .query_row("SELECT state FROM wire_reading", [], |row| row.get(0))
        .unwrap();
    let count: i64 = index
        .query_row(
            "SELECT count FROM wire_type WHERE json_type='string'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
    let shape = stored_shape(&index);
    let admission: i64 = index.query_row("SELECT p.admitted FROM message_projection p JOIN message_source s ON s.version_hash=p.version_hash JOIN message m ON m.pk=s.message_pk WHERE m.id=?1", [message.id.to_string()], |row| row.get(0)).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&retained).unwrap()["export_cursor"]["last_ordinal"],
        2
    );
    apply(
        &mut index,
        &resolver,
        &[
            lifecycle(21, "input-export", ConfirmedChange::Archived),
            barrier(21),
        ],
    )
    .unwrap();
    assert_eq!(
        index
            .query_row("SELECT state FROM wire_reading", [], |row| row
                .get::<_, String>(0))
            .unwrap(),
        retained
    );
    assert_eq!(index.query_row("SELECT p.admitted FROM message_projection p JOIN message_source s ON s.version_hash=p.version_hash JOIN message m ON m.pk=s.message_pk WHERE m.id=?1", [message.id.to_string()], |row| row.get::<_, i64>(0)).unwrap(), admission);
    assert_eq!(stored_shape(&index), shape);
    drop(index);
    let rebuilt_root = directory.path().join("rebuilt");
    let (rebuilt_archive, _rebuilt_lock) =
        crate::Archive::open_primary(&rebuilt_root, &directory.path().join("bytes")).unwrap();
    let mut rebuilt = crate::archive::open_write(&rebuilt_archive.index_db(), 1).unwrap();
    let rebuilt_resolver =
        crate::writer::resolve::Resolver::new(&rebuilt_archive, &rebuilt, None).unwrap();
    apply(&mut rebuilt, &rebuilt_resolver, &changes).unwrap();
    assert_eq!(
        rebuilt
            .query_row("SELECT state FROM wire_reading", [], |row| row
                .get::<_, String>(0))
            .unwrap(),
        retained
    );
    assert_eq!(rebuilt.query_row("SELECT p.admitted FROM message_projection p JOIN message_source s ON s.version_hash=p.version_hash JOIN message m ON m.pk=s.message_pk WHERE m.id=?1", [message.id.to_string()], |row| row.get::<_, i64>(0)).unwrap(), admission);
    assert_eq!(
        rebuilt
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(stored_shape(&rebuilt), shape);
    let mut unbound = crate::ParseStep::new(Default::default(), Default::default(), 0..0, false);
    unbound.reading_version = "crate::export::VERSION=2".into();
    unbound
        .seal_export(
            0,
            ExportDeliverySource::ContainerHeader,
            ExportDelivery::seed(),
        )
        .unwrap();
    unbound.prepare();
    let control = crate::primary::products::parse(&input, 1, Some(3), &unbound)
        .unwrap()
        .pop()
        .unwrap();
    let error = apply(
        &mut rebuilt,
        &rebuilt_resolver,
        &[fact(99, "unbound-export", control), barrier(99)],
    )
    .unwrap_err();
    assert!(
        matches!(&error,StoreError::Invalid(reason) if reason.contains("no retained confirmed source revision")),
        "{error}"
    );
    assert_eq!(
        rebuilt
            .query_row("SELECT state FROM wire_reading", [], |row| row
                .get::<_, String>(0))
            .unwrap(),
        retained
    );
}

/// Every stored shape and type row, in order.
fn stored_shape(conn: &rusqlite::Connection) -> String {
    conn.query_row(
        "SELECT (SELECT json_group_array(json_array(record_kind,path,table_key,unknown_kinds,unknown_tables,unused,first,last)) FROM (SELECT * FROM wire_shape ORDER BY record_kind,path,table_key))
          ||(SELECT json_group_array(json_array(record_kind,path,table_key,json_type,count,first,last)) FROM (SELECT * FROM wire_type ORDER BY record_kind,path,table_key,json_type))",
        [],
        |row| row.get(0),
    )
    .unwrap()
}
