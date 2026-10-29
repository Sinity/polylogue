use super::*;
use polylogue_model::{
    Block, Message, MessageId, MessageRow, Role, Session, SessionId, SessionKind,
};

fn fact(position: i64, event: &str, product: Product) -> Confirmed {
    Confirmed {
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(product)),
    }
}
fn lifecycle(position: i64, event: &str, change: ConfirmedChange) -> Confirmed {
    Confirmed {
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
        input: crate::InputKey {
            id: input.into(),
            format: "synthetic".into(),
            arrival_rank: 2,
        },
        chain: 1,
        reading_version: "synthetic-v1".into(),
        order_from: None,
        repeats_whole_conversations: false,
        start: 0,
        parsed_upto: 2,
        at_end: true,
        no_parser_yet: false,
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
                rows: crate::input_rows::InputRows {
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
                },
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
                    offset: 0,
                    end: 2,
                    message: subset,
                }),
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
