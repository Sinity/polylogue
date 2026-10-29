use super::*;

fn cache(path: &std::path::Path) -> Connection {
    let index = Connection::open(path).unwrap();
    index.execute_batch(crate::schema::INDEX).unwrap();
    index.execute_batch(crate::variants::SCHEMA).unwrap();
    index
        .execute_batch(&crate::source::schema().unwrap().ddl)
        .unwrap();
    index.execute_batch(SCHEMA).unwrap();
    index
}

fn input(position: i64, id: &str) -> Confirmed {
    Confirmed {
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: crate::input_rows::InputRows {
                input: id.into(),
                format: "synthetic".into(),
                revision: None,
                extent: None,
                observation: None,
            },
            material: None,
        })),
    }
}
fn barrier(position: i64) -> Confirmed {
    Confirmed {
        position,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::PositionComplete,
    }
}

#[test]
fn staged_pages_survive_restart_and_a_cumulative_barrier_commits_them_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let mut index = cache(&path);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    assert_eq!(
        apply(&mut index, &resolver, &[input(2, "first")]).unwrap(),
        0
    );
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    index.execute_batch("CREATE TRIGGER stop_projection BEFORE INSERT ON source_input WHEN new.id='second' BEGIN SELECT raise(ABORT,'stop'); END").unwrap();
    assert!(apply(&mut index, &resolver, &[input(4, "second"), barrier(5)]).is_err());
    assert_eq!(checkpoint(&index).unwrap(), None);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_fact", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    index.execute_batch("DROP TRIGGER stop_projection").unwrap();
    assert_eq!(apply(&mut index, &resolver, &[]).unwrap(), 5);
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM source_input", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        apply(&mut index, &resolver, &[input(2, "first"), barrier(5)]).unwrap(),
        5
    );
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM primary_fact", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert!(!archive.root.join("archive.db").exists());
}

#[test]
fn an_empty_confirmed_database_is_known_and_zero_data_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    assert_eq!(checkpoint(&index).unwrap(), None);
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    apply(&mut index, &resolver, &[barrier(0)]).unwrap();
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
    assert!(apply(&mut index, &resolver, &[input(0, "invalid")]).is_err());
    assert_eq!(checkpoint(&index).unwrap(), Some(0));
}

#[test]
fn source_lifecycle_preserves_independent_frozen_authored_effects() {
    use polylogue_model::user::{UserRequest, UserWrite};
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let request = UserRequest {
        id: "setting-request".into(),
        machine: "machine".into(),
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        write: UserWrite::SetSetting {
            key: "saved-key".into(),
            value: "saved-value".into(),
            author: polylogue_model::Author::Operator,
            machine: None,
        },
    };
    let prepared =
        crate::primary::authored::prepare(&index, &request, |_| Ok(()), |_| Ok(())).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            input(1, "source"),
            Confirmed {
                position: 2,
                output_index: 0,
                event: "authored".into(),
                change: ConfirmedChange::Live(Box::new(Product::Request(prepared))),
            },
            barrier(2),
        ],
    )
    .unwrap();
    for (position, change, present) in [
        (3, ConfirmedChange::Archived, 0),
        (4, ConfirmedChange::Restored, 1),
        (5, ConfirmedChange::Tombstoned, 0),
    ] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    output_index: 0,
                    event: "event-source".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert_eq!(
            index
                .query_row("SELECT count(*) FROM source_input", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            present
        );
        assert_eq!(
            index
                .query_row(
                    "SELECT value FROM source_setting WHERE key='saved-key'",
                    [],
                    |row| row.get::<_, String>(0)
                )
                .unwrap(),
            "saved-value"
        );
    }
    assert!(
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position: 6,
                    output_index: 0,
                    event: "event-source".into(),
                    change: ConfirmedChange::Restored
                },
                barrier(6)
            ]
        )
        .is_err()
    );
    assert_eq!(position(&index).unwrap(), 5);
    let body: Option<Vec<u8>> = index
        .query_row(
            "SELECT product FROM primary_fact WHERE event='event-source'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(body, None);
}

#[test]
fn confirmed_slices_keep_full_material_descriptors_without_opening_source_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let bytes_root = dir.path().join("unavailable-bytes");
    let archive = crate::Archive::on_sinex(dir.path(), &bytes_root);
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let hash = polylogue_model::BlobHash::of(b"abcd");
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 4,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    let material = Material {
        id: "shared-material".into(),
        hash,
        length: 4,
        extents: vec![extent.encode()],
    };
    let facts = |position, id: &str, blob_offset| Confirmed {
        position,
        output_index: 0,
        event: format!("event-{id}"),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: crate::input_rows::InputRows {
                input: id.into(),
                format: "synthetic".into(),
                observation: None,
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
                    blob_offset,
                }),
            },
            material: Some(material.clone()),
        })),
    };
    apply(
        &mut index,
        &resolver,
        &[facts(7, "first", 1), facts(8, "second", 0), barrier(8)],
    )
    .unwrap();
    assert_eq!(
        material_at(&index, "first", 1, 1).unwrap(),
        Some(super::super::SourceLocation {
            material: "shared-material".into(),
            anchor: 2
        })
    );
    assert_eq!(
        material_at(&index, "second", 1, 1).unwrap(),
        Some(super::super::SourceLocation {
            material: "shared-material".into(),
            anchor: 1
        })
    );
    assert_eq!(
        crate::source::Source::admitted_through(&index, "first", 1, 2).unwrap(),
        Some(7)
    );
    let retained: (i64, Vec<u8>) = index
        .query_row("SELECT size,storage FROM source_blob", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(retained, (4, extent.encode()));
    assert!(!bytes_root.exists());
}

#[test]
fn later_material_confirmation_and_lifecycle_refresh_original_attachment_facts() {
    use polylogue_model::{
        Attachment, AttachmentDirection, Block, Message, MessageId, MessageRow, Output,
        RecordCounts, RecordOutcome, Role, Session, SessionId, SessionKind,
    };
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("unavailable-bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let hash = polylogue_model::BlobHash::of(b"image");
    let session = SessionId::parse("claude-code:session").unwrap();
    let message = Message {
        id: MessageId::parse("claude-code:message").unwrap(),
        session: session.clone(),
        parent: None,
        role: Role::User,
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind: None,
        blocks: vec![Block::Image(Attachment {
            name: None,
            media_type: Some("image/png".into()),
            size: None,
            blob: Some(hash),
            url: None,
            source_id: None,
            direction: AttachmentDirection::Input,
        })],
        usage: Default::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    };
    let mut counts = RecordCounts {
        records: 1,
        ..Default::default()
    };
    counts.add("message", &RecordOutcome::Used);
    let mut step = crate::ParseStep::new(
        Output {
            sessions: vec![Session::new(session, SessionKind::Main)],
            messages: vec![MessageRow {
                offset: 0,
                end: 1,
                message,
            }],
            ..Default::default()
        },
        counts,
        0..1,
        true,
    );
    step.reading_version = "synthetic-v1".into();
    step.prepare();
    let key = crate::InputKey {
        id: "input".into(),
        format: "synthetic".into(),
        arrival_rank: 0,
    };
    let products = crate::primary::products::parse(&key, 1, &step).unwrap();
    let mut changes = vec![Confirmed {
        position: 1,
        output_index: 0,
        event: "input-fact".into(),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: crate::input_rows::InputRows {
                input: "input".into(),
                format: "synthetic".into(),
                revision: Some(crate::input_rows::Revision {
                    number: 1,
                    produced: "first".into(),
                    chain: 1,
                    length: 1,
                }),
                extent: None,
                observation: None,
            },
            material: None,
        })),
    }];
    changes.extend(
        products
            .into_iter()
            .enumerate()
            .map(|(index, product)| Confirmed {
                position: 1,
                output_index: index as u64,
                event: format!("parsed-{index}"),
                change: ConfirmedChange::Live(Box::new(product)),
            }),
    );
    changes.push(barrier(1));
    apply(&mut index, &resolver, &changes).unwrap();
    let shown = |conn: &Connection| {
        conn.query_row(
            "SELECT blob_hash FROM block WHERE kind='image'",
            [],
            |row| row.get::<_, Option<Vec<u8>>>(0),
        )
        .unwrap()
    };
    assert_eq!(shown(&index), None);
    let extent = hashed_blobs::Extent {
        segment: "segments/confirmed.hb".into(),
        offset: 0,
        len: 100,
        decoded_len: 5,
        hash: hashed_blobs::Hash(hash.0),
        kind: hashed_blobs::Kind::Plain,
        index_hash: hashed_blobs::Hash([1; 32]),
        encoded_hash: hashed_blobs::Hash([2; 32]),
    };
    let blob = Product::Blob(Material {
        id: "image-material".into(),
        hash,
        length: 5,
        extents: vec![extent.encode()],
    });
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                position: 2,
                output_index: 0,
                event: "blob-fact".into(),
                change: ConfirmedChange::Live(Box::new(blob)),
            },
            barrier(2),
        ],
    )
    .unwrap();
    assert_eq!(shown(&index), Some(hash.0.to_vec()));
    for (position, change, present) in [
        (3, ConfirmedChange::Archived, false),
        (4, ConfirmedChange::Restored, true),
    ] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    position,
                    output_index: 0,
                    event: "blob-fact".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert_eq!(shown(&index), present.then(|| hash.0.to_vec()));
    }
}

#[test]
fn hiding_one_equal_range_event_preserves_the_other_native_identity() {
    use polylogue_model::{Event, Output, RecordCounts, RecordOutcome};
    let dir = tempfile::tempdir().unwrap();
    let mut index = cache(&dir.path().join("index.db"));
    let archive = crate::Archive::on_sinex(dir.path(), &dir.path().join("unavailable-bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let event = Event {
        session: None,
        kind: "native".into(),
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: None,
        range: Some((0, 1)),
        data: serde_json::json!({"same":true}),
    };
    let mut counts = RecordCounts {
        records: 1,
        ..Default::default()
    };
    counts.add("native", &RecordOutcome::Used);
    let mut step = crate::ParseStep::new(
        Output {
            events: vec![event.clone(), event],
            ..Default::default()
        },
        counts,
        0..1,
        true,
    );
    step.reading_version = "synthetic-v1".into();
    step.prepare();
    let key = crate::InputKey {
        id: "input".into(),
        format: "synthetic".into(),
        arrival_rank: 0,
    };
    let products = crate::primary::products::parse(&key, 1, &step).unwrap();
    let ids = products
        .iter()
        .filter_map(|product| match product {
            Product::Event { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_ne!(ids[0], ids[1]);
    let mut changes = vec![Confirmed {
        position: 1,
        output_index: 0,
        event: "input-fact".into(),
        change: ConfirmedChange::Live(Box::new(Product::Input {
            rows: crate::input_rows::InputRows {
                input: "input".into(),
                format: "synthetic".into(),
                revision: Some(crate::input_rows::Revision {
                    number: 1,
                    produced: "first".into(),
                    chain: 1,
                    length: 1,
                }),
                extent: None,
                observation: None,
            },
            material: None,
        })),
    }];
    changes.extend(
        products
            .into_iter()
            .enumerate()
            .map(|(index, product)| Confirmed {
                position: 1,
                output_index: index as u64,
                event: format!("parsed-{index}"),
                change: ConfirmedChange::Live(Box::new(product)),
            }),
    );
    changes.push(barrier(1));
    apply(&mut index, &resolver, &changes).unwrap();
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                position: 2,
                output_index: 0,
                event: "parsed-0".into(),
                change: ConfirmedChange::Archived,
            },
            barrier(2),
        ],
    )
    .unwrap();
    let remaining: String = index
        .query_row("SELECT id FROM event", [], |row| row.get(0))
        .unwrap();
    assert_eq!(remaining, ids[1]);
    apply(
        &mut index,
        &resolver,
        &[
            Confirmed {
                position: 3,
                output_index: 0,
                event: "parsed-0".into(),
                change: ConfirmedChange::Restored,
            },
            barrier(3),
        ],
    )
    .unwrap();
    let actual = index
        .prepare("SELECT id FROM event ORDER BY id")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    let mut expected = ids;
    expected.sort();
    assert_eq!(actual, expected);
}

mod receipts;
