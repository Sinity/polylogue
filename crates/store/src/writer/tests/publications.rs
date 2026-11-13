use super::*;
use crate::primary::{ParsedSource, PublicationReceipt};

pub(super) fn publication(frame: &str, number: i64) -> PublicationReceipt {
    PublicationReceipt {
        producer: "fixture-producer".into(),
        frame: frame.into(),
        number,
        commit_seq: Some(number + 100),
        root_material: "00000000-0000-0000-0000-000000000001".into(),
    }
}

fn interpretation(text: &str, version: &str) -> Batch {
    let mut batch = batch(
        "original",
        Route::Live,
        0,
        vec![row(
            12,
            text_message(
                "message",
                "session",
                Role::Assistant,
                text,
                Some("end_turn"),
            ),
        )],
        &["session"],
    );
    batch.change = Change::Reparse { chain: 1 };
    batch.observation = None;
    batch.parse[0].reading_version = version.into();
    batch.parse[0].prepare();
    batch
}

pub(super) fn apply(
    index: &mut Connection,
    mut batch: Batch,
    publication: &PublicationReceipt,
) -> Result<()> {
    batch.parse[0].prepare();
    super::super::source_eligibility::retain_source_records(
        index,
        &batch.input,
        1,
        batch.length,
        &mut batch.parse[0],
    )?;
    let root = std::path::Path::new(index.path().unwrap())
        .parent()
        .unwrap();
    let archive = Archive::on_sinex(root, &root.join("bytes"));
    let resolver = resolve::Resolver::new(&archive, index, None)?;
    resolver.batches(&mut [&mut batch])?;
    let texts = crate::text_rows::TextIds::new(index, false)?;
    let text_transaction = texts.transaction();
    let tx = index.transaction()?;
    let counter = bump_counter(&tx)?;
    let changes = crate::change_summary::ChangeSummary::start(&tx)?;
    let step = &batch.parse[0];
    rows::write_published_rows(
        &texts,
        &tx,
        &batch,
        step,
        step.resolution.as_ref().unwrap(),
        counter,
        None,
        &fixture_receipt(),
        publication,
        step.reset && step.delivery.is_none(),
        &changes,
    )?;
    rows::finish(&tx, &changes, None, &texts)?;
    tx.commit()?;
    text_transaction.commit();
    Ok(())
}

fn source(batch: &Batch) -> ParsedSource {
    ParsedSource {
        root_material: publication("unused", 1).root_material,
        reading: crate::primary::Reading::from_step(
            &batch.input,
            1,
            Some(batch.length),
            &batch.parse[0],
        ),
    }
}

fn text(index: &Connection) -> String {
    let hash: Vec<u8> = index
        .query_row("SELECT version_hash FROM message_source", [], |row| {
            row.get(0)
        })
        .unwrap();
    crate::versions::read_message(index, &hash)
        .unwrap()
        .unwrap()
        .0
        .text()
}

#[test]
fn native_ack_does_not_cover_a_rolled_back_interpretation_at_an_old_cursor() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("index.db");
    let mut index = Connection::open(&path).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let original = interpretation("original interpretation", "synthetic=1");
    let first = publication("first-frame", 11);
    apply(&mut index, original.clone(), &first).unwrap();
    assert_eq!(text(&index), "original interpretation");
    let replacement = interpretation("new interpretation", "synthetic=2");
    let second = publication("second-frame", 12);
    let processed = |index: &Connection, batch: &Batch, receipt: &PublicationReceipt| {
        crate::primary::projection::processed_source(index, &source(batch), receipt).unwrap()
    };
    assert!(processed(&index, &original, &first));
    assert!(!processed(&index, &replacement, &second));
    index.execute_batch("CREATE TRIGGER reject_checkpoint BEFORE INSERT ON parse_state WHEN new.reading_version='synthetic=2' BEGIN SELECT raise(ABORT,'fixture interruption'); END").unwrap();
    assert!(apply(&mut index, replacement.clone(), &second).is_err());
    assert_eq!(text(&index), "original interpretation");
    assert!(!processed(&index, &replacement, &second));
    drop(index);
    let mut index = Connection::open(&path).unwrap();
    index
        .execute_batch("DROP TRIGGER reject_checkpoint")
        .unwrap();
    assert!(!processed(&index, &replacement, &second));
    apply(&mut index, replacement.clone(), &second).unwrap();
    assert_eq!(text(&index), "new interpretation");
    assert!(processed(&index, &replacement, &second));
    assert!(processed(&index, &original, &first));
    let mut wrong = second.clone();
    wrong.frame = "another-frame".into();
    assert!(!processed(&index, &replacement, &wrong));
    wrong = second.clone();
    wrong.root_material = "00000000-0000-0000-0000-000000000002".into();
    assert!(!processed(&index, &replacement, &wrong));
    wrong = second.clone();
    wrong.producer = "another-producer".into();
    assert!(!processed(&index, &replacement, &wrong));
}

#[test]
fn native_parser_diagnostic_lifecycle_preserves_the_local_model_and_processed_source() {
    use crate::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let mut index = Connection::open(directory.path().join("index.db")).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let original = interpretation("locally parsed original", "synthetic=1");
    let receipt = publication("original-frame", 11);
    apply(&mut index, original.clone(), &receipt).unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let diagnostic = Product::ParseProgress {
        reading: source(&original).reading,
        counts: Default::default(),
        stopped_attempt: None,
    };
    for (position, change) in [
        (
            1,
            ConfirmedChange::Live(Box::new(crate::primary::Product::Interpreted {
                reading: diagnostic.reading().unwrap().clone(),
                record: crate::primary::SourceRecord::of(&diagnostic)
                    .unwrap()
                    .unwrap(),
            })),
        ),
        (2, ConfirmedChange::Archived),
        (3, ConfirmedChange::Restored),
        (4, ConfirmedChange::Tombstoned),
    ] {
        crate::primary::projection::apply(
            &crate::text_rows::TextIds::new(&index, false).unwrap(),
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "diagnostic".into(),
                    change,
                },
                Confirmed {
                    native: None,
                    position: 0,
                    output_index: 0,
                    event: String::new(),
                    change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
                        position,
                        delivery: None,
                    })),
                },
            ],
        )
        .unwrap();
        assert_eq!(text(&index), "locally parsed original");
        assert!(
            crate::primary::projection::processed_source(&index, &source(&original), &receipt)
                .unwrap()
        );
    }
}

#[test]
fn an_archived_value_does_not_turn_a_processed_incomplete_tail_into_an_unknown_prefix() {
    use crate::primary::{Confirmed, ConfirmedChange, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let mut index = Connection::open(directory.path().join("index.db")).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let mut original = interpretation("archived original", "synthetic=1");
    original.parse[0].parsed_upto = 90;
    let message = crate::primary::products::parse(
        &original.input,
        1,
        Some(original.length),
        &original.parse[0],
    )
    .unwrap()
    .into_iter()
    .find(|product| matches!(product, crate::primary::Product::Message { .. }))
    .unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    for (position, change) in [
        (
            1,
            ConfirmedChange::Live(Box::new(crate::primary::Product::Interpreted {
                reading: message.reading().unwrap().clone(),
                record: crate::primary::SourceRecord::of(&message).unwrap().unwrap(),
            })),
        ),
        (2, ConfirmedChange::Archived),
    ] {
        crate::primary::projection::apply(
            &crate::text_rows::TextIds::new(&index, false).unwrap(),
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "hidden-message".into(),
                    change,
                },
                Confirmed {
                    native: None,
                    position: 0,
                    output_index: 0,
                    event: String::new(),
                    change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
                        position,
                        delivery: None,
                    })),
                },
            ],
        )
        .unwrap();
    }
    let receipt = publication("processed-hidden-source", 12);
    apply(&mut index, original.clone(), &receipt).unwrap();
    assert_eq!(
        index
            .query_row("SELECT count(*) FROM message", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .query_row(
                "SELECT status,parsed_upto,at_length FROM parse_state",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                }
            )
            .unwrap(),
        ("waiting".into(), 90, 100)
    );
    assert!(
        crate::primary::projection::processed_source(&index, &source(&original), &receipt).unwrap()
    );

    // The parser's private continuation can still name the suppressed response
    // when its next slice contains only an incomplete trailing record.
    let row = original.parse[0].complete_output().unwrap().messages[0].clone();
    let version = row.message.content_hash().unwrap();
    assert!(
        crate::versions::read_message(&index, &version.0)
            .unwrap()
            .is_none()
    );
    index
        .execute(
            "INSERT INTO parser_context_state SELECT pk,1,'synthetic=1',90 FROM input_ref WHERE id='original'",
            [],
        )
        .unwrap();
    let delta = polylogue_model::parser_context::Delta {
        version: 1,
        base: 90,
        through: 90,
        entries: vec![polylogue_model::parser_context::Entry {
            key: polylogue_model::parser_context::Key::Response {
                namespace: String::new(),
                session: row.message.session.to_string(),
                key: "native-response".into(),
            },
            value: polylogue_model::parser_context::Reference::Response {
                message: row.message.id.clone(),
                version,
                offset: row.offset,
                end: row.end,
            },
        }],
    };
    let mut tail = original.clone();
    let mut step = ParseStep::new(Default::default(), Default::default(), 90..90, true);
    step.reading_version = "synthetic=1".into();
    step.checkpoint = Some(polylogue_model::batch::ParserCheckpoint::new(
        "synthetic=1".into(),
        90,
        Some(serde_json::json!({ "_polylogue_context": delta })),
    ));
    step.prepare();
    tail.parse = vec![step];
    let receipt = publication("processed-hidden-tail", 13);
    apply(&mut index, tail.clone(), &receipt).unwrap();
    assert_eq!(
        index
            .query_row(
                "SELECT status,parsed_upto,at_length,checkpoint FROM parse_state",
                [],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<Vec<u8>>>(3)?,
                )),
            )
            .unwrap(),
        ("waiting".into(), 90, 100, None)
    );
    for table in ["message", "parser_context", "parser_context_state"] {
        assert_eq!(
            index
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0,
            "{table}",
        );
    }
    assert!(
        crate::primary::projection::processed_source(&index, &source(&tail), &receipt).unwrap()
    );
}

pub(super) fn confirm(index: &mut Connection, archive: &Archive, batch: &Batch, position: i64) {
    use crate::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    let mut batch = batch.clone();
    batch.parse[0].prepare();
    let resolver = resolve::Resolver::new(archive, index, None).unwrap();
    let mut events =
        crate::primary::products::parse(&batch.input, 1, Some(batch.length), &batch.parse[0])
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(ordinal, product)| Confirmed {
                native: None,
                position,
                output_index: ordinal as u64,
                event: format!("{}:{ordinal}", batch.input.id),
                change: ConfirmedChange::Live(Box::new(Product::Interpreted {
                    reading: product.reading().unwrap().clone(),
                    record: crate::primary::SourceRecord::of(&product).unwrap().unwrap(),
                })),
            })
            .collect::<Vec<_>>();
    events.push(Confirmed {
        native: None,
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
            position,
            delivery: None,
        })),
    });
    crate::primary::projection::apply(
        &crate::text_rows::TextIds::new(index, false).unwrap(),
        index,
        &resolver,
        &events,
    )
    .unwrap();
}

#[test]
fn native_fact_admissions_follow_identity_through_filtering_and_reordering() {
    use crate::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let mut index = Connection::open(archive.index_db()).unwrap();
    crate::schema::ensure_primary_index(&index).unwrap();
    let resolver = resolve::Resolver::new(&archive, &index, None).unwrap();
    let mut original = batch(
        "original",
        Route::Live,
        0,
        vec![
            row(
                12,
                text_message(
                    "hidden",
                    "session",
                    Role::Assistant,
                    "hidden",
                    Some("end_turn"),
                ),
            ),
            row(
                30,
                text_message("kept", "session", Role::Assistant, "kept", Some("end_turn")),
            ),
            row(
                60,
                text_message(
                    "other",
                    "session",
                    Role::Assistant,
                    "other",
                    Some("end_turn"),
                ),
            ),
        ],
        &["session", "child"],
    );
    original.change = Change::Reparse { chain: 1 };
    original.observation = None;
    let hidden = original.parse[0].output.messages[0].message.id.clone();
    let kept = original.parse[0].output.messages[1].message.id.clone();
    let other = original.parse[0].output.messages[2].message.id.clone();
    let mut document = polylogue_model::Document::new(
        polylogue_model::DocumentId::parse("claude-code:document").unwrap(),
        polylogue_model::DocumentPurpose::Context,
    );
    document.blocks.push(Block::Text {
        text: "document".into(),
    });
    original.parse[0]
        .output
        .documents
        .push(polylogue_model::DocumentRow {
            offset: 70,
            end: 71,
            document: document.clone(),
        });
    let link = Link {
        kind: polylogue_model::LinkKind::Spawn,
        copies_prefix: false,
        how: None,
        parent: sid("session"),
        child: sid("child"),
        at: None,
        call_id: None,
        evidence: "explicit".into(),
    };
    original.parse[0].output.links.push(link.clone());
    original.parse[0].output.events.push(serde_json::from_value(serde_json::json!({
        "session":"claude-code:session", "kind":"captured.event", "range":[80,81], "data":{"original":true}
    })).unwrap());
    original.parse[0].output.messages[2]
        .message
        .blocks
        .push(Block::Image(polylogue_model::Attachment {
            blob: Some(BlobHash::of(b"unavailable original attachment")),
            size: Some(4),
            ..Default::default()
        }));
    let mut confirmed = original.clone();
    confirmed.parse[0].prepare();
    let products = crate::primary::products::parse(
        &confirmed.input,
        1,
        Some(confirmed.length),
        &confirmed.parse[0],
    )
    .unwrap();
    let mut expected = std::collections::BTreeMap::new();
    let hidden_reference = format!("message:{hidden}:12");
    let mut hidden_event = String::new();
    let mut events = Vec::new();
    for (ordinal, product) in products.into_iter().enumerate() {
        let position = 10 + ordinal as i64;
        let record = crate::primary::SourceRecord::of(&product).unwrap().unwrap();
        let event = format!("fact-{ordinal}");
        if record.reference == hidden_reference {
            hidden_event = event.clone();
        }
        expected.insert(record.reference.clone(), position as u64);
        events.push(Confirmed {
            native: None,
            position,
            output_index: 0,
            event,
            change: ConfirmedChange::Live(Box::new(Product::Interpreted {
                reading: product.reading().unwrap().clone(),
                record,
            })),
        });
    }
    let position = events.last().unwrap().position + 1;
    events.push(Confirmed {
        native: None,
        position,
        output_index: 0,
        event: hidden_event,
        change: ConfirmedChange::Archived,
    });
    events.push(Confirmed {
        native: None,
        position: 0,
        output_index: 0,
        event: String::new(),
        change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
            position,
            delivery: None,
        })),
    });
    crate::primary::projection::apply(
        &crate::text_rows::TextIds::new(&index, false).unwrap(),
        &mut index,
        &resolver,
        &events,
    )
    .unwrap();
    original.parse[0].output.messages.swap(0, 2);
    apply(
        &mut index,
        original,
        &publication("interpreted-original", 20),
    )
    .unwrap();
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(reader.message_with_projection(&hidden).unwrap().is_none());
    for (message, offset) in [(&kept, 30), (&other, 60)] {
        assert_eq!(
            reader.message_admission(message).unwrap(),
            Some(expected[&format!("message:{message}:{offset}")])
        );
    }
    let shown = reader.message_with_projection(&other).unwrap().unwrap().0;
    assert!(
        shown
            .blocks
            .iter()
            .any(|block| matches!(block, Block::Image(attachment) if attachment.blob.is_none()))
    );
    assert_eq!(
        reader.session_commit(&sid("session")).unwrap(),
        Some(expected[&format!("session:{}", sid("session"))])
    );
    assert_eq!(
        reader.document_commit(&document.id).unwrap(),
        Some(expected[&format!("document:{}:70", document.id)])
    );
    assert_eq!(
        reader.link_commit(&link).unwrap(),
        Some(expected[&link.selection_tag()])
    );
    let actual: u64 = index
        .query_row("SELECT admitted FROM event_source", [], |row| row.get(0))
        .unwrap();
    let expected_event = expected
        .iter()
        .find(|(reference, _)| reference.starts_with("event:"))
        .unwrap()
        .1;
    assert_eq!(actual, *expected_event);
    assert_ne!(actual, fixture_receipt().sequence as u64);
}
