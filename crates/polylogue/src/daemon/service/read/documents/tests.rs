use super::*;
use polylogue_model::{BlobHash, Document, DocumentPurpose, DocumentRow, Output, read::*};
use polylogue_render::pager::{PageBuilder, Push};
use polylogue_store::{ReadPool, StoreError, fixture::writer::WriterArchive};

fn fixture() -> (WriterArchive, DocumentId) {
    let mut writer = WriterArchive::new();
    let id = DocumentId::parse("claude-ai:lazy-document").unwrap();
    let mut document = Document::new(id.clone(), DocumentPurpose::Context);
    document.blocks = ["first", "second", "third"]
        .into_iter()
        .map(|text| Block::Text { text: text.into() })
        .collect();
    writer.write_output(
        "document-input",
        "claude-ai",
        "/synthetic/document.json",
        Output {
            documents: vec![DocumentRow {
                offset: 0,
                end: 1,
                document,
            }],
            ..Default::default()
        },
        1,
    );
    (writer, id)
}

fn corrupt(writer: &WriterArchive, ordinal: u64) {
    let index = rusqlite::Connection::open(writer.archive.index_db()).unwrap();
    assert_eq!(
        index
            .execute(
                "UPDATE document_block SET body=x'00' WHERE idx=?1",
                [i64::try_from(ordinal).unwrap()],
            )
            .unwrap(),
        1
    );
}

fn options() -> ReadOptions {
    ReadOptions {
        view: View::Transcript,
        ..Default::default()
    }
}

fn selected(_: &Source) -> Result<bool, crate::daemon::Failure> {
    Ok(false)
}
fn ready(
    _: &polylogue_model::MessageId,
    _: u64,
    _: u64,
    _: Option<&str>,
) -> Result<ReadyCall, crate::daemon::Failure> {
    panic!("a plain document block has no receiving attachment action")
}
fn layout<'a>(
    options: &'a ReadOptions,
    evidence: &'a ProjectionEvidence,
    backwards: bool,
) -> evidence::Layout<'a> {
    evidence::Layout {
        options,
        evidence,
        selected: &selected,
        ready_attachment: &ready,
        backwards,
        anchor: None,
    }
}

fn item(record: &ReadRecord) -> ReadItem {
    let ReadRecord::Presentation { parts } = record else {
        panic!("page exposed a partial canonical document")
    };
    let [ReadPart::Item(item)] = parts.as_slice() else {
        panic!("one block item per callback")
    };
    item.clone()
}

fn expected(id: &DocumentId, ordinal: u64, text: &str) -> ReadItem {
    ReadItem {
        source: Source {
            block: Some(ordinal),
            ..Source::record(format!("document:{id}"))
        },
        kind: ItemKind::Text { text: text.into() },
    }
}

#[test]
fn a_document_page_stops_before_corrupt_later_blocks_but_full_transfer_refuses() {
    let (writer, id) = fixture();
    corrupt(&writer, 2);
    let pool = ReadPool::open(&writer.archive).unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let evidence = ProjectionEvidence::default();
    let mut page = PageBuilder::new(
        header(&id),
        PageBudget {
            items: std::num::NonZeroUsize::new(1).unwrap(),
            ..Default::default()
        },
        None,
        ReadFraming::Query {
            question: QuestionId([7; 32]),
        },
    )
    .unwrap();
    let mut seen = Vec::new();
    assert!(
        !visit(
            &reader,
            &id,
            7,
            None,
            &layout(&options, &evidence, false),
            &mut |record, at, local| {
                let value = item(record);
                seen.push((value.clone(), at, local));
                Ok(page.push_at(at, local.unwrap(), value).unwrap() != Push::Full)
            }
        )
        .unwrap()
    );
    assert_eq!(
        seen,
        vec![
            (expected(&id, 0, "first"), position(7, 1), Some(0)),
            (expected(&id, 1, "second"), position(7, 2), Some(0)),
        ]
    );
    assert_eq!(page.position().unwrap().selection, position(7, 2));
    let page = page.finish(ReadFooter::default()).unwrap();
    assert_eq!(page.items, vec![expected(&id, 0, "first")]);
    let mut canonical = 0;
    assert!(
        super::super::records::reference(
            &reader,
            &polylogue_model::read::ReadTarget::Document(id),
            false,
            None,
            &mut |_, _, _| {
                canonical += 1;
                Ok(true)
            }
        )
        .is_err()
    );
    assert_eq!(canonical, 0);
}

fn header(id: &DocumentId) -> ReadHeader {
    ReadHeader {
        target: format!("document:{id}"),
        view: View::Transcript,
        title: ValueState::NotRecorded,
        kind: HeaderKind::Record {
            record_type: "document".into(),
        },
        figures: vec![],
    }
}

#[test]
fn document_after_and_before_seek_original_blocks_and_keep_full_item_digest_checks() {
    for backwards in [false, true] {
        let (writer, id) = fixture();
        corrupt(&writer, if backwards { 2 } else { 0 });
        let pool = ReadPool::open(&writer.archive).unwrap();
        let reader = pool.reader().unwrap();
        let ordinal = if backwards { 1 } else { 2 };
        let original = expected(&id, ordinal, if backwards { "second" } else { "third" });
        let start = ItemPlace {
            selection: position(7, ordinal + 1),
            item: 0,
            character: 0,
            item_hash: BlobHash::of(&serde_json::to_vec(&original).unwrap()),
        };
        let framing = ReadFraming::Query {
            question: QuestionId([7; 32]),
        };
        let mut page = if backwards {
            PageBuilder::new_before(header(&id), PageBudget::default(), start, framing)
        } else {
            PageBuilder::new(header(&id), PageBudget::default(), Some(start), framing)
        }
        .unwrap();
        let options = options();
        let evidence = ProjectionEvidence::default();
        assert!(
            visit(
                &reader,
                &id,
                7,
                Some(start),
                &layout(&options, &evidence, backwards),
                &mut |record, at, local| {
                    assert_eq!(
                        page.push_at(at, local.unwrap(), item(record)).unwrap(),
                        Push::Accepted
                    );
                    Ok(true)
                }
            )
            .unwrap()
        );
        let document = page.finish(ReadFooter::default()).unwrap();
        assert_eq!(
            document.items,
            vec![if backwards {
                expected(&id, 0, "first")
            } else {
                original
            }]
        );
        let mut changed = start;
        changed.item_hash = BlobHash::of(b"another complete item");
        let framing = ReadFraming::Query {
            question: QuestionId([7; 32]),
        };
        let mut page = if backwards {
            PageBuilder::new_before(header(&id), PageBudget::default(), changed, framing)
        } else {
            PageBuilder::new(header(&id), PageBudget::default(), Some(changed), framing)
        }
        .unwrap();
        let mut mismatch = None;
        assert!(
            !visit(
                &reader,
                &id,
                7,
                Some(changed),
                &layout(&options, &evidence, backwards),
                &mut |record, at, local| {
                    mismatch = Some(page.push_at(at, local.unwrap(), item(record)).unwrap_err());
                    Ok(false)
                }
            )
            .unwrap()
        );
        assert_eq!(
            mismatch,
            Some(polylogue_render::pager::PageError::ItemChanged)
        );
    }
}

#[test]
fn document_block_decode_obeys_cancellation_and_generation_changes_between_items() {
    use polylogue_store::read::limit::{ReadCancellation, ReadLimit};
    for reset in [false, true] {
        let (writer, id) = fixture();
        corrupt(&writer, 1);
        let cancellation = ReadCancellation::default();
        ReadLimit::without_deadline(cancellation.clone()).scope(|| {
            let pool = ReadPool::open(&writer.archive).unwrap();
            let reader = pool.reader().unwrap();
            let options = options();
            let evidence = ProjectionEvidence::default();
            let mut emitted = 0;
            let result = visit(
                &reader,
                &id,
                0,
                None,
                &layout(&options, &evidence, false),
                &mut |record, _, _| {
                    assert_eq!(item(record), expected(&id, 0, "first"));
                    emitted += 1;
                    if reset {
                        writer.archive.advance_generation_for_test().unwrap();
                    } else {
                        cancellation.cancel();
                    }
                    Ok(true)
                },
            );
            let Err(failure) = &result else {
                panic!("changed request continued: {result:?}")
            };
            let kind = failure.code().to_owned();
            let expected = if reset {
                StoreError::ScopeChanged
            } else {
                StoreError::ReadCancelled
            };
            assert_eq!(kind, expected.code());
            assert_eq!(emitted, 1);
        });
    }
}

#[test]
fn unsupported_document_pages_preserve_direct_errors_and_mixed_coverage() {
    let options = ReadOptions {
        view: View::Dialogue,
        ..Default::default()
    };
    let mut footer = ReadFooter::default();
    assert!(
        output::document_omission("document:claude-ai:doc", &options, false, &mut footer).is_err()
    );
    assert!(footer.coverage.is_empty());
    let omitted = output::document_omission("document:claude-ai:doc", &options, true, &mut footer)
        .unwrap()
        .unwrap();
    assert!(matches!(item(&omitted).kind, ItemKind::Omission { .. }));
    assert_eq!(footer.coverage.len(), 1);
    assert_eq!(footer.coverage[0].omitted, 1);
    assert_eq!(footer.coverage[0].target, TargetKind::Document);
}
