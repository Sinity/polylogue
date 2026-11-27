use super::*;
use crate::output;
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
    // Installs the built index: the cancellation test below holds a reader
    // while it writes, which needs the installed index's WAL
    // (docs/architecture.md, Durability).
    writer.writer.derive().unwrap();
    (writer, id)
}

fn options() -> ReadOptions {
    ReadOptions {
        view: View::Transcript,
        ..Default::default()
    }
}

fn selected(_: &Source) -> Result<bool, crate::Error> {
    Ok(false)
}
fn ready(
    _: &polylogue_model::MessageId,
    _: u64,
    _: u64,
    _: Option<&str>,
) -> Result<ReadyCall, crate::Error> {
    panic!("a plain document block has no receiving attachment action")
}
fn layout<'a>(
    options: &'a ReadOptions,
    facts: &'a LayoutFacts,
    backwards: bool,
) -> layout_facts::Layout<'a> {
    layout_facts::Layout {
        options,
        facts,
        selected: &selected,
        ready_attachment: &ready,
        per_item_source: false,
        backwards,
        page_start: None,
    }
}

fn item(record: &ReadRecord) -> ReadItem {
    let ReadRecord::Presentation { parts } = record else {
        panic!("page exposed a partial stored document")
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
    writer.corrupt_document_block(2);
    let pool = ReadPool::open(&writer.archive).unwrap();
    let reader = pool.reader().unwrap();
    let options = options();
    let facts = LayoutFacts::default();
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
            &layout(&options, &facts, false),
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
    let mut visited = 0;
    let refusal = super::super::records::reference(
        &reader,
        &polylogue_model::read::ReadTarget::Document(id),
        false,
        None,
        &mut |_, _, _| {
            visited += 1;
            Ok(true)
        },
    )
    .unwrap_err();
    // An undecodable stored body is an io failure, not a missing or
    // not-yet-available document.
    assert_eq!(refusal.code(), polylogue_model::Code::Io);
    assert_eq!(visited, 0);
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
        zone: None,
        snapshot: None,
        options: None,
    }
}

#[test]
fn document_after_and_before_seek_original_blocks_and_keep_full_item_digest_checks() {
    for backwards in [false, true] {
        let (writer, id) = fixture();
        writer.corrupt_document_block(if backwards { 2 } else { 0 });
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
        let facts = LayoutFacts::default();
        assert!(
            visit(
                &reader,
                &id,
                7,
                Some(start),
                &layout(&options, &facts, backwards),
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
                &layout(&options, &facts, backwards),
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
    use polylogue_store::{cancellation::Cancellation, read::limit::ReadLimit};
    for reset in [false, true] {
        let (writer, id) = fixture();
        writer.corrupt_document_block(1);
        let cancellation = Cancellation::default();
        ReadLimit::without_deadline(cancellation.clone()).scope(|| {
            let pool = ReadPool::open(&writer.archive).unwrap();
            let reader = pool.reader().unwrap();
            let options = options();
            let facts = LayoutFacts::default();
            let mut emitted = 0;
            let result = visit(
                &reader,
                &id,
                0,
                None,
                &layout(&options, &facts, false),
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
            let kind = failure.code();
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
fn document_views_without_a_document_form_refuse_unsupported_target() {
    for view in [
        View::Dialogue,
        View::Overview,
        View::Tree,
        View::Changes,
        View::Live,
    ] {
        let options = ReadOptions {
            view,
            ..Default::default()
        };
        let reason =
            format!("view {view} does not read document:claude-ai:doc; use transcript or raw");
        let mut footer = ReadFooter::default();
        let refusal =
            output::document_omission("document:claude-ai:doc", &options, false, &mut footer)
                .unwrap_err();
        assert_eq!(refusal.code(), polylogue_model::Code::BadRequest);
        assert_eq!(refusal.to_string(), reason);
        assert!(footer.coverage.is_empty());
        let omitted =
            output::document_omission("document:claude-ai:doc", &options, true, &mut footer)
                .unwrap()
                .unwrap();
        assert_eq!(
            item(&omitted),
            ReadItem {
                source: Source::record("document:claude-ai:doc"),
                kind: ItemKind::Omission {
                    count: ValueState::known(1),
                    unit: "document".into(),
                    why: reason,
                    continue_call: None,
                },
            }
        );
        output::document_omission("document:claude-ai:other", &options, true, &mut footer)
            .unwrap()
            .unwrap();
        assert_eq!(footer.coverage.len(), 1);
        assert_eq!(footer.coverage[0].view, view);
        assert_eq!(footer.coverage[0].omitted, 2);
        assert_eq!(footer.coverage[0].target, TargetKind::Document);
    }
}

#[test]
fn documents_are_not_omitted_from_views_that_present_them() {
    let mut footer = ReadFooter::default();
    for view in [View::Transcript, View::Raw] {
        let options = ReadOptions {
            view,
            ..Default::default()
        };
        assert!(
            output::document_omission("document:claude-ai:doc", &options, true, &mut footer)
                .unwrap()
                .is_none()
        );
    }
    assert!(footer.coverage.is_empty());
}
