use super::*;
use crate::primary::products::{SourceRecord, allowed, needs_publication};

fn message(input: &str, offset: u64) -> Product {
    let mut step = crate::ParseStep::new(Default::default(), Default::default(), 0..100, true);
    step.reading_version = "synthetic=1".into();
    Product::Message {
        reading: crate::primary::Reading::from_step(
            &crate::InputKey {
                id: input.into(),
                format: "synthetic".into(),
                route: polylogue_model::Route::Live,
            },
            1,
            Some(100),
            &step,
        ),
        facts: Box::new(polylogue_model::MessageRow {
            offset,
            end: offset + 10,
            page_text: false,
            id_kind: polylogue_model::IdKind::Native,
            message: crate::fixture::writer::message(
                "message",
                "session",
                polylogue_model::Role::Assistant,
                1,
                vec![polylogue_model::Block::Text {
                    text: "original body".into(),
                }],
            ),
        }),
        order: None,
        projection: None,
        native_fields: None,
        declared_injections: None,
        declared_attachment_bindings: None,
    }
}

fn live(position: i64, event: &str, product: Product) -> Confirmed {
    Confirmed {
        native: None,
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(Product::Interpreted {
            reading: product.reading().unwrap().clone(),
            record: SourceRecord::of(&product).unwrap().unwrap(),
        })),
    }
}

#[test]
fn native_lifecycle_selects_exact_original_values_and_preserves_independent_sources() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let first = message("first-source", 10);
    let independent = message("second-source", 10);
    let visible = |index: &Connection, product: &Product| {
        allowed(
            index,
            product.reading().unwrap(),
            &SourceRecord::of(product).unwrap().unwrap(),
        )
        .unwrap()
    };
    assert!(
        needs_publication(
            &index,
            first.reading().unwrap(),
            &SourceRecord::of(&first).unwrap().unwrap()
        )
        .unwrap()
    );
    apply(
        &mut index,
        &resolver,
        &[
            live(1, "first", first.clone()),
            live(1, "independent", independent.clone()),
            barrier(1),
        ],
    )
    .unwrap();
    assert!(visible(&index, &first));
    assert!(visible(&index, &independent));
    for (position, change, kept) in [
        (2, ConfirmedChange::Archived, false),
        (3, ConfirmedChange::Restored, true),
        (4, ConfirmedChange::Tombstoned, false),
    ] {
        apply(
            &mut index,
            &resolver,
            &[
                Confirmed {
                    native: None,
                    position,
                    output_index: 0,
                    event: "first".into(),
                    change,
                },
                barrier(position),
            ],
        )
        .unwrap();
        assert_eq!(visible(&index, &first), kept);
        assert!(
            !needs_publication(
                &index,
                first.reading().unwrap(),
                &SourceRecord::of(&first).unwrap().unwrap()
            )
            .unwrap()
        );
        assert!(visible(&index, &independent));
    }
    drop(index);
    let index = Connection::open(directory.path().join("index.db")).unwrap();
    assert!(!visible(&index, &first));
    assert!(visible(&index, &independent));
}

#[test]
fn covered_absence_does_not_hide_an_unread_tail_or_a_new_parser_interpretation() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let resolver = crate::writer::resolve::Resolver::new(&archive, &index, None).unwrap();
    let original = message("source", 10);
    let mut reading = original.reading().unwrap().clone();
    reading.parsed_upto = 30;
    reading.progress = polylogue_model::batch::StepProgress::Partway;
    let progress = Product::ParseProgress {
        reading: reading.clone(),
        counts: Default::default(),
        stopped_attempt: None,
    };
    apply(
        &mut index,
        &resolver,
        &[live(1, "covered", progress), barrier(1)],
    )
    .unwrap();
    let record = SourceRecord::of(&original).unwrap().unwrap();
    assert!(!allowed(&index, &reading, &record).unwrap());
    let tail = SourceRecord::of(&message("source", 40)).unwrap().unwrap();
    assert!(allowed(&index, &reading, &tail).unwrap());
    reading.reading_version = "synthetic=2".into();
    assert!(allowed(&index, &reading, &record).unwrap());
}
