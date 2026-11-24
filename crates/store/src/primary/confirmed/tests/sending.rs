use super::*;
use crate::primary::products::{SourceRecord, hidden_messages, needs_sending};

fn message(input: &str, offset: u64) -> Product {
    let mut step = crate::ParseStep::new(Default::default(), Default::default(), 0..100, true);
    step.parse_version = "synthetic=1".into();
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
            response: None,
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
        stated_fields: None,
        declared_injections: None,
        declared_attachment_bindings: None,
    }
}

fn live(position: i64, event: &str, product: Product) -> Confirmed {
    Confirmed {
        sinex_context: None,
        position,
        output_index: 0,
        event: event.into(),
        change: ConfirmedChange::Live(Box::new(Product::Interpreted {
            reading: product.reading().unwrap().clone(),
            record: SourceRecord::of(&product).unwrap().unwrap(),
        })),
    }
}

fn unsent(index: &Connection, reading: &crate::primary::Reading, product: &Product) -> bool {
    needs_sending(index, reading, &SourceRecord::of(product).unwrap().unwrap()).unwrap()
}

#[test]
fn confirmed_event_is_not_sent_again_in_any_state() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let first = message("first-source", 10);
    let independent = message("second-source", 10);
    let Product::Message { facts, reading, .. } = &first else {
        unreachable!("fixture message")
    };
    let mut step = crate::ParseStep::new(Default::default(), Default::default(), 0..100, true);
    step.parse_version = reading.parse_version.clone();
    step.output.messages.push((**facts).clone());
    step.prepare();
    assert!(unsent(&index, first.reading().unwrap(), &first));
    apply(
        &mut index,
        &[live(1, "first", first.clone()), complete_at(1)],
    )
    .unwrap();
    assert!(!unsent(&index, first.reading().unwrap(), &first));
    assert!(unsent(&index, independent.reading().unwrap(), &independent));
    for (position, change, suppressed) in [
        (2, ConfirmedChange::Archived, true),
        (3, ConfirmedChange::Restored, false),
        (4, ConfirmedChange::Tombstoned, true),
    ] {
        apply(
            &mut index,
            &[
                Confirmed {
                    sinex_context: None,
                    position,
                    output_index: 0,
                    event: "first".into(),
                    change,
                },
                complete_at(position),
            ],
        )
        .unwrap();
        assert!(!unsent(&index, first.reading().unwrap(), &first));
        assert_eq!(
            hidden_messages(&index, &reading.input, reading.chain, &step)
                .unwrap()
                .contains(&0),
            suppressed,
        );
    }
}

#[test]
fn covered_absent_event_is_not_sent_but_unread_tail_is() {
    let directory = tempfile::tempdir().unwrap();
    let mut index = cache(&directory.path().join("index.db"));
    let _archive = crate::Archive::on_sinex(directory.path(), &directory.path().join("bytes"));
    let original = message("source", 10);
    let mut reading = original.reading().unwrap().clone();
    reading.parsed_upto = 30;
    reading.progress = polylogue_model::batch::StepProgress::Partway;
    let progress = Product::ParseProgress {
        reading: reading.clone(),
        counts: Default::default(),
        stopped_attempt: None,
    };
    apply(&mut index, &[live(1, "covered", progress), complete_at(1)]).unwrap();
    assert!(!unsent(&index, &reading, &original));
    assert!(unsent(&index, &reading, &message("source", 40)));
    reading.parse_version = "synthetic=2".into();
    assert!(unsent(&index, &reading, &original));
}
