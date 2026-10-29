use super::*;
use polylogue_model::SessionId;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

fn at(kind: ReadRecordKind, ordinal: u64) -> ReadRecordLocator {
    ReadRecordLocator { kind, ordinal }
}
#[test]
fn previous_coordinates_invert_the_existing_session_family_order() {
    use ReadRecordKind::{Event, Link, Message, Session};
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    let target = fixture
        .session(SessionFacts {
            id: "test:target",
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "test:child",
            ..Default::default()
        })
        .unwrap();
    for id in ["test:second", "test:first"] {
        fixture
            .message(
                target,
                MessageFacts {
                    id,
                    role: "user",
                    text: Some("payload is irrelevant to metadata seek"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.event(target, "notice", None).unwrap();
    fixture.event(target, "notice", None).unwrap();
    fixture
        .link("test:target", "test:child", "continuation", "fork")
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let reference = Reference::Session(SessionId::parse("test:target").unwrap());
    let mut current = last_record(&reader, &reference, false).unwrap();
    let mut seen = Vec::new();
    while let Some(position) = current {
        seen.push(position);
        current = previous_record(&reader, &reference, false, &position).unwrap();
    }
    assert_eq!(
        seen,
        vec![
            at(Link, 0),
            at(Event, 1),
            at(Event, 0),
            at(Message, 1),
            at(Message, 0),
            at(Session, 0)
        ]
    );
    assert!(previous_record(&reader, &reference, false, &at(Message, 2)).is_err());
    assert!(previous_record(&reader, &reference, false, &at(ReadRecordKind::Blob, 0)).is_err());
}
#[test]
fn singleton_and_empty_session_edges_have_no_shifted_fallback() {
    use ReadRecordKind::{Message, Session};
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    fixture
        .session(SessionFacts {
            id: "test:empty",
            ..Default::default()
        })
        .unwrap();
    let other = fixture
        .session(SessionFacts {
            id: "test:other",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            other,
            MessageFacts {
                id: "test:present",
                role: "user",
                text: Some("record"),
                ..Default::default()
            },
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let reference = Reference::parse("session:test:empty").unwrap();
    assert_eq!(
        last_record(&reader, &reference, false).unwrap(),
        Some(at(Session, 0))
    );
    assert_eq!(
        previous_record(&reader, &reference, false, &at(Session, 0)).unwrap(),
        None
    );
    assert_eq!(
        last_record(
            &reader,
            &Reference::parse("session:test:absent").unwrap(),
            false
        )
        .unwrap(),
        None
    );
    assert!(
        previous_record(
            &reader,
            &Reference::parse("session:test:absent").unwrap(),
            false,
            &at(Session, 0)
        )
        .is_err()
    );
    let direct = Reference::parse("message:test:present").unwrap();
    assert_eq!(
        last_record(&reader, &direct, false).unwrap(),
        Some(at(Message, 0))
    );
    assert_eq!(
        previous_record(&reader, &direct, false, &at(Message, 0)).unwrap(),
        None
    );
    assert!(previous_record(&reader, &direct, false, &at(Message, 1)).is_err());
}
