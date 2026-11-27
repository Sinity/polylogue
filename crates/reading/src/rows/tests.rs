use super::*;
use crate::records::{SelectedRecord, step};
use polylogue_model::SessionId;
use polylogue_model::read::ReadTarget;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

fn last_record(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
) -> Result<Option<ReadRecordLocator>, Error> {
    step(reader, target, inherited, None, ReadDirection::Before)
}
fn previous_record(
    reader: &Reader<'_>,
    target: &ReadTarget,
    inherited: bool,
    from: &ReadRecordLocator,
) -> Result<Option<ReadRecordLocator>, Error> {
    step(reader, target, inherited, Some(from), ReadDirection::Before)
}
fn read_target(text: &str) -> Result<ReadTarget, polylogue_model::read::NotReadTarget> {
    ReadTarget::try_from(polylogue_model::Reference::parse(text).unwrap())
}
fn at(kind: ReadRecordKind, ordinal: u64) -> ReadRecordLocator {
    ReadRecordLocator { kind, ordinal }
}
#[test]
fn previous_coordinates_invert_the_existing_session_family_order() {
    use ReadRecordKind::{Event, Link, Message, Session};
    use polylogue_model::{
        Block, Event as StoredEvent, IdKind, Link as StoredLink, LinkKind, MessageId, MessageRow,
        Output, Role, Session as StoredSession, SessionKind, TimeQuality,
    };
    use polylogue_store::{
        ReadPool,
        fixture::writer::{WriterArchive, message},
    };

    let mut fixture = WriterArchive::new();
    let target = SessionId::parse("test:target").unwrap();
    let child = SessionId::parse("test:child").unwrap();
    let messages = ["test:second", "test:first"]
        .into_iter()
        .enumerate()
        .map(|(ordinal, id)| {
            let mut message = message(
                id,
                "target",
                Role::User,
                ordinal as i64 + 1,
                vec![Block::Text {
                    text: "record body".into(),
                }],
            );
            message.id = MessageId::parse(id).unwrap();
            message.session = target.clone();
            MessageRow {
                response: None,
                page_text: false,
                id_kind: IdKind::Native,
                offset: ordinal as u64,
                end: ordinal as u64 + 1,
                message,
            }
        })
        .collect();
    let events = (2..4)
        .map(|offset| StoredEvent {
            session: Some(target.clone()),
            kind: "notice".into(),
            at: None,
            at_quality: TimeQuality::Unknown,
            at_precision: None,
            at_until: None,
            after: None,
            range: Some((offset, offset + 1)),
            data: serde_json::json!({"notice": offset - 1}),
        })
        .collect();
    fixture.write_output(
        "row-input",
        "claude-code",
        "/synthetic/rows.jsonl",
        Output {
            sessions: vec![
                StoredSession::new(target.clone(), SessionKind::Main),
                StoredSession::new(child.clone(), SessionKind::Main),
            ],
            messages,
            events,
            links: vec![StoredLink {
                kind: LinkKind::Fork,
                copies_prefix: false,
                reason: None,
                parent: target,
                child,
                branch_point: None,
                call_id: None,
                evidence: "fork marker".into(),
            }],
            ..Default::default()
        },
        4,
    );
    fixture.writer.derive().unwrap();
    let pool = ReadPool::open(&fixture.archive).unwrap();
    let reader = pool.reader().unwrap();
    let reference = ReadTarget::Session(SessionId::parse("test:target").unwrap());
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
    let mut current = step(&reader, &reference, false, None, ReadDirection::After).unwrap();
    let mut forward = Vec::new();
    while let Some(position) = current {
        forward.push(position);
        current = step(
            &reader,
            &reference,
            false,
            Some(&position),
            ReadDirection::After,
        )
        .unwrap();
    }
    assert_eq!(
        forward,
        vec![
            at(Session, 0),
            at(Message, 0),
            at(Message, 1),
            at(Event, 0),
            at(Event, 1),
            at(Link, 0)
        ]
    );
    // Each locator names the record the fixture put there: messages in
    // stored offset order (not id order), events in offset order.
    let mut named = Vec::new();
    crate::records::reference(
        &reader,
        &reference,
        false,
        None,
        &mut |record, _, position| {
            if let SelectedRecord::Original(record) = record {
                match *record {
                    polylogue_model::bundle::Record::Message(message) => {
                        named.push((position, message.reference));
                    }
                    polylogue_model::bundle::Record::Event(event) => {
                        named.push((position, format!("notice {}", event.event.data["notice"])));
                    }
                    _ => {}
                }
            }
            Ok(true)
        },
    )
    .unwrap();
    assert_eq!(
        named,
        [
            (at(Message, 0), "message:test:second".to_string()),
            (at(Message, 1), "message:test:first".to_string()),
            (at(Event, 0), "notice 1".to_string()),
            (at(Event, 1), "notice 2".to_string()),
        ]
    );
    for stale in [at(Message, 2), at(ReadRecordKind::Blob, 0)] {
        assert_eq!(
            previous_record(&reader, &reference, false, &stale)
                .unwrap_err()
                .code(),
            polylogue_model::Code::StaleCursor
        );
    }
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
    let reference = read_target("session:test:empty").unwrap();
    assert_eq!(
        last_record(&reader, &reference, false).unwrap(),
        Some(at(Session, 0))
    );
    assert_eq!(
        previous_record(&reader, &reference, false, &at(Session, 0)).unwrap(),
        None
    );
    assert_eq!(
        last_record(&reader, &read_target("session:test:absent").unwrap(), false).unwrap(),
        None
    );
    assert_eq!(
        previous_record(
            &reader,
            &read_target("session:test:absent").unwrap(),
            false,
            &at(Session, 0)
        )
        .unwrap_err()
        .code(),
        polylogue_model::Code::StaleCursor
    );
    let direct = read_target("message:test:present").unwrap();
    assert_eq!(
        last_record(&reader, &direct, false).unwrap(),
        Some(at(Message, 0))
    );
    assert_eq!(
        previous_record(&reader, &direct, false, &at(Message, 0)).unwrap(),
        None
    );
    assert_eq!(
        previous_record(&reader, &direct, false, &at(Message, 1))
            .unwrap_err()
            .code(),
        polylogue_model::Code::StaleCursor
    );
}

#[test]
fn lineage_issue_coordinates_resume_and_reverse_at_both_family_boundaries() {
    use ReadRecordKind::{LinkIssue, Message, Session};
    let root = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(root.path()).unwrap();
    let target = fixture
        .session(SessionFacts {
            id: "test:child",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            target,
            MessageFacts {
                id: "test:own",
                role: "user",
                text: Some("own body"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .link("test:missing", "test:child", "fork", None)
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let reference = read_target("session:test:child").unwrap();
    assert_eq!(
        previous_record(&reader, &reference, true, &at(Message, 0)).unwrap(),
        Some(at(LinkIssue, 0))
    );
    assert_eq!(
        previous_record(&reader, &reference, true, &at(LinkIssue, 0)).unwrap(),
        Some(at(Session, 0))
    );
    assert_eq!(
        previous_record(&reader, &reference, false, &at(LinkIssue, 0))
            .unwrap_err()
            .code(),
        polylogue_model::Code::StaleCursor
    );
    let mut observed = Vec::new();
    crate::records::reference(&reader, &reference, true, Some(&at(LinkIssue, 0)), &mut |record, _, position| {
        let SelectedRecord::LinkIssue { session, issue } = record else { panic!("issue coordinate resolved another family") };
        assert_eq!(session.to_string(), "test:child");
        assert!(matches!(*issue, polylogue_model::read::ParentIssue::MissingParent { parent, .. } if parent.to_string() == "test:missing"));
        observed.push(position);
        Ok(false)
    }).unwrap();
    assert_eq!(observed, [at(LinkIssue, 0)]);
}

#[test]
fn a_null_column_is_an_unknown_figure() {
    let figures = figures(&serde_json::json!({"cost": null, "count": 0}));
    assert_eq!(figures.len(), 2);
    assert_eq!(figures[0].name, "cost");
    assert_eq!(figures[0].value, ValueState::Unknown { reason: None });
    assert_eq!(figures[1].name, "count");
    assert_eq!(figures[1].value, ValueState::known(serde_json::json!(0)));
}
