use super::*;
use polylogue_model::{Block, Role};
use polylogue_store::fixture::writer::{WriterArchive, message};

fn messages() -> (WriterArchive, Vec<MessageId>) {
    let mut archive = WriterArchive::new();
    let rows: Vec<_> = ["a", "b", "c", "d"]
        .into_iter()
        .map(|id| {
            message(
                id,
                "range",
                Role::Assistant,
                100,
                vec![Block::Text { text: id.into() }],
            )
        })
        .collect();
    let ids = rows.iter().map(|row| row.id.clone()).collect();
    archive.write("range", rows);
    (archive, ids)
}

#[test]
fn message_range_is_inclusive_and_keeps_physical_order_version_and_admission() {
    let (archive, ids) = messages();
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut seen = Vec::new();
    assert!(
        range(
            &reader,
            &ids[1],
            &ids[2],
            None,
            &mut |record, evidence, locator| {
                let Record::Message(row) = record else {
                    panic!("canonical message");
                };
                row.validate().unwrap();
                assert!(!evidence.inherited);
                assert_eq!(locator.kind, ReadRecordKind::Message);
                assert_eq!(locator.ordinal, row.order - 1);
                assert!(row.commit > 0);
                assert_eq!(
                    row.source_version,
                    reader
                        .message_with_projection(&row.message.id)
                        .unwrap()
                        .unwrap()
                        .1
                        .source_version
                );
                seen.push((row.message.id.clone(), row.order));
                Ok(true)
            }
        )
        .unwrap()
    );
    assert_eq!(seen, vec![(ids[1].clone(), 1), (ids[2].clone(), 2)]);
}

#[test]
fn message_range_callback_stops_before_the_next_selected_record() {
    let (archive, ids) = messages();
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut seen = Vec::new();
    assert!(
        !range(&reader, &ids[1], &ids[3], None, &mut |record, _, _| {
            seen.push(record.reference().unwrap().to_owned());
            Ok(false)
        })
        .unwrap()
    );
    assert_eq!(seen, vec![format!("message:{}", ids[1])]);
}

#[test]
fn message_range_rejects_reversed_cross_session_and_absent_endpoints() {
    let (mut archive, ids) = messages();
    let elsewhere = message("one", "other", Role::User, 100, vec![]);
    let elsewhere_id = elsewhere.id.clone();
    archive.write("other", vec![elsewhere]);
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    for (first, last, expected) in [
        (&ids[2], &ids[1], "bad_request"),
        (&ids[1], &elsewhere_id, "bad_request"),
        (
            &ids[1],
            &MessageId::parse("codex:absent").unwrap(),
            "not_found",
        ),
    ] {
        let mut visited = false;
        let failure = range(&reader, first, last, None, &mut |_, _, _| {
            visited = true;
            Ok(true)
        })
        .unwrap_err();
        let Response::Error { kind, .. } = failure else {
            panic!("typed range error");
        };
        assert_eq!(kind, expected);
        assert!(!visited);
    }
}

#[test]
fn message_range_resumes_from_relative_metadata_without_renumbering_source_order() {
    let (archive, ids) = messages();
    let pool = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let resume = ReadRecordLocator {
        kind: ReadRecordKind::Message,
        ordinal: 1,
    };
    let mut seen = Vec::new();
    assert!(
        range(
            &reader,
            &ids[1],
            &ids[2],
            Some(&resume),
            &mut |record, _, locator| {
                let Record::Message(row) = record else {
                    panic!("canonical message");
                };
                seen.push((row.message.id, row.order, locator.ordinal));
                Ok(true)
            }
        )
        .unwrap()
    );
    assert_eq!(seen, vec![(ids[2].clone(), 2, 1)]);
    for resume in [
        ReadRecordLocator {
            kind: ReadRecordKind::Message,
            ordinal: 2,
        },
        ReadRecordLocator {
            kind: ReadRecordKind::Document,
            ordinal: 0,
        },
    ] {
        let mut visited = false;
        let failure = range(&reader, &ids[1], &ids[2], Some(&resume), &mut |_, _, _| {
            visited = true;
            Ok(true)
        })
        .unwrap_err();
        let Response::Error { kind, .. } = failure else {
            panic!("typed cursor error");
        };
        assert_eq!(kind, "stale_cursor");
        assert!(!visited);
    }
}
