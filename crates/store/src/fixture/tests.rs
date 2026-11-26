use super::*;
use crate::StoreError;
use polylogue_model::bundle::BundleError;
use polylogue_model::{
    Precision, SessionId, TimeAccuracy, TimeAccuracyError, TimeQuality, Timestamp,
};

#[test]
fn time_accuracy_and_order_survive_fixture_storage() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    let session_id = SessionId::parse("claude-code:clock").unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:clock",
            ..Default::default()
        })
        .unwrap();
    let start = Timestamp::from_unix_ms(0).unwrap();
    let bounded = TimeAccuracy {
        quality: TimeQuality::Intrinsic,
        precision: Some(Precision::Bounded),
        until: Timestamp::from_unix_ms(10_000),
    };
    fixture.session_time(session, Some(start), bounded).unwrap();
    assert!(matches!(
        fixture.session_time(session, bounded.until, bounded),
        Err(StoreError::Bundle(BundleError::Time(
            TimeAccuracyError::InvalidBounds
        )))
    ));
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:clock-message",
                role: "assistant",
                written_by: "model",
                seq: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
    let at = Timestamp::from_unix_ms(5_000).unwrap();
    let accuracy = TimeAccuracy {
        until: Timestamp::from_unix_ms(9_000),
        ..bounded
    };
    fixture.message_time(message, Some(at), accuracy).unwrap();
    assert_eq!(fixture.message_order(message).unwrap(), 7);
    let cached: Option<i64> = fixture
        .index
        .query_row(
            "SELECT last_activity_at FROM session WHERE pk=?1",
            [session],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(cached, Some(5_000));
    {
        let reads = fixture.reads().unwrap();
        let reader = reads.reader().unwrap();
        let stored = reader.session(&session_id).unwrap().unwrap();
        assert_eq!(
            (stored.started, stored.time_accuracy()),
            (Some(start), bounded)
        );
        let messages = reader.messages(&session_id, false).unwrap();
        assert_eq!(
            (messages[0].at, messages[0].time_accuracy()),
            (Some(at), accuracy)
        );
    }
    let invalid = TimeAccuracy {
        until: Some(at),
        ..accuracy
    };
    assert!(matches!(
        fixture.message_time(message, Some(at), invalid),
        Err(StoreError::Bundle(BundleError::Time(
            TimeAccuracyError::InvalidBounds
        )))
    ));
    let reads = fixture.reads().unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .messages(&session_id, false)
            .unwrap()[0]
            .time_accuracy(),
        accuracy
    );
    drop(reads);
    fixture
        .message_time(message, None, TimeAccuracy::UNKNOWN)
        .unwrap();
    let reads = fixture.reads().unwrap();
    let messages = reads
        .reader()
        .unwrap()
        .messages(&session_id, false)
        .unwrap();
    assert_eq!(
        (messages[0].at, messages[0].time_accuracy()),
        (None, TimeAccuracy::UNKNOWN)
    );
    let cached: Option<i64> = fixture
        .index
        .query_row(
            "SELECT last_activity_at FROM session WHERE pk=?1",
            [session],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(cached, None);
}

/// One stored message for the corruption tests below.
fn stored_message(directory: &std::path::Path) -> (Fixture, SessionId) {
    let mut fixture = Fixture::new(&directory.join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:decode",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:decode-message",
                role: "assistant",
                written_by: "model",
                ..Default::default()
            },
        )
        .unwrap();
    (fixture, SessionId::parse("claude-code:decode").unwrap())
}

/// Attributes that do not parse are a read error, never empty attributes.
#[test]
fn unparseable_message_attrs_fail_the_read() {
    let directory = tempfile::tempdir().unwrap();
    let (fixture, session) = stored_message(directory.path());
    fixture
        .index
        .execute("UPDATE message SET attrs_json = '{not json'", [])
        .unwrap();
    let reads = fixture.reads().unwrap();
    let error = reads
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap_err();
    assert!(
        matches!(
            &error,
            StoreError::StoredValue { column: Some(0), reason }
                if reason == "message claude-code:decode-message has unparseable attributes"
        ),
        "{error:?}"
    );
}

/// An `attrs` entry that is not an object is a read error, never empty attributes.
#[test]
fn non_object_message_attrs_fail_the_read() {
    let directory = tempfile::tempdir().unwrap();
    let (fixture, session) = stored_message(directory.path());
    fixture
        .index
        .execute("UPDATE message SET attrs_json = '{\"attrs\":[1]}'", [])
        .unwrap();
    let reads = fixture.reads().unwrap();
    let error = reads
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap_err();
    assert!(
        matches!(
            &error,
            StoreError::StoredValue { column: Some(0), reason }
                if reason == "message claude-code:decode-message has unparseable attributes"
        ),
        "{error:?}"
    );
}
