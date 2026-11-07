use super::*;
use crate::StoreError;
use polylogue_model::bundle::BundleError;
use polylogue_model::{
    Precision, SessionId, TimeEvidence, TimeEvidenceError, TimeQuality, Timestamp,
};

#[test]
fn source_time_evidence_and_order_survive_native_fixture_storage() {
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
    let bounded = TimeEvidence {
        quality: TimeQuality::Intrinsic,
        precision: Some(Precision::Bounded),
        until: Timestamp::from_unix_ms(10_000),
    };
    fixture.session_time(session, Some(start), bounded).unwrap();
    assert!(matches!(
        fixture.session_time(session, bounded.until, bounded),
        Err(StoreError::Projection(BundleError::Time(
            TimeEvidenceError::InvalidBounds
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
    let evidence = TimeEvidence {
        until: Timestamp::from_unix_ms(9_000),
        ..bounded
    };
    fixture.message_time(message, Some(at), evidence).unwrap();
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
            (stored.started, stored.time_evidence()),
            (Some(start), bounded)
        );
        let messages = reader.messages(&session_id, false).unwrap();
        assert_eq!(
            (messages[0].at, messages[0].time_evidence()),
            (Some(at), evidence)
        );
    }
    let invalid = TimeEvidence {
        until: Some(at),
        ..evidence
    };
    assert!(matches!(
        fixture.message_time(message, Some(at), invalid),
        Err(StoreError::Projection(BundleError::Time(
            TimeEvidenceError::InvalidBounds
        )))
    ));
    let reads = fixture.reads().unwrap();
    assert_eq!(
        reads
            .reader()
            .unwrap()
            .messages(&session_id, false)
            .unwrap()[0]
            .time_evidence(),
        evidence
    );
    drop(reads);
    fixture
        .message_time(message, None, TimeEvidence::UNKNOWN)
        .unwrap();
    let reads = fixture.reads().unwrap();
    let messages = reads
        .reader()
        .unwrap()
        .messages(&session_id, false)
        .unwrap();
    assert_eq!(
        (messages[0].at, messages[0].time_evidence()),
        (None, TimeEvidence::UNKNOWN)
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
