use super::*;
use crate::StoreError;
use polylogue_model::bundle::BundleError;
use polylogue_model::{
    Precision, SessionId, TimeEvidence, TimeEvidenceError, TimeQuality, Timestamp,
};

#[test]
fn completed_build_measurement_pays_its_wal_and_refuses_a_live_reader() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer =
        crate::archive::open_runtime_writer(&archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    writer
        .execute_batch("CREATE TABLE measured(value INTEGER); INSERT INTO measured VALUES(1)")
        .unwrap();
    let reader = Connection::open(archive.index_db()).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    writer.execute("UPDATE measured SET value=2", []).unwrap();
    assert!(checkpoint_first_build(&archive).is_err());
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    reader.execute_batch("ROLLBACK").unwrap();
    let databases = checkpoint_first_build(&archive).unwrap();
    let index = databases
        .iter()
        .find(|database| database.name == "index")
        .unwrap();
    assert!(index.wal_before > 0);
    assert!(databases.iter().all(|database| database.wal_after == 0));
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}

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

/// A role the model does not have is a read error, never a user message.
#[test]
fn unknown_stored_role_fails_the_read() {
    let directory = tempfile::tempdir().unwrap();
    let (fixture, session) = stored_message(directory.path());
    assert_eq!(
        fixture
            .reads()
            .unwrap()
            .reader()
            .unwrap()
            .messages(&session, false)
            .unwrap()[0]
            .role,
        polylogue_model::Role::Assistant
    );
    fixture
        .index
        .execute("UPDATE message SET role = 'narrator'", [])
        .unwrap();
    let reads = fixture.reads().unwrap();
    let error = reads
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap_err();
    assert!(error.to_string().contains("narrator"), "{error}");
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
        error.to_string().contains("unparseable attributes"),
        "{error}"
    );
}
