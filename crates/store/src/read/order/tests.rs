use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn message_session_lookup_never_hydrates_the_endpoint_body() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "codex:opaque:session",
            ..Default::default()
        })
        .unwrap();
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "codex:unrelated:message",
                role: "assistant",
                written_by: "model",
                text: Some("body"),
                ..Default::default()
            },
        )
        .unwrap();
    let connection =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    connection
        .execute("UPDATE block SET body=X'00' WHERE message_pk=?1", [message])
        .unwrap();
    drop(connection);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let id = MessageId::parse("codex:unrelated:message").unwrap();
    assert_eq!(
        reader.message_session(&id).unwrap(),
        Some(SessionId::parse("codex:opaque:session").unwrap())
    );
    assert!(reader.message_with_projection(&id).is_err());
    assert_eq!(
        reader
            .message_session(&MessageId::parse("codex:absent").unwrap())
            .unwrap(),
        None
    );
}

#[test]
fn message_ordinal_seek_is_metadata_only_and_uses_canonical_physical_order() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "codex:ordinal",
            ..Default::default()
        })
        .unwrap();
    for id in ["codex:z", "codex:a", "codex:m"] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    text: Some("body"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let connection =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    connection
        .execute("UPDATE block SET body=X'00' WHERE session_pk=?1", [session])
        .unwrap();
    drop(connection);
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader
        .select_messages(&SessionId::parse("codex:ordinal").unwrap(), false)
        .unwrap();
    for (ordinal, id) in ["codex:z", "codex:a", "codex:m"].into_iter().enumerate() {
        let expected = selected
            .locate(Some(&MessageId::parse(id).unwrap()))
            .unwrap()
            .unwrap();
        let located = selected.locate_ordinal(ordinal as u64).unwrap().unwrap();
        assert_eq!(located, expected);
        assert_eq!(
            reader
                .message_order(&MessageId::parse(id).unwrap())
                .unwrap(),
            Some(ordinal as u64)
        );
    }
    assert_eq!(selected.locate_ordinal(3).unwrap(), None);
    assert_eq!(selected.locate_ordinal(u64::MAX).unwrap(), None);
    assert!(
        reader
            .messages(&SessionId::parse("codex:ordinal").unwrap(), false)
            .is_err()
    );
}
