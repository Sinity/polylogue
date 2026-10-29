use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn chosen_hit_batches_preserve_source_text_and_ignore_unselected_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:snippets",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let header = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:snippets/header",
                role: "assistant",
                written_by: "harness",
                prompt_kind: Some("harness"),
                text: Some("-----BEGIN PRIVATE KEY-----private prefix"),
                ..Default::default()
            },
        )
        .unwrap();
    let message = fixture.message(session, MessageFacts { id:"claude-code:snippets/hit", role:"user", written_by:"human", prompt_kind:Some("prose"), text:Some("one two three four five six seven eight nine ten eleven twelve thirteen fourteen needleprivate -----END PRIVATE KEY----- visible"), ..Default::default() }).unwrap();
    let late = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:snippets/late",
                role: "assistant",
                written_by: "model",
                text: Some("later unrelated body"),
                ..Default::default()
            },
        )
        .unwrap();
    let unrelated = fixture
        .session(SessionFacts {
            id: "claude-code:unrelated",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let unrelated = fixture
        .message(
            unrelated,
            MessageFacts {
                id: "claude-code:unrelated/body",
                role: "user",
                written_by: "human",
                text: Some("unselected unrelated body"),
                ..Default::default()
            },
        )
        .unwrap();
    let write =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    write
        .execute(
            "UPDATE block SET body=X'00' WHERE message_pk IN (?1,?2)",
            [late, unrelated],
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let scopes = [
        Within::Message(message),
        Within::Message(message),
        Within::Session(session),
    ];
    let terms = ["needleprivate".into()];
    let snippets = reader.snippets("needleprivate", &scopes, &terms).unwrap();
    assert_eq!(snippets.len(), 3);
    let first = snippets[0].as_ref().unwrap();
    assert!(first.text.contains("needleprivate"));
    assert!(!first.marks.is_empty());
    assert_eq!(first.session, "claude-code:snippets");
    assert_eq!(first.message, "claude-code:snippets/hit");
    assert!(
        snippets
            .iter()
            .all(|snippet| snippet.as_ref() == Some(first))
    );

    // Unselected earlier rows do not affect a snippet; the original reader
    // also retains its consistent selected source snapshot.
    let ordinary = crate::encoding::pack_text("ordinary header").unwrap();
    write.execute("UPDATE text SET body=?1 WHERE id=(SELECT text_id FROM block WHERE message_pk=?2 LIMIT 1)", params![ordinary, header]).unwrap();
    let fresh = pool.reader().unwrap();
    let changed = fresh.snippets("needleprivate", &scopes, &terms).unwrap();
    assert!(changed[0].as_ref().unwrap().text.contains("needleprivate"));
    drop(fresh);
    assert_eq!(
        reader.snippets("needleprivate", &scopes, &terms).unwrap(),
        snippets
    );
    write
        .execute("UPDATE block SET body=X'00' WHERE message_pk=?1", [header])
        .unwrap();
    let fresh = pool.reader().unwrap();
    assert_eq!(
        fresh.snippets("needleprivate", &scopes, &terms).unwrap(),
        snippets
    );
    drop(fresh);
    write
        .execute("UPDATE block SET body=X'00' WHERE message_pk=?1", [message])
        .unwrap();
    let fresh = pool.reader().unwrap();
    assert!(
        fresh.snippets("needleprivate", &scopes, &terms).is_err(),
        "corruption in the selected hit must refuse the read"
    );
    assert_eq!(
        reader.snippets("needleprivate", &scopes, &terms).unwrap(),
        snippets
    );
}

#[test]
fn credential_shaped_hits_keep_their_text_even_after_an_open_key_header() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:alone",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let text = |id: &'static str, text: &'static str| MessageFacts {
        id,
        role: "user",
        written_by: "human",
        prompt_kind: Some("prose"),
        text: Some(text),
        ..Default::default()
    };
    let first = fixture
        .message(
            session,
            text(
                "claude-code:alone/1",
                "wombat token AKIAIOSFODNN7EXAMPLE here",
            ),
        )
        .unwrap();
    fixture
        .message(
            session,
            text("claude-code:alone/2", "-----BEGIN PRIVATE KEY-----open"),
        )
        .unwrap();
    let after = fixture
        .message(
            session,
            text("claude-code:alone/3", "wombat keybody continues"),
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let terms = ["wombat".into()];
    let snippets = reader
        .snippets(
            "wombat",
            &[Within::Message(first), Within::Message(after)],
            &terms,
        )
        .unwrap();
    let first = snippets[0].as_ref().unwrap();
    assert_eq!(first.text, "wombat token AKIAIOSFODNN7EXAMPLE here");
    let after = snippets[1].as_ref().unwrap();
    assert_eq!(after.text, "wombat keybody continues");
}
