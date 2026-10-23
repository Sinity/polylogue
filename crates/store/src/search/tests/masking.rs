use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn chosen_hit_batches_share_source_context_and_ignore_later_unselected_corruption() {
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
    assert!(!first.text.contains("needleprivate"));
    assert!(first.marks.is_empty());
    assert_eq!(first.session, "claude-code:snippets");
    assert_eq!(first.message, "claude-code:snippets/hit");
    assert!(
        snippets
            .iter()
            .all(|snippet| snippet.as_ref() == Some(first))
    );

    // The admitted reader retains its selected hit and masking prefix while a
    // fresh reader sees a different prefix in the same synthetic archive.
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
    assert!(
        fresh.snippets("needleprivate", &scopes, &terms).is_err(),
        "corruption in the required prefix must refuse, not silently drop a selected hit"
    );
    assert_eq!(
        reader.snippets("needleprivate", &scopes, &terms).unwrap(),
        snippets
    );
}
