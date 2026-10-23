use super::*;
use crate::fixture::{MessageFacts, SessionFacts};
use std::ops::ControlFlow;

#[test]
fn tied_source_positions_use_ids_in_physical_logical_prompt_and_fingerprint_reads() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:ties",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for id in [
        "claude-code:z-tied",
        "claude-code:a-tied",
        "claude-code:zz-earlier",
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    text: Some(id),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let conn =
        crate::archive::open_write(&fixture.archive.index_db(), crate::archive::cache::SETUP)
            .unwrap();
    conn.execute(
        "UPDATE message SET seq=CASE WHEN id='claude-code:zz-earlier' THEN 0 ELSE 7 END",
        [],
    )
    .unwrap();
    let expected = [
        "claude-code:zz-earlier",
        "claude-code:a-tied",
        "claude-code:z-tied",
    ];
    let pool = fixture.reads().unwrap();
    let id = SessionId::parse("claude-code:ties").unwrap();
    let mut fingerprints = Vec::new();
    for changed_plan in [false, true] {
        if changed_plan {
            // Reverse tied native IDs in the actual read index. Row identities,
            // bodies and source positions remain unchanged.
            conn.execute_batch("DROP INDEX message_session; CREATE INDEX message_session ON message(session_pk,order_input_id,seq,id DESC)").unwrap();
        }
        let r = pool.reader().unwrap();
        for logical in [false, true] {
            let selected = r.select_messages(&id, logical).unwrap();
            let mut visited = Vec::new();
            selected
                .visit(|message| {
                    visited.push(message.id.to_string());
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(
                visited, expected,
                "logical={logical}, changed_plan={changed_plan}"
            );
            let mut key_ids = Vec::new();
            for pk in selected.keys().unwrap() {
                key_ids.push(
                    r.query(
                        "SELECT id FROM current_message WHERE pk=?1",
                        &[rusqlite::types::Value::Integer(pk)],
                    )
                    .unwrap()[0]["id"]
                        .as_str()
                        .unwrap()
                        .to_owned(),
                );
            }
            assert_eq!(key_ids, visited);
            if changed_plan {
                assert_eq!(
                    selected.fingerprint().unwrap(),
                    fingerprints[usize::from(logical)]
                );
            } else {
                fingerprints.push(selected.fingerprint().unwrap());
            }
        }
        assert_eq!(
            r.prompts("claude-code:ties")
                .unwrap()
                .iter()
                .map(|prompt| prompt.message_id.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        let mut blocks = Vec::new();
        crate::ordered_blocks::visit(r.conn().unwrap(), session, |block| {
            blocks.push(block.block.text().unwrap().to_owned());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
        assert_eq!(blocks, expected);
    }
}
