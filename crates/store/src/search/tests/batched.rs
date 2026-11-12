use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn batched_hits_preserve_query_priority_scope_eligibility_and_cyclic_ancestry() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let mut session = |id, kind| {
        fixture
            .session(SessionFacts {
                id,
                kind,
                ..Default::default()
            })
            .unwrap()
    };
    let root = session("test:snippet-root", "main");
    let leaf = session("test:snippet-leaf", "subagent");
    let empty = session("test:snippet-empty", "main");
    let mut message = |pk, id, text, written_by, on_active_path| {
        fixture
            .message(
                pk,
                MessageFacts {
                    id,
                    text: Some(text),
                    written_by,
                    on_active_path,
                    prompt_kind: (written_by == "harness").then_some("harness"),
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let cedar = message(root, "cedar", "cedar", "human", Some(true));
    let needle = message(leaf, "needle", "needle", "model", Some(true));
    let inactive = message(leaf, "inactive", "cedar", "human", Some(false));
    let harness = message(root, "harness", "cedar", "harness", Some(true));
    fixture
        .link("test:snippet-root", "test:snippet-leaf", "spawn", None)
        .unwrap();
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    write.execute("INSERT INTO link(parent_id,child_id,child_pk,kind,how,evidence,input_pk) VALUES('test:snippet-leaf','test:snippet-root',?1,'spawn',NULL,'fixture',(SELECT pk FROM input_ref LIMIT 1))",[root]).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let terms = [
        [polylogue_query::search::SnippetTerms::Word {
            text: "cedar".into(),
            index: SnippetIndex::Prose,
        }],
        [polylogue_query::search::SnippetTerms::Word {
            text: "needle".into(),
            index: SnippetIndex::Prose,
        }],
    ];
    let queries = [
        SnippetQuery {
            index: SnippetIndex::Prose,
            terms: &terms[0],
        },
        SnippetQuery {
            index: SnippetIndex::Prose,
            terms: &terms[1],
        },
    ];
    let scopes = [
        Within::Session(root),
        Within::Message(needle),
        Within::Session(empty),
        Within::Message(inactive),
        Within::Message(harness),
        Within::Session(root),
        Within::Message(cedar),
    ];
    let snippets = reader.snippets(&queries, &scopes).unwrap();
    let chosen = snippets
        .iter()
        .map(|hit| hit.as_ref().map(|hit| hit.message.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(
        chosen,
        [
            Some("cedar"),
            Some("needle"),
            None,
            Some("inactive"),
            None,
            Some("cedar"),
            Some("cedar")
        ]
    );
    assert_eq!(snippets[0], snippets[5]);
    assert!(reader.snippets(&queries, &[]).unwrap().is_empty());
    assert_eq!(
        reader.snippets(&[], &scopes).unwrap(),
        vec![None; scopes.len()]
    );
}

#[test]
fn a_large_scope_uses_index_hits_and_preserves_unicode_prefixes_and_nested_index_terms() {
    use polylogue_query::search::SnippetTerms;
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:large-snippet",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for i in 0..300 {
        fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("filler-{i}"),
                    role: "user",
                    written_by: "human",
                    text: Some("ordinary filler"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture
        .message(
            session,
            MessageFacts {
                id: "unicode-hit",
                role: "user",
                written_by: "human",
                text: Some("needle Łódźharbor"),
                ..Default::default()
            },
        )
        .unwrap();
    let terms = [SnippetTerms::All(vec![
        SnippetTerms::Word {
            text: "needle".into(),
            index: SnippetIndex::Prose,
        },
        SnippetTerms::Word {
            text: "lodz*".into(),
            index: SnippetIndex::Prose,
        },
        SnippetTerms::Any(vec![SnippetTerms::Word {
            text: "not-in-this-index".into(),
            index: SnippetIndex::Output,
        }]),
    ])];
    let queries = [SnippetQuery {
        index: SnippetIndex::Prose,
        terms: &terms,
    }];
    for (analyzed, deferred) in [(false, false), (true, false), (true, true)] {
        let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
        if analyzed {
            write.execute_batch("ANALYZE").unwrap();
        }
        if deferred {
            for index in crate::writer::deferred_index_names() {
                write
                    .execute_batch(&format!("DROP INDEX IF EXISTS {index}"))
                    .unwrap();
            }
        }
        drop(write);
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        let result = reader
            .snippets(
                &queries,
                &[Within::Session(session), Within::Session(session)],
            )
            .unwrap();
        assert_eq!(result[0], result[1]);
        let hit = result[0].as_ref().unwrap();
        assert_eq!(hit.message, "unicode-hit");
        assert_eq!(hit.text, "needle Łódźharbor");
        assert_eq!(
            hit.marks
                .iter()
                .map(|&(a, b)| hit.text.chars().skip(a).take(b - a).collect::<String>())
                .collect::<Vec<_>>(),
            ["needle", "Łódźharbor"]
        );
    }
}
