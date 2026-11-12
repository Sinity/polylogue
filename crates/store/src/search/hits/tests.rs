use super::*;
use crate::fixture::{Fixture, ManySessions, MessageFacts, SessionFacts};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn scope_drives(reader: &Reader<'_>, fts: &str, wanted: &str) -> bool {
    let scope = scope_ctes(SnippetIndex::Prose, false);
    reader
        .search_scope_drives(
            &format!("WITH RECURSIVE {scope} SELECT DISTINCT block FROM eligible"),
            &[
                crate::SqlValue::Text(fts.into()),
                crate::SqlValue::Text(wanted.into()),
            ],
            SnippetIndex::Prose,
            fts,
        )
        .unwrap()
}

#[test]
fn scoped_snippets_read_the_eligible_blocks_once() {
    let mut work = Vec::new();
    for count in [64, 256] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        let session = fixture
            .session(SessionFacts {
                id: "test:scoped-bodies",
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        for i in 0..count {
            fixture
                .message(
                    session,
                    MessageFacts {
                        id: &format!("body-{i:04}"),
                        role: "user",
                        written_by: "human",
                        at: Some(i),
                        text: Some(&format!("needle body {i:04}")),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let outside = fixture
            .session(SessionFacts {
                id: "test:outside-scope",
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        for i in 0..257 {
            fixture
                .message(
                    outside,
                    MessageFacts {
                        id: &format!("outside-{i}"),
                        role: "user",
                        written_by: "human",
                        text: Some("needle outside filler"),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let terms = [polylogue_query::search::SnippetTerms::Word {
            text: "needle".into(),
            index: SnippetIndex::Prose,
        }];
        let mut measured = Vec::new();
        for analyzed in [false, true] {
            if analyzed {
                crate::archive::open_write(&fixture.archive.index_db(), 1024)
                    .unwrap()
                    .execute_batch("ANALYZE")
                    .unwrap();
            }
            let pool = fixture.reads().unwrap();
            let reader = pool.reader().unwrap();
            let conn = reader.conn().unwrap();
            assert!(scope_drives(
                &reader,
                "needle",
                &serde_json::to_string(&[(0, session, true)]).unwrap()
            ));
            let steps = Arc::new(AtomicU64::new(0));
            let counter = Arc::clone(&steps);
            conn.progress_handler(
                1,
                Some(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    false
                }),
            )
            .unwrap();
            let snippets = reader
                .snippets(
                    &[SnippetQuery {
                        index: SnippetIndex::Prose,
                        terms: &terms,
                    }],
                    &[Within::Session(session)],
                )
                .unwrap();
            conn.progress_handler(0, None::<fn() -> bool>).unwrap();
            measured.push(steps.load(Ordering::Relaxed));
            let snippet = snippets[0].as_ref().unwrap();
            assert_eq!(snippet.message, format!("body-{:04}", count - 1));
            assert_eq!(snippet.text, format!("needle body {:04}", count - 1));
        }
        work.push(measured);
    }
    for analyzed in 0..2 {
        assert!(
            work[1][analyzed] < work[0][analyzed] * 7,
            "four times the scoped blocks should not repeat the scope for each matching text: {work:?}"
        );
    }
}

#[test]
fn shared_text_reference_fanout_and_actually_rare_text_choose_different_plans() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let root = fixture
        .session(SessionFacts {
            id: "test:fanout-scope",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for i in 0..301 {
        fixture
            .message(
                root,
                MessageFacts {
                    id: &format!("own-{i}"),
                    role: "user",
                    written_by: "human",
                    text: Some("needle shared"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture
        .message(
            root,
            MessageFacts {
                id: "rare",
                role: "user",
                written_by: "human",
                text: Some("actuallyrare body"),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .many_sessions(ManySessions {
            prefix: "/synthetic/fanout",
            sessions: 20_000,
            messages: 1,
            answered: 0,
            last: 1000,
        })
        .unwrap();
    let write = crate::archive::open_write(&fixture.archive.index_db(), 1024).unwrap();
    let body = crate::encoding::pack_block(&polylogue_model::Block::Text {
        text: String::new(),
    })
    .unwrap();
    let text: i64 = write
        .query_row(
            "SELECT text_id FROM block b JOIN message m ON m.pk=b.message_pk WHERE m.id='own-0'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let insert = "INSERT INTO block(message_pk,session_pk,idx,kind,text_id,body) SELECT m.pk,m.session_pk,0,'text',?1,?2 FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id LIKE 'test:many-%' AND NOT EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk) ORDER BY m.pk LIMIT ?3";
    write.execute(insert, params![text, body, 1000]).unwrap();
    let wanted = serde_json::to_string(&[(0, root, true)]).unwrap();
    {
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        assert!(!scope_drives(&reader, "needle", &wanted));
    }
    write.execute(insert, params![text, body, 19_000]).unwrap();
    for analyzed in [false, true] {
        if analyzed {
            write.execute_batch("ANALYZE").unwrap();
        }
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        let conn = reader.conn().unwrap();
        assert_eq!(
            conn.query_row(
                "SELECT doc FROM fts_vocab_text WHERE term='needle'",
                [],
                |r| r.get::<_, u64>(0)
            )
            .unwrap(),
            1
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM block WHERE text_id=?1", [text], |r| r
                .get::<_, u64>(0))
                .unwrap(),
            20_301
        );
        assert!(scope_drives(&reader, "needle", &wanted));
        assert!(!scope_drives(&reader, "actuallyrare", &wanted));
        let terms = [polylogue_query::search::SnippetTerms::Word {
            text: "needle".into(),
            index: SnippetIndex::Prose,
        }];
        let snippets = reader
            .snippets(
                &[SnippetQuery {
                    index: SnippetIndex::Prose,
                    terms: &terms,
                }],
                &[Within::Session(root)],
            )
            .unwrap();
        assert_eq!(snippets[0].as_ref().unwrap().text, "needle shared");
        assert_eq!(snippets[0].as_ref().unwrap().message, "own-0");
    }
}
