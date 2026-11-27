use super::*;

/// Status's message total, the `messages` listing and its count agree when a
/// session is hidden: each is of the visible sessions.
#[test]
fn status_and_the_messages_unit_agree_on_a_hidden_session() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (name, messages) in [("one", 3), ("two", 2)] {
        let id = format!("claude-code:{name}");
        let pk = f
            .session(SessionFacts {
                id: &id,
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        for i in 0..messages {
            f.message(
                pk,
                MessageFacts {
                    id: &format!("claude-code:{name}-{i}"),
                    role: "user",
                    written_by: "human",
                    text: Some("text"),
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    // What the counts summary writes for each session.
    rusqlite::Connection::open(f.archive.index_db())
        .unwrap()
        .execute_batch(
            "UPDATE session SET message_count=(SELECT count(*) FROM message m WHERE m.session_pk=session.pk)",
        )
        .unwrap();
    let figures = |f: &Fixture| {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        let mut options = options();
        options.fields = vec!["id".into()];
        options.page_size = Some(20);
        let program = crate::query::compile("messages", &options).unwrap();
        (
            reader.status(&[]).unwrap().messages,
            reader.query_page(&program, None).unwrap().rows.len() as u64,
            reader.query_count(&program).unwrap().map(|counts| counts.0),
        )
    };
    assert_eq!(figures(&f), (5, 5, Some(5)));
    f.note("hidden", None, None, "session:claude-code:two")
        .unwrap();
    assert_eq!(figures(&f), (3, 3, Some(3)));
}

/// Status's two session totals, the `sessions` listings with no filter and
/// their counts agree: logical sessions as the default lists them, every
/// session as `sessions where physical:true` does; each with a transcript and not
/// hidden (docs/operations.md, figures).
#[test]
fn status_and_the_sessions_unit_agree() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (id, kind) in [
        ("test:one", "main"),
        ("test:two", "main"),
        ("test:helper", "subagent"),
        ("test:named", "main"),
    ] {
        f.session(SessionFacts {
            id,
            kind,
            last: Some(1000),
            ..Default::default()
        })
        .unwrap();
    }
    // A subagent folds into the session that spawned it; a session a hook
    // named but no transcript holds is not listed.
    f.link("test:one", "test:helper", "spawn", None).unwrap();
    rusqlite::Connection::open(f.archive.index_db())
        .unwrap()
        .execute(
            "UPDATE session SET has_transcript = 0 WHERE id = 'test:named'",
            [],
        )
        .unwrap();
    f.derive().unwrap();
    let figures = |f: &Fixture| {
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        let mut options = options();
        options.fields = vec!["id".into()];
        options.page_size = Some(20);
        let status = reader.status(&[]).unwrap();
        let listed = |text: &str| {
            let program = crate::query::compile(text, &options).unwrap();
            (
                reader.query_page(&program, None).unwrap().rows.len() as u64,
                reader.query_count(&program).unwrap().map(|counts| counts.0),
            )
        };
        (
            (status.sessions, listed("sessions")),
            (
                status.sessions_physical,
                listed("sessions where physical:true"),
            ),
        )
    };
    assert_eq!(figures(&f), ((2, (2, Some(2))), (3, (3, Some(3)))));
    f.note("hidden", None, None, "session:test:two").unwrap();
    assert_eq!(figures(&f), ((1, (1, Some(1))), (2, (2, Some(2)))));
}
