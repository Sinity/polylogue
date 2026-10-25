use super::*;

#[test]
fn unreadable_delivered_brief_refuses_session_excision_before_commit() {
    let w = ClaudeHome::new();
    let brief = "synthetic private message in a delivered brief";
    w.append("victim.jsonl", &user("v1", "victim", brief));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let before = w.export("claude-code:victim");
    let archive_conn = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
    archive_conn.execute("INSERT INTO delivery(id,recipient_kind,sender,event,at,content_hash,bytes,body,refs_json,commit_seq)
        VALUES('bad-brief','operator','cli','recall',1,?1,?2,x'010203','[\"claude-code:victim\"]',1)",
        rusqlite::params![&[0u8;32][..],brief.len() as i64]).unwrap();
    for dry_run in [true, false] {
        let error = engine
            .excise(ExciseRequest {
                id: "bad-delivery".into(),
                scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                reason: None,
                dry_run,
            })
            .unwrap_err();
        assert!(
            matches!(
                error,
                polylogue_engine::EngineError::Store(polylogue_store::StoreError::Io(_))
            ),
            "{error:?}"
        );
        assert_eq!(w.export("claude-code:victim"), before);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 1);
    }
    let index = rusqlite::Connection::open(Archive::at(&w.root).index_db()).unwrap();
    let (body, hash): (Vec<u8>, Vec<u8>) = index
        .query_row("SELECT body,hash FROM text LIMIT 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    archive_conn
        .execute(
            "UPDATE delivery SET body=?1,content_hash=?2",
            rusqlite::params![body, hash],
        )
        .unwrap();
    let done = engine
        .excise(ExciseRequest {
            id: "bad-delivery".into(),
            scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    assert_eq!(done.deliveries_redacted, 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 0);
}

#[test]
fn malformed_excision_locations_refuse_collection_after_index_rebuild() {
    for malformed in [
        "not json",
        r#"{"locations":null}"#,
        r#"{"locations":[{"machine":"test","path":7}]}"#,
    ] {
        let w = ClaudeHome::new();
        w.append(
            "victim.jsonl",
            &user("v1", "victim", "synthetic removed message"),
        );
        {
            let (mut engine, _lock) = w.engine(64 << 20);
            engine.converge().unwrap();
            engine
                .excise(ExciseRequest {
                    id: "removed".into(),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .unwrap();
        }
        let archive_conn = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
        archive_conn
            .execute("UPDATE excision SET refused_json=?1", [malformed])
            .unwrap();
        drop(archive_conn);
        delete_index(&w.root);
        let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
        assert!(
            Engine::new(w.config(64 << 20), archive)
                .and_then(|mut engine| engine.converge())
                .is_err(),
            "invalid removal metadata must not become an empty path list: {malformed}"
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 0);
    }
}

#[test]
fn an_index_that_missed_a_committed_excision_is_rebuilt() {
    let w = ClaudeHome::new();
    w.append(
        "victim.jsonl",
        &user("v1", "victim", "private synthetic message"),
    );
    w.append("kept.jsonl", &user("k1", "kept", "keep this exact message"));
    let old_index = w.dir.path().join("index-before-excision.db");
    let expected;
    {
        let (mut e, lock) = w.engine(64 << 20);
        assert!(e.converge().unwrap().errors.is_empty());
        expected = w.export("claude-code:kept");
        // A consistent SQLite image of the index before the first commit.
        let index = rusqlite::Connection::open(Archive::at(&w.root).index_db()).unwrap();
        index
            .execute("VACUUM INTO ?1", [old_index.to_str().unwrap()])
            .unwrap();
        drop(index);
        e.excise(ExciseRequest {
            id: "interrupted-excision".into(),
            scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
            reason: None,
            dry_run: false,
        })
        .unwrap();
        assert!(messages_of(&w.root, "claude-code:victim").is_empty());
        drop(e);
        drop(lock);
    }
    // Reproduce the reachable commit boundary: new archive, old index. Both
    // images come from the real operation; the original provider file remains.
    delete_index(&w.root);
    fs::rename(old_index, Archive::at(&w.root).index_db()).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(
        messages_of(&w.root, "claude-code:victim").is_empty(),
        "restart must not serve index rows removed by an already committed excision"
    );
    assert_eq!(w.export("claude-code:kept"), expected);
    assert!(
        !fs::read_dir(&w.root).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("index.db.excision-")),
        "obsolete derived indexes must not keep removed messages"
    );
}

#[test]
fn an_embedding_build_stores_nothing_for_a_text_excised_meanwhile() {
    use polylogue_store::vectors::Vector;
    let w = ClaudeHome::new();
    w.append(
        "victim.jsonl",
        &user("v1", "victim", "removed synthetic text"),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let text = reader.prose_texts(0, 10, None).unwrap().remove(0);
    let mut held = reader.vectors_writer().unwrap();
    let vector = Vector {
        text_hash: text.hash.clone(),
        chunk: 0,
        values: vec![1.0],
    };
    held.put("synthetic", std::slice::from_ref(&vector))
        .unwrap();
    let request = |dry_run| ExciseRequest {
        id: "remove-vectors".into(),
        scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
        reason: None,
        dry_run,
    };
    e.excise(request(true)).unwrap();
    held.put("synthetic", std::slice::from_ref(&vector))
        .unwrap();
    e.excise(request(false)).unwrap();
    // The text is gone from the index, so the late build stores nothing.
    held.put("synthetic", &[vector]).unwrap();
    held.fail("synthetic", &text.hash, "late failure").unwrap();
    drop(held);
    let fresh = pool.reader().unwrap();
    let vectors = fresh.vectors().unwrap();
    assert!(vectors.done("synthetic").unwrap().is_empty());
}

#[test]
fn unreadable_stored_text_refuses_excision_and_a_repaired_fixture_can_retry() {
    let w = ClaudeHome::new();
    w.append(
        "victim.jsonl",
        &user("v1", "victim", "synthetic removed body"),
    );
    w.append("kept.jsonl", &user("k1", "kept", "keep this exact body"));
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    let kept = w.export("claude-code:kept");
    let revisions = w.count("SELECT count(*) AS n FROM a.revision");
    let db = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
    let (text_id, body): (i64, Vec<u8>) = db.query_row(
        "SELECT t.id,t.body FROM text t JOIN block b ON b.text_id=t.id JOIN session s ON s.pk=b.session_pk WHERE s.id='claude-code:victim'",
        [], |r| Ok((r.get(0)?, r.get(1)?)),
    ).unwrap();
    db.execute("UPDATE text SET body=x'626164' WHERE id=?1", [text_id])
        .unwrap();
    let request = ExciseRequest {
        id: "unreadable-text".into(),
        scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
        reason: None,
        dry_run: false,
    };
    let error = e.excise(request.clone()).unwrap_err();
    assert!(matches!(
        error,
        polylogue_engine::EngineError::Store(polylogue_store::StoreError::Io(_))
    ));
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), revisions);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id='claude-code:victim'"),
        1
    );
    assert_eq!(w.export("claude-code:kept"), kept);
    assert_eq!(
        db.query_row("SELECT body FROM text WHERE id=?1", [text_id], |r| r
            .get::<_, Vec<u8>>(0))
            .unwrap(),
        b"bad"
    );
    db.execute(
        "UPDATE text SET body=?1 WHERE id=?2",
        rusqlite::params![body, text_id],
    )
    .unwrap();
    e.excise(request).unwrap();
    assert!(messages_of(&w.root, "claude-code:victim").is_empty());
    assert_eq!(w.export("claude-code:kept"), kept);
}

#[test]
fn session_excision_reuses_one_writer_after_preview_refusal_and_success() {
    let w = ClaudeHome::new();
    for id in ["first", "second", "kept"] {
        w.append(&format!("{id}.jsonl"), &user(&format!("m-{id}"), id, id));
    }
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let expected = w.export("claude-code:kept");
    let mut request = ExciseRequest {
        id: "first-preview".into(),
        scope: ExciseScope::Session(SessionId::parse("claude-code:first").unwrap()),
        reason: None,
        dry_run: true,
    };
    assert_eq!(
        engine.excise(request.clone()).unwrap().sessions,
        ["claude-code:first"]
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
    request.scope = ExciseScope::Session(SessionId::parse("claude-code:missing").unwrap());
    assert!(engine.excise(request.clone()).is_err());
    for id in ["first", "second"] {
        request.id = format!("remove-{id}");
        request.scope =
            ExciseScope::Session(SessionId::parse(&format!("claude-code:{id}")).unwrap());
        request.dry_run = false;
        assert_eq!(
            engine.excise(request.clone()).unwrap().sessions,
            [format!("claude-code:{id}")]
        );
    }
    assert_eq!(w.export("claude-code:kept"), expected);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 2);
}
