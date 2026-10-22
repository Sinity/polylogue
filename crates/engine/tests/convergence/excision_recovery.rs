use super::*;

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
        let canonical = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
        canonical
            .execute("UPDATE excision SET refused_json=?1", [malformed])
            .unwrap();
        drop(canonical);
        delete_index(&w.root);
        let (mut engine, _lock) = w.engine(64 << 20);
        assert!(
            engine.converge().is_err(),
            "invalid removal metadata must not become an empty path list: {malformed}"
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 0);
    }
}

#[test]
fn an_archive_committed_excision_recovers_when_the_index_commit_was_lost() {
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
        "obsolete derived indexes must not retain removed messages"
    );
}

#[test]
fn older_or_conflicting_excision_authority_is_not_rebuilt_away() {
    for (conflicting, rebuild) in [
        (0, "none"),
        (1, "none"),
        (2, "none"),
        (2, "schema"),
        (2, "unsynced"),
    ] {
        let w = ClaudeHome::new();
        w.append(
            "victim.jsonl",
            &user("v1", "victim", "synthetic removed message"),
        );
        let before = w.dir.path().join("archive-before-excision.db");
        {
            let (mut e, lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            let a = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
            a.execute("VACUUM INTO ?1", [before.to_str().unwrap()])
                .unwrap();
            drop(a);
            e.excise(ExciseRequest {
                id: "removed".into(),
                scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                reason: None,
                dry_run: false,
            })
            .unwrap();
            drop(e);
            drop(lock);
        }
        let index = rusqlite::Connection::open(Archive::at(&w.root).index_db()).unwrap();
        let applied: String = index
            .query_row(
                "SELECT value FROM meta WHERE key='excision_version'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        drop(index);
        for name in ["archive.db", "archive.db-wal", "archive.db-shm"] {
            if let Err(error) = fs::remove_file(w.root.join(name)) {
                assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
            }
        }
        fs::rename(before, Archive::at(&w.root).archive_db()).unwrap();
        if conflicting != 0 {
            // An independent history can have the same or a greater counter
            // while lacking the removal already applied by this index.
            let a = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
            a.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('different','session:claude-code:other',1,9)",[]).unwrap();
            a.execute(
                "UPDATE meta SET value=?1 WHERE key='excision_version'",
                [(applied.parse::<u64>().unwrap() + u64::from(conflicting == 2)).to_string()],
            )
            .unwrap();
        }
        let index = rusqlite::Connection::open(Archive::at(&w.root).index_db()).unwrap();
        match rebuild {
            "schema" => index
                .pragma_update(
                    None,
                    "user_version",
                    polylogue_store::schema::INDEX_VERSION - 1,
                )
                .unwrap(),
            "unsynced" => {
                index.execute("INSERT INTO meta(key,value) VALUES ('unsynced','1') ON CONFLICT(key) DO UPDATE SET value='1'", []).unwrap();
            }
            _ => {}
        }
        drop(index);
        let embeddings = rusqlite::Connection::open(Archive::at(&w.root).embeddings_db()).unwrap();
        embeddings.execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES (?1,'unresolved-history',0,1,?2,1)", rusqlite::params![vec![7u8; 32], 1.0f32.to_le_bytes().as_slice()]).unwrap();
        drop(embeddings);
        let canonical = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
        canonical
            .execute(
                "INSERT INTO pending_vector_excision VALUES (?1,'unresolved-history')",
                [vec![7u8; 32]],
            )
            .unwrap();
        drop(canonical);
        let result = Archive::open_or_create(&w.root);
        assert!(
            matches!(
                result,
                Err(polylogue_store::StoreError::ExcisionConflict { .. })
            ),
            "fork={conflicting}, rebuild={rebuild}: {result:?}"
        );
        let embeddings = rusqlite::Connection::open(Archive::at(&w.root).embeddings_db()).unwrap();
        assert_eq!(
            embeddings
                .query_row(
                    "SELECT count(*) FROM embedded WHERE model='unresolved-history'",
                    [],
                    |r| r.get::<_, i64>(0)
                )
                .unwrap(),
            1,
            "authority conflict must refuse before any purchased-vector cleanup"
        );
        assert!(Archive::at(&w.root).index_db().exists());
        assert!(!fs::read_dir(&w.root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("index.db.excision-")
        }));
        let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
        assert!(matches!(
            pool.reader().unwrap().status(),
            Err(polylogue_store::StoreError::ExcisionPending)
        ));
    }
}

#[test]
fn an_embedding_build_cannot_write_across_an_excision_but_a_fresh_build_can() {
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
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.pending_vector_excision"),
        0
    );
    e.excise(request(false)).unwrap();
    assert!(matches!(
        held.put("synthetic", &[vector]),
        Err(polylogue_store::StoreError::ExcisionPending)
    ));
    assert!(matches!(
        held.fail("synthetic", &text.hash, "late failure"),
        Err(polylogue_store::StoreError::ExcisionPending)
    ));
    let fresh = pool.reader().unwrap();
    let mut vectors = fresh.vectors_writer().unwrap();
    assert!(vectors.done("synthetic").unwrap().is_empty());
    vectors
        .put(
            "synthetic",
            &[Vector {
                text_hash: vec![7; 32],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    vectors
        .fail("synthetic", &[8; 32], "fresh failure")
        .unwrap();
    assert_eq!(vectors.done("synthetic").unwrap().len(), 2);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.pending_vector_excision"),
        0
    );
}

#[test]
fn excision_removes_retired_index_payloads_but_preview_preserves_them() {
    let w = ClaudeHome::new();
    w.append(
        "victim.jsonl",
        &user("v1", "victim", "synthetic removed body"),
    );
    w.append("kept.jsonl", &user("k1", "kept", "keep this exact body"));
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    let kept = w.export("claude-code:kept");
    let old = w.root.join("index.db.v26");
    let current = rusqlite::Connection::open(Archive::at(&w.root).index_db()).unwrap();
    current
        .execute("VACUUM INTO ?1", [old.to_str().unwrap()])
        .unwrap();
    drop(current);
    let before = fs::read(&old).unwrap();
    let mut request = ExciseRequest {
        id: "retire-old-cache".into(),
        scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
        reason: None,
        dry_run: true,
    };
    e.excise(request.clone()).unwrap();
    assert_eq!(fs::read(&old).unwrap(), before);
    assert_eq!(messages_of(&w.root, "claude-code:victim").len(), 1);
    request.dry_run = false;
    e.excise(request).unwrap();
    assert!(!old.exists());
    assert!(!w.root.join("index.db.v26-wal").exists());
    assert!(!w.root.join("index.db.v26-shm").exists());
    assert!(messages_of(&w.root, "claude-code:victim").is_empty());
    assert_eq!(w.export("claude-code:kept"), kept);
    assert!(
        w.file("victim.jsonl").exists(),
        "provider files remain outside excision"
    );
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
