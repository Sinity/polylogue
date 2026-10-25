use super::*;

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
