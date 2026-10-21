use super::*;

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
        fs::read_dir(&w.root).unwrap().any(|entry| entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("index.db.excision-")),
        "keep the stale derived index as recovery evidence"
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
        let result = Archive::open_or_create(&w.root);
        assert!(
            matches!(
                result,
                Err(polylogue_store::StoreError::ExcisionConflict { .. })
            ),
            "fork={conflicting}, rebuild={rebuild}: {result:?}"
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
            pool.reader(),
            Err(polylogue_store::StoreError::ExcisionPending)
        ));
    }
}
