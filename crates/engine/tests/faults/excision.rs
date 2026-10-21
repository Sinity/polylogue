use super::*;
use polylogue_engine::{ExciseRequest, ExciseScope};
use polylogue_model::SessionId;
use polylogue_store::StoreError;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn a_partially_committed_excision_stops_writes_until_restart() {
    for point in ["excision.between_commits", "excision.after_index_commit"] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append(
            "victim.jsonl",
            &claude_user("v1", "victim", "synthetic removed message"),
        );
        w.append(
            "kept.jsonl",
            &claude_user("k1", "kept", "untouched exact message"),
        );
        let expected;
        {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            expected = w.export("claude-code:kept");
            fault::arm(point, "interrupted", Action::Fail, 0, 1);
            let error = e
                .excise(ExciseRequest {
                    id: "interrupted".into(),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .unwrap_err();
            assert!(error.to_string().contains("injected fault"), "{error}");
            assert!(matches!(
                error,
                polylogue_engine::EngineError::Store(StoreError::ExcisionInterrupted(_))
            ));
            let continued = e
                .writer()
                .and_then(|sender| sender.derive().map_err(Into::into));
            assert!(
                matches!(
                    continued,
                    Err(polylogue_engine::EngineError::Store(StoreError::WriterGone))
                ),
                "{point}: a failed derived commit must not let later writes acknowledge incomplete removal"
            );
            let later = e.excise(ExciseRequest {
                id: "must-not-run".into(),
                scope: ExciseScope::Session(SessionId::parse("claude-code:kept").unwrap()),
                reason: None,
                dry_run: false,
            });
            assert!(later.is_err());
            let pool = w.reads();
            assert!(matches!(
                pool.reader().unwrap().status(),
                Err(StoreError::ExcisionPending)
            ));
        }
        fault::clear();
        let (mut e, _lock) = w.engine(64 << 20);
        assert!(e.converge().unwrap().errors.is_empty());
        assert_eq!(w.export("claude-code:kept"), expected);
        assert!(
            w.reads()
                .reader()
                .unwrap()
                .messages(&SessionId::parse("claude-code:victim").unwrap(), false)
                .unwrap()
                .is_empty()
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 1);
    }
}

#[test]
fn restart_finishes_vector_removal_without_discarding_other_purchased_vectors() {
    use polylogue_store::vectors::Vector;
    for point in [
        "excision.between_commits",
        "excision.after_index_commit",
        "excision.after_vector_commit",
    ] {
        fault::clear();
        let w = ClaudeHome::new();
        w.append(
            "victim.jsonl",
            &claude_user("v1", "victim", "synthetic removed text"),
        );
        w.append(
            "kept.jsonl",
            &claude_user("k1", "kept", "untouched purchased text"),
        );
        let key = "synthetic/test/1/document/unit";
        let removed;
        let kept;
        let unrelated = vec![9; 32];
        {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            {
                let pool = w.reads();
                let reader = pool.reader().unwrap();
                let texts = reader.prose_texts(0, 10, None).unwrap();
                removed = texts
                    .iter()
                    .find(|t| t.text == "synthetic removed text")
                    .unwrap()
                    .hash
                    .clone();
                kept = texts
                    .iter()
                    .find(|t| t.text == "untouched purchased text")
                    .unwrap()
                    .hash
                    .clone();
                let mut vectors = reader.vectors_writer().unwrap();
                let entries: Vec<_> = [&removed, &kept, &unrelated]
                    .into_iter()
                    .map(|hash| Vector {
                        text_hash: hash.clone(),
                        chunk: 0,
                        values: vec![1.0],
                    })
                    .collect();
                vectors.put(key, &entries).unwrap();
                vectors
                    .fail(key, &removed, "synthetic failure to remove")
                    .unwrap();
            }
            let held_pool = w.reads();
            let held_reader = held_pool.reader().unwrap();
            let mut held = held_reader.vectors_writer().unwrap();
            fault::arm(point, "vector-interrupted", Action::Fail, 0, 1);
            assert!(
                e.excise(ExciseRequest {
                    id: "vector-interrupted".into(),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .is_err()
            );
            assert!(matches!(
                held.put(
                    key,
                    &[Vector {
                        text_hash: removed.clone(),
                        chunk: 0,
                        values: vec![1.0]
                    }]
                ),
                Err(StoreError::ExcisionPending)
            ));
            assert!(matches!(
                held.fail(key, &removed, "late provider failure"),
                Err(StoreError::ExcisionPending)
            ));
            assert!(matches!(
                w.reads().reader().unwrap().status(),
                Err(StoreError::ExcisionPending)
            ));
        }
        fault::clear();
        fault::arm("excision.after_vector_cleanup", "", Action::Fail, 0, 1);
        assert!(polylogue_store::Archive::open_or_create(&w.root).is_err());
        fault::clear();
        let canonical = rusqlite::Connection::open(w.root.join("archive.db")).unwrap();
        assert_eq!(
            canonical
                .query_row("SELECT count(*) FROM pending_vector_excision", [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        drop(canonical);
        for _ in 0..2 {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            let pool = w.reads();
            let reader = pool.reader().unwrap();
            let vectors = reader.vectors().unwrap();
            let done = vectors.done(key).unwrap();
            assert!(
                !done.contains(&removed),
                "restart retained vectors/failures of the excised text"
            );
            assert_eq!(
                done,
                [kept.clone(), unrelated.clone()].into_iter().collect()
            );
            assert_eq!(vectors.status().unwrap().keys[0].failures, 0);
        }
    }
}
