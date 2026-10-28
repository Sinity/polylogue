use super::*;
use polylogue_engine::{ExciseRequest, ExciseTarget};
use polylogue_model::SessionId;
use polylogue_store::StoreError;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn a_stop_between_commits_stops_the_writer_and_reopening_finishes() {
    for point in [
        "excision.after_embeddings_commit",
        "excision.after_archive_commit",
        "excision.after_index_commit",
    ] {
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
            assert!(e.run_pass().unwrap().errors.is_empty());
            expected = w.export("claude-code:kept");
            fault::arm(point, "interrupted", Action::Fail, 0, 1);
            let error = e
                .excise(ExciseRequest {
                    id: "interrupted".into(),
                    target: ExciseTarget::Session(SessionId::parse("claude-code:victim").unwrap()),
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
                target: ExciseTarget::Session(SessionId::parse("claude-code:kept").unwrap()),
                reason: None,
                dry_run: false,
            });
            assert!(later.is_err());
        }
        fault::clear();
        let (mut e, _lock) = w.engine(64 << 20);
        assert!(e.run_pass().unwrap().errors.is_empty());
        assert_eq!(w.export("claude-code:kept"), expected);
        // Before the archive's commit nothing was excised; after it the
        // reopened index applies the recorded removal.
        let in_force = point != "excision.after_embeddings_commit";
        assert_eq!(
            w.reads()
                .reader()
                .unwrap()
                .messages(&SessionId::parse("claude-code:victim").unwrap(), false)
                .unwrap()
                .is_empty(),
            in_force,
            "{point}"
        );
        assert_eq!(
            w.count("SELECT count(*) AS n FROM a.excision"),
            i64::from(in_force)
        );
    }
}

#[test]
fn vectors_of_excised_texts_go_first_and_others_stay() {
    use polylogue_store::vectors::Vector;
    for point in [
        "excision.after_embeddings_commit",
        "excision.after_archive_commit",
        "excision.after_index_commit",
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
            assert!(e.run_pass().unwrap().errors.is_empty());
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
                // A vector for a text the index does not hold is not stored.
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
                    target: ExciseTarget::Session(SessionId::parse("claude-code:victim").unwrap()),
                    reason: None,
                    dry_run: false,
                })
                .is_err()
            );
            // A build that read the text before the excision cannot store
            // its vector afterwards.
            let late = held.put(
                key,
                &[Vector {
                    text_hash: removed.clone(),
                    chunk: 0,
                    values: vec![1.0],
                }],
            );
            let late_failure = held.fail(key, &removed, "late provider failure");
            if point == "excision.after_archive_commit" {
                assert!(late.is_err() && late_failure.is_err(), "{point}");
            }
        }
        fault::clear();
        for _ in 0..2 {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.run_pass().unwrap().errors.is_empty());
            let pool = w.reads();
            let reader = pool.reader().unwrap();
            let vectors = reader.vectors().unwrap();
            let done = vectors.done(key).unwrap();
            // A stop before the archive's commit excised nothing.
            let in_force = point != "excision.after_embeddings_commit";
            let mut expected: std::collections::HashSet<_> = [kept.clone()].into_iter().collect();
            if !in_force {
                expected.insert(removed.clone());
            }
            assert_eq!(done, expected, "{point}");
            assert_eq!(
                vectors.status().unwrap().keys[0].failures,
                u64::from(!in_force)
            );
        }
    }
}
