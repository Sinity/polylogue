use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use polylogue_store::fault::{self, Action};

#[test]
fn a_user_request_yields_between_complete_copies_and_resume_selects_only_the_remainder() {
    fault::clear();
    let home = ClaudeHome::new();
    let path = home.file("same.jsonl");
    let a = claude_user("one", "same", "first body");
    let b = claude_user("one", "same", "other body");
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
        for bytes in [&a, &b, &a, &b] {
            std::fs::write(&path, bytes).unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
        }
    }
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    engine.load().unwrap();
    let mut pass = Pass::default();
    engine
        .reparse_behind(RevisionChains::Newest, &mut pass)
        .unwrap();
    let (first, remaining, input) = {
        let reader = engine.reads.reader().unwrap();
        let revisions = reader
            .query(
                "SELECT input,chain FROM a.revision ORDER BY number DESC",
                &[],
            )
            .unwrap();
        assert_eq!(revisions.len(), 4);
        // The newest B is read. The older A is parsed before copying the
        // previous B and then the previous A, in descending chain order.
        (
            revisions[2]["chain"].as_i64().unwrap(),
            revisions[3]["chain"].as_i64().unwrap(),
            revisions[0]["input"].as_str().unwrap().to_owned(),
        )
    };
    let (send, receive) = std::sync::mpsc::channel();
    engine.requests = Some(Requests::new(receive));
    fault::arm("write.copy_table", "unused", Action::Hold, 0, 1);
    let enqueue = std::thread::spawn(move || {
        let started = Instant::now();
        while !fault::held() {
            assert!(started.elapsed() < Duration::from_secs(30));
            std::thread::yield_now();
        }
        let (reply, _wait) = std::sync::mpsc::sync_channel(1);
        let sent = send.send(LoopMsg::Import {
            paths: Vec::new(),
            format: None,
            from_archive: None,
            dry_run: false,
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            parent: None,
            reply,
        });
        fault::release();
        sent.unwrap();
    });
    let result = engine.reparse_behind(RevisionChains::Older, &mut pass);
    enqueue.join().unwrap();
    fault::clear();
    result.unwrap();
    assert_eq!(pass.readings_copied, 1);
    assert!(matches!(
        engine.requests.as_ref().unwrap().pending,
        Some(LoopMsg::Import { .. })
    ));
    let reader = engine.reads.reader().unwrap();
    assert_eq!(
        reader
            .query(
                "SELECT chain FROM parse_state WHERE chain IN (?1,?2) ORDER BY chain",
                &[
                    polylogue_store::SqlValue::Int(first),
                    polylogue_store::SqlValue::Int(remaining),
                ]
            )
            .unwrap(),
        vec![serde_json::json!({"chain":first})]
    );
    let behind = reader
        .chains_behind_index(
            &[],
            polylogue_store::StoppedChains::Finished,
            RevisionChains::Older,
            Some(&[input]),
        )
        .unwrap();
    assert_eq!(
        behind
            .iter()
            .map(|chain| chain.progress.chain)
            .collect::<Vec<_>>(),
        vec![remaining]
    );
    drop(reader);
    // Once the request has had its turn, the normal selection resumes
    // only the uncopied chain, without reading the source bytes again.
    engine.requests = None;
    let mut resumed = Pass::default();
    engine
        .reparse_behind(RevisionChains::Older, &mut resumed)
        .unwrap();
    assert_eq!(resumed.readings_copied, 1);
    assert_eq!(resumed.bytes_read, 0);
    let mut repeat = Pass::default();
    engine
        .reparse_behind(RevisionChains::Older, &mut repeat)
        .unwrap();
    assert_eq!((repeat.readings_copied, repeat.bytes_read), (0, 0));
}
