use super::*;

/// A rebuild can spend minutes applying a single reading. Stopping may discard
/// that uncommitted reading, while its source and previous reading stay usable.
#[cfg(feature = "faults")]
#[test]
fn stopping_retained_byte_replay_rolls_back_the_index_and_restart_replays_it() {
    for mode in 0..3 {
        crate::fault::clear();
        let w = test_archive();
        w.store_reading(
            batch(
                "replay-stop",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message("old", "s", Role::User, "committed body", None),
                )],
                &["s"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
        let replacement = || {
            let mut b = batch(
                "replay-stop",
                Route::Live,
                0,
                (0..48)
                    .map(|n| {
                        row(
                            n,
                            text_message(
                                &format!("replacement-{n}"),
                                "s",
                                Role::User,
                                "replacement body",
                                None,
                            ),
                        )
                    })
                    .collect(),
                &["s"],
            );
            b.parse[0].reading_version = "synthetic=2".into();
            b.change = Change::Reparse { chain: 1 };
            b.observation = None;
            b
        };
        let b = replacement();
        let sender = w.writer.sender().unwrap();
        let store = Arc::new(w.archive.blobs().unwrap());
        let stop = w.writer.stop_handle();
        crate::fault::arm("write.replay_rows", "", crate::fault::Action::Hold, 0, 1);
        let (done, finished) = sync_channel(1);
        let worker = std::thread::spawn(move || {
            let stopped = match mode {
                0 => {
                    sender.send(b).unwrap();
                    {
                        let (reply, report) = sync_channel(1);
                        sender.0.send(Msg::Flush(reply, false)).unwrap();
                        !report.recv().unwrap().errors.is_empty()
                    }
                }
                1 => sender
                    .store_group(
                        vec![crate::staging::PreparedBatch {
                            batch: b,
                            files: Vec::new(),
                        }],
                        store,
                    )
                    .unwrap()
                    .into_iter()
                    .all(|result| matches!(result, Err(StoreError::WriterGone))),
                _ => {
                    let mut b = b;
                    let mut reading = ChainReading::new().unwrap();
                    reading.push(&mut b.parse[0]).unwrap();
                    matches!(
                        sender.replace_chain(b.input, 1, b.length, reading),
                        Err(StoreError::WriterGone)
                    )
                }
            };
            done.send(stopped).unwrap();
        });
        let started = std::time::Instant::now();
        while !crate::fault::held() {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "replay did not reach index transaction"
            );
            std::thread::yield_now();
        }
        stop.request();
        crate::fault::release();
        assert_eq!(
            finished.recv_timeout(std::time::Duration::from_secs(5)),
            Ok(true),
            "mode {mode}"
        );
        worker.join().unwrap();
        crate::fault::clear();
        let reads = ReadPool::open(&w.archive).unwrap();
        let r = reads.reader().unwrap();
        assert_eq!(
            r.messages(&sid("s"), false)
                .unwrap()
                .iter()
                .map(|m| m.text())
                .collect::<Vec<_>>(),
            ["committed body"]
        );
        assert_eq!(
            r.query("SELECT records,parsed_upto,status FROM parse_state", &[])
                .unwrap(),
            vec![serde_json::json!({"records":1,"parsed_upto":100,"status":"done"})]
        );
        assert_eq!(
            r.query("SELECT length FROM a.revision", &[]).unwrap(),
            vec![serde_json::json!({"length":100})]
        );
        drop(r);
        drop(reads);
        let b = replacement();
        let TestArchive {
            writer,
            archive,
            _lock,
            _dir,
        } = w;
        drop(writer);
        let writer = Writer::start(&archive).unwrap();
        writer.send(b).unwrap();
        let report = writer.flush().unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
        let reads = ReadPool::open(&archive).unwrap();
        assert_eq!(
            reads
                .reader()
                .unwrap()
                .messages(&sid("s"), false)
                .unwrap()
                .len(),
            48
        );
    }
}

#[test]
fn stopping_still_commits_an_explicit_parse_failure() {
    let w = test_archive();
    let mut b = w
        .commit_source(
            batch("diagnosed", Route::Live, 0, vec![], &[]),
            &SOURCE_RECORD,
        )
        .unwrap();
    let mut step = ParseStep::new(Default::default(), Default::default(), 0..0, false);
    step.reset = false;
    step.reading_version = "synthetic=1".into();
    step.progress = polylogue_model::batch::StepProgress::Refused {
        reason: "declared source is invalid".into(),
    };
    b.parse = vec![step];
    w.writer.stop_handle().request();
    w.writer.send(b).unwrap();
    assert!(w.writer.flush_rows().unwrap().errors.is_empty());
    let reads = ReadPool::open(&w.archive).unwrap();
    let r = reads.reader().unwrap();
    assert_eq!(
        r.query("SELECT status,reason,parsed_upto FROM parse_state", &[])
            .unwrap(),
        vec![
            serde_json::json!({"status":"refused","reason":"declared source is invalid","parsed_upto":0})
        ]
    );
}
