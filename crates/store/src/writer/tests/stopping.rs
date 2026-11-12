use super::*;

/// A rebuild can spend minutes applying a single reading. Stopping may discard
/// that uncommitted reading, while its source and previous reading stay usable.
#[cfg(feature = "faults")]
#[test]
fn stopping_retained_byte_replay_rolls_back_the_index_and_restart_replays_it() {
    for mode in [
        "queued",
        "staged",
        "replacement",
        "existing_asset",
        "new_asset",
    ] {
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
        assert!(w.writer.flush_rows().unwrap().errors.is_empty());
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
        let mut b = replacement();
        let mut files = Vec::new();
        let decoded = if matches!(mode, "existing_asset" | "new_asset") {
            let bytes: &[u8] = if mode == "existing_asset" {
                &SOURCE_RECORD
            } else {
                b"decoded inline asset"
            };
            let (blob, prepared) = w.archive.prepare_decoded(bytes).unwrap();
            b.blobs.push(blob.clone());
            files.extend(prepared);
            Some(blob)
        } else {
            None
        };
        let sender = w.writer.sender().unwrap();
        let store = Arc::new(w.archive.blobs().unwrap());
        let stop = w.writer.stop_handle();
        crate::fault::arm("write.replay_rows", "", crate::fault::Action::Hold, 0, 1);
        let (done, finished) = sync_channel(1);
        let worker = std::thread::spawn(move || {
            let stopped = match mode {
                "queued" => {
                    sender.send(b).unwrap();
                    {
                        let (reply, report) = sync_channel(1);
                        sender.0.send(Msg::Flush(reply, false)).unwrap();
                        {
                            let report = report.recv().unwrap();
                            report.batches == 0 && report.errors.is_empty()
                        }
                    }
                }
                "staged" | "existing_asset" | "new_asset" => sender
                    .store_group(
                        vec![crate::staging::PreparedBatch { batch: b, files }],
                        store,
                    )
                    .unwrap()
                    .into_iter()
                    .all(|result| matches!(result, Err(StoreError::WriteCancelled))),
                _ => {
                    let mut b = b;
                    let mut reading = ChainReading::new().unwrap();
                    reading.push(&mut b.parse[0]).unwrap();
                    matches!(
                        sender.replace_chain(b.input, 1, b.length, reading),
                        Err(StoreError::WriteCancelled)
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
        if let Some(blob) = decoded {
            assert_eq!(
                r.conn()
                    .unwrap()
                    .query_row(
                        "SELECT size FROM a.blob WHERE hash=?1",
                        [blob.hash.0.as_slice()],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                i64::try_from(blob.size).unwrap()
            );
        }
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

#[cfg(feature = "faults")]
#[test]
fn a_diagnosed_member_does_not_keep_other_replays_running_during_stop() {
    for staged in [false, true] {
        crate::fault::clear();
        let w = test_archive();
        w.store_reading(
            batch(
                "normal",
                Route::Live,
                0,
                vec![row(
                    0,
                    text_message("old", "s", Role::User, "before stop", None),
                )],
                &["s"],
            ),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
        let mut diagnosis = w
            .commit_source(
                batch("diagnosis", Route::Live, 0, Vec::new(), &[]),
                &SOURCE_RECORD,
            )
            .unwrap();
        let mut step = ParseStep::new(Default::default(), Default::default(), 0..0, false);
        step.reset = false;
        step.reading_version = "synthetic=1".into();
        step.progress = polylogue_model::batch::StepProgress::Refused {
            reason: "diagnosis survives stop".into(),
        };
        diagnosis.parse = vec![step];
        let mut normal = batch(
            "normal",
            Route::Live,
            0,
            (0..48)
                .map(|n| {
                    row(
                        n,
                        text_message(&format!("new-{n}"), "s", Role::User, "after stop", None),
                    )
                })
                .collect(),
            &["s"],
        );
        normal.change = Change::Reparse { chain: 1 };
        normal.observation = None;
        normal.parse[0].reading_version = "synthetic=2".into();
        assert!(w.writer.flush_rows().unwrap().errors.is_empty());
        // The queued route holds the preceding committed group's memory release
        // until both members and their flush are queued. They enter one group.
        let gate = if staged {
            None
        } else {
            let mut gate = w
                .commit_source(
                    batch("gate", Route::Live, 0, Vec::new(), &[]),
                    &SOURCE_RECORD,
                )
                .unwrap();
            let (entered, held) = sync_channel(1);
            let (release, resumed) = sync_channel(1);
            gate.release = InFlight::new(move || {
                let _ = entered.send(());
                let _ = resumed.recv();
            });
            w.writer.send(gate).unwrap();
            held.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            Some(release)
        };
        crate::fault::arm("write.replay_rows", "", crate::fault::Action::Hold, 0, 1);
        let sender = w.writer.sender().unwrap();
        let store = Arc::new(w.archive.blobs().unwrap());
        let (done, finished) = sync_channel(1);
        let worker = std::thread::spawn(move || {
            let clean = if staged {
                let results = sender
                    .store_group(
                        vec![
                            crate::staging::PreparedBatch {
                                batch: normal,
                                files: Vec::new(),
                            },
                            crate::staging::PreparedBatch {
                                batch: diagnosis,
                                files: Vec::new(),
                            },
                        ],
                        store,
                    )
                    .unwrap();
                matches!(results.first(), Some(Err(StoreError::WriteCancelled)))
                    && matches!(results.get(1), Some(Ok(_)))
            } else {
                sender.send(normal).unwrap();
                sender.send(diagnosis).unwrap();
                let (reply, report) = sync_channel(1);
                sender.0.send(Msg::Flush(reply, false)).unwrap();
                gate.unwrap().send(()).unwrap();
                report.recv().unwrap().errors.is_empty()
            };
            done.send(clean).unwrap();
        });
        let started = std::time::Instant::now();
        while !crate::fault::held() {
            assert!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "mixed replay group did not reach index write"
            );
            std::thread::yield_now();
        }
        w.writer.stop_handle().request();
        crate::fault::release();
        assert_eq!(
            finished.recv_timeout(std::time::Duration::from_secs(5)),
            Ok(true),
            "staged={staged}"
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
            ["before stop"]
        );
        assert_eq!(r.query("SELECT p.status,p.reason,p.parsed_upto FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id='diagnosis'", &[]).unwrap(), vec![serde_json::json!({"status":"refused","reason":"diagnosis survives stop","parsed_upto":0})]);
        assert_eq!(r.query("SELECT p.records,p.parsed_upto,p.reading_version FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id='normal'", &[]).unwrap(), vec![serde_json::json!({"records":1,"parsed_upto":100,"reading_version":"synthetic=1"})]);
    }
}
