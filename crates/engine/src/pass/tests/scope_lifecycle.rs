use super::*;
use polylogue_model::Json;
use polylogue_sources::{Arrival, Capture, OriginSpec, Parse, Root, Step, StepParser};

fn source_rows(engine: &Engine, input: &str) -> Vec<Json> {
    engine.reads.reader().unwrap().query(
        "SELECT 'revision' AS row,number AS n,chain,length,produced AS detail FROM a.revision WHERE input=?1
         UNION ALL SELECT 'extent',offset,chain,length,hex(blob) FROM a.extent WHERE input=?1 ORDER BY row,n",
        &[polylogue_store::SqlValue::Text(input.to_owned())],
    ).unwrap()
}

#[test]
fn a_completed_scope_reset_reloads_confirmed_state_without_recapturing() {
    let home = ClaudeHome::new();
    let text = claude_user("one", "scope", "original confirmed body");
    home.append("scope.jsonl", &text);
    let path = home.file("scope.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let chain = engine.inputs[&path].chain.unwrap();
    assert!(engine.inputs[&path].parser_state.is_some());
    let captured = source_rows(&engine, &input);
    let checkpoint = engine
        .reads
        .reader()
        .unwrap()
        .parser_checkpoint(
            &input,
            chain,
            &engine.inputs[&path].origin.reading_version(),
        )
        .unwrap();
    let mut admitted_state = checkpoint
        .as_ref()
        .and_then(|stored| stored.checkpoint.as_ref())
        .and_then(|checkpoint| checkpoint.parser_state.as_ref())
        .expect("Claude continuation is admitted with its cursor")
        .clone();
    // The kept state leaves out the keyed context's changes, which the
    // index holds (docs/architecture.md, pipeline).
    admitted_state
        .as_object_mut()
        .unwrap()
        .remove(polylogue_model::parser_context::FIELD);
    assert_eq!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state)
    );
    // Fault only the process-local cache, leaving the admitted continuation
    // untouched. Exact equality after reset distinguishes restoration from reuse.
    engine
        .inputs
        .get_mut(&path)
        .unwrap()
        .parser_state
        .as_mut()
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert(
            "_scope_test_stale_cache".into(),
            Json::String("previous scope".into()),
        );
    assert_ne!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state)
    );
    engine.archive.advance_generation_for_test().unwrap();
    let (resumed, _) = engine.one_pass(false, &Default::default()).unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!(resumed.full_scans, 1, "the new scope repeats discovery");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(engine.inputs[&path].input_id, input);
    assert_eq!(engine.inputs[&path].chain, Some(chain));
    assert_eq!(engine.inputs[&path].parsed_upto, text.len() as u64);
    assert_eq!(
        engine.inputs[&path].parser_state.as_ref(),
        Some(&admitted_state),
        "the new scope restores admitted state instead of reusing the old cache"
    );
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .parser_checkpoint(
                &input,
                chain,
                &engine.inputs[&path].origin.reading_version()
            )
            .unwrap(),
        checkpoint
    );
}

#[test]
fn a_cached_scope_cannot_be_rebound_after_its_entry_check() {
    let home = ClaudeHome::new();
    home.append(
        "scope-race.jsonl",
        &claude_user("one", "scope-race", "admitted before the reset"),
    );
    let path = home.file("scope-race.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let captured = source_rows(&engine, &input);
    // This is the valid entry check, followed by a complete reset before
    // the shared load/turn token-acquisition seam runs.
    engine.read_scope.as_ref().unwrap().check().unwrap();
    engine.archive.advance_generation_for_test().unwrap();
    engine.archive.scope_token().unwrap().check().unwrap();
    assert!(matches!(
        engine.cached_scope(),
        Err(EngineError::Store(
            polylogue_store::StoreError::ScopeChanged
        ))
    ));
    assert!(matches!(
        engine.read_scope.as_ref().unwrap().check(),
        Err(polylogue_store::StoreError::ScopeChanged)
    ));
    let resumed = engine.run_pass().unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(engine.inputs[&path].input_id, input);
    engine.read_scope.as_ref().unwrap().check().unwrap();
}

struct Control {
    started: Sender<()>,
    release: Receiver<()>,
}
static CONTROL: Mutex<Option<Control>> = Mutex::new(None);
static READS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
struct Held(Box<dyn StepParser>, PathBuf);
impl StepParser for Held {
    fn step(
        &self,
        state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
        context: &polylogue_model::parser_context::Handle,
    ) -> Result<(Step, Json), polylogue_sources::ParseError> {
        READS.lock().unwrap().push(self.1.clone());
        let control = CONTROL.lock().unwrap().take();
        if let Some(control) = control {
            control.started.send(()).unwrap();
            control.release.recv().unwrap();
        }
        self.0.step(state, bytes, base, finished, context)
    }
}
fn held(path: &Path) -> Option<Parse> {
    let Parse::Lines(parser) = (polylogue_sources::origin::by_name("claude-code")
        .unwrap()
        .open)(path)?
    else {
        panic!("Claude line parser")
    };
    Some(Parse::Lines(Box::new(Held(parser, path.to_path_buf()))))
}
fn held_origin() -> &'static OriginSpec {
    let native = polylogue_sources::origin::by_name("claude-code").unwrap();
    Box::leak(Box::new(OriginSpec {
        name: native.name,
        harness_prefixes: native.harness_prefixes,
        modules: native.modules,
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: polylogue_model::Route::Live,
        input_identity: polylogue_sources::InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: native.recognize,
        open: held,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        web: None,
        assets: None,
        collector: None,
    }))
}

#[test]
fn a_scope_reset_during_preparation_cannot_complete_with_stale_state_or_capture_twice() {
    let home = ClaudeHome::new();
    let first = claude_user("one", "scope-session", "first confirmed body");
    home.append("scope-session.jsonl", &first);
    let path = home.file("scope-session.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.live_every = Duration::ZERO;
    assert!(engine.run_pass().unwrap().errors.is_empty());
    // Pause the actual Claude parser without changing its version or output.
    engine.inputs.get_mut(&path).unwrap().origin = held_origin();
    let input = engine.inputs[&path].input_id.clone();
    let chain = engine.inputs[&path].chain.unwrap();
    let second = claude_user("two", "scope-session", "prepared suffix");
    let expected = format!("{first}{second}");
    std::fs::write(&path, &expected).unwrap();
    let archive = engine.archive.clone();
    let (started, waiting) = channel();
    let (release, continue_reading) = channel();
    *CONTROL.lock().unwrap() = Some(Control {
        started,
        release: continue_reading,
    });
    let (mut engine, interrupted) = std::thread::scope(|threads| {
        let work = threads.spawn(move || {
            let (tx, rx) = channel();
            let status = Arc::new(Mutex::new(LoopStatus::default()));
            let result = engine.run(tx, rx, &status, Lifetime::UntilIdle, || {});
            (engine, result)
        });
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("actual parser entered after acquiring the suffix");
        archive.advance_generation_for_test().unwrap();
        release.send(()).unwrap();
        work.join().unwrap()
    });
    let summary = interrupted.unwrap();
    assert!(
        summary.passes >= 2,
        "the interrupted turn must not establish idle: {summary:?}"
    );
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    let captured = source_rows(&engine, &input);
    assert_eq!(
        captured
            .iter()
            .filter(|row| row["row"] == "revision")
            .count(),
        2
    );
    assert_eq!(
        captured.iter().filter(|row| row["row"] == "extent").count(),
        2
    );
    let (resumed, _) = engine.one_pass(false, &Default::default()).unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(engine.inputs[&path].input_id, input);
    assert_eq!(engine.inputs[&path].chain, Some(chain));
    assert_eq!(engine.inputs[&path].parsed_upto, expected.len() as u64);
    let reader = engine.reads.reader().unwrap();
    let stored = reader
        .stored_chain(&input, chain, expected.len() as u64)
        .unwrap();
    assert_eq!(
        stored
            .read_range(&engine.blobs, 0, expected.len() as u64)
            .unwrap(),
        expected.as_bytes()
    );
    assert_eq!(reader.query("SELECT count(*) AS n FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:scope-session'", &[]).unwrap()[0]["n"], 2);
}

#[test]
fn queued_import_yields_after_admitted_capture_or_replay_and_resumes_from_durable_positions() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    use std::sync::mpsc::sync_channel;

    for (retained, reads_sources) in [(false, true), (true, false), (true, true)] {
        for stop_after_import in [false, true] {
            let home = ClaudeHome::new();
            let first = claude_user("first", "request-turn", "first retained record");
            let suffix = (0..48)
                .map(|n| {
                    claude_user(
                        &format!("tail-{n}"),
                        "request-turn",
                        &format!("tail {n} {}", "x".repeat(512)),
                    )
                })
                .collect::<String>();
            let complete = format!("{first}{suffix}");
            home.append("request-turn.jsonl", &first);
            if retained {
                home.append("request-turn.jsonl", &suffix);
            }
            let path = home.file("request-turn.jsonl");
            let (mut engine, mut lock) = crate::import::tests::engine(&home, 256);
            assert!(engine.run_pass().unwrap().errors.is_empty());
            if retained {
                drop(engine);
                drop(lock);
                // Synthetic cold replay: every original byte stays in the
                // register and the native reader cannot reopen the source.
                polylogue_testkit::delete_index(&home.root);
                std::fs::remove_file(&path).unwrap();
                (engine, lock) = crate::import::tests::engine(&home, 256);
                engine.config.reads_sources = reads_sources;
                engine.load().unwrap();
            } else {
                home.append("request-turn.jsonl", &suffix);
            }
            let undiscovered = retained && reads_sources;
            if undiscovered {
                // This file predates run and needs no watcher event. The held
                // stored replay precedes planning the full scan's new files.
                home.append(
                    "unseen-source.jsonl",
                    &claude_user("unseen", "unseen-source", "preexisting new file"),
                );
            }
            engine.config.workers = 1;
            engine.config.live_every = Duration::from_millis(10);
            engine.inputs.get_mut(&path).unwrap().origin = held_origin();
            let input = engine.inputs[&path].input_id.clone();
            let captured_before = source_rows(&engine, &input);
            let import = home.dir.path().join("queued-conversation.json");
            std::fs::write(
                &import,
                serde_json::to_vec(&serde_json::json!({
                    "uuid":"queued-import", "chat_messages":[{
                        "uuid":"imported-message", "sender":"human", "text":"explicit request body"
                    }]
                }))
                .unwrap(),
            )
            .unwrap();
            let (entered, waiting) = channel();
            let (release, held) = channel();
            *CONTROL.lock().unwrap() = Some(Control {
                started: entered,
                release: held,
            });
            let stop = engine.stop_request();
            let (tx, rx) = channel();
            let worker_tx = tx.clone();
            let status = Arc::new(Mutex::new(LoopStatus::default()));
            let worker = std::thread::spawn(move || {
                let result = engine.run(worker_tx, rx, &status, Lifetime::UntilIdle, || {});
                (engine, result)
            });
            let entered = waiting.recv_timeout(Duration::from_secs(5));
            // Rendezvous keeps the engine in the real import response until
            // the test checks the committed background prefix.
            let (reply, answer) = sync_channel(0);
            let queued = tx.send(LoopMsg::Import {
                paths: vec![import],
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
            let _ = release.send(());
            let until = Instant::now() + Duration::from_secs(5);
            let mut imported = false;
            while Instant::now() < until {
                if home.count("SELECT count(*) AS n FROM session WHERE native_id='queued-import'")
                    == 1
                {
                    imported = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            let cursor = home.reads().reader().unwrap().query(
                "SELECT COALESCE(max(p.parsed_upto),0) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1",
                &[polylogue_store::SqlValue::Text(input.clone())],
            ).unwrap()[0]["n"].as_u64().unwrap();
            let captured = home
                .reads()
                .reader()
                .unwrap()
                .query(
                    "SELECT max(length) AS n FROM a.revision WHERE input=?1",
                    &[polylogue_store::SqlValue::Text(input.clone())],
                )
                .unwrap()[0]["n"]
                .as_u64()
                .unwrap();
            let response = answer.recv_timeout(Duration::from_secs(5));
            drop(answer);
            if stop_after_import || response.is_err() || !imported || entered.is_err() {
                stop.request();
                let _ = tx.send(LoopMsg::Stop);
            }
            let (mut engine, result) = worker.join().unwrap();
            assert!(
                entered.is_ok(),
                "the native parser did not reach its held chunk"
            );
            assert!(queued.is_ok());
            assert!(
                imported,
                "a request waited behind the rest of the stored history"
            );
            assert!(
                cursor < complete.len() as u64,
                "the import answered only after the whole input: {cursor}"
            );
            if retained {
                assert_eq!(captured, complete.len() as u64);
                assert_eq!(source_rows(&engine, &input), captured_before);
            } else {
                assert!(
                    captured > first.len() as u64,
                    "the admitted capture chunk must commit before yielding"
                );
                assert!(captured < complete.len() as u64);
            }
            let report = response.unwrap().unwrap();
            assert!(report.refused.is_empty(), "{report:?}");
            assert!(report.pass.errors.is_empty(), "{report:?}");
            let summary = result.unwrap();
            assert!(summary.work.errors.is_empty(), "{summary:?}");
            assert_eq!(summary.imports, 1);
            if stop_after_import {
                drop(engine);
                drop(lock);
                (engine, lock) = crate::import::tests::engine(&home, 256);
                engine.config.reads_sources = reads_sources;
                let resumed = engine.run_pass().unwrap();
                assert!(resumed.errors.is_empty(), "{resumed:?}");
            }
            assert_eq!(home.texts("request-turn").len(), 49);
            if undiscovered {
                assert_eq!(home.texts("unseen-source"), ["preexisting new file"]);
            }
            let reader = engine.reads.reader().unwrap();
            let progress = reader.input_progress(&input).unwrap().unwrap();
            assert_eq!(progress.parsed_upto, complete.len() as u64);
            let stored = reader
                .stored_chain(&input, progress.chain, complete.len() as u64)
                .unwrap();
            assert_eq!(
                stored
                    .read_range(&engine.blobs, 0, complete.len() as u64)
                    .unwrap(),
                complete.as_bytes()
            );
            drop(reader);
            drop(engine);
            drop(lock);
        }
    }
}

#[test]
fn a_stop_message_drains_the_admitted_capture_chunk_without_reading_the_whole_input() {
    let home = ClaudeHome::new();
    let first = claude_user("first", "stop-request", "initial body");
    home.append("stop-request.jsonl", &first);
    let path = home.file("stop-request.jsonl");
    let (mut engine, lock) = crate::import::tests::engine(&home, 256);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let suffix = (0..32)
        .map(|n| claude_user(&format!("tail-{n}"), "stop-request", &"x".repeat(512)))
        .collect::<String>();
    let complete = format!("{first}{suffix}");
    home.append("stop-request.jsonl", &suffix);
    engine.config.workers = 1;
    engine.config.live_every = Duration::from_millis(10);
    engine.inputs.get_mut(&path).unwrap().origin = held_origin();
    let input = engine.inputs[&path].input_id.clone();
    let stop = engine.stop_request();
    let (entered, waiting) = channel();
    let (release, held) = channel();
    *CONTROL.lock().unwrap() = Some(Control {
        started: entered,
        release: held,
    });
    let (tx, rx) = channel();
    let worker_tx = tx.clone();
    let worker = std::thread::spawn(move || {
        let status = Arc::new(Mutex::new(LoopStatus::default()));
        let result = engine.run(worker_tx, rx, &status, Lifetime::UntilStopped, || {});
        (engine, result)
    });
    let entered = waiting.recv_timeout(Duration::from_secs(5));
    let sent = tx.send(LoopMsg::Stop);
    let until = Instant::now() + Duration::from_secs(5);
    while !stop.requested() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    // The channel itself must reach the ordinary shutdown request while
    // the admitted parser chunk is still held. No external signal does it.
    let requested_while_held = stop.requested();
    let _ = release.send(());
    if !requested_while_held {
        stop.request();
    }
    let (engine, result) = worker.join().unwrap();
    assert!(entered.is_ok());
    assert!(sent.is_ok());
    assert!(
        requested_while_held,
        "Stop waited for the whole background pass"
    );
    assert!(result.unwrap().work.errors.is_empty());
    let reader = engine.reads.reader().unwrap();
    let progress = reader.input_progress(&input).unwrap().unwrap();
    assert!(
        progress.length > first.len() as u64,
        "new bytes already in hand must commit"
    );
    assert!(progress.length < complete.len() as u64);
    let stored = reader
        .stored_chain(&input, progress.chain, progress.length)
        .unwrap();
    assert_eq!(
        stored
            .read_range(&engine.blobs, 0, progress.length)
            .unwrap(),
        &complete.as_bytes()[..progress.length as usize]
    );
    drop(reader);
    drop(engine);
    drop(lock);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    let resumed = engine.run_pass().unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!(home.texts("stop-request").len(), 33);
}

#[test]
fn an_import_answers_before_queued_captures_or_replays_start_and_the_next_turn_finishes_them() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    use std::sync::mpsc::sync_channel;

    struct Input {
        session: String,
        path: PathBuf,
        id: String,
        bytes: String,
        captured_before: u64,
        before: Vec<Json>,
    }
    let queued = || {
        polylogue_store::timing::queues()
            .into_iter()
            .find(|(name, _)| *name == "parse")
            .unwrap()
            .1
    };
    for retained in [false, true] {
        let home = ClaudeHome::new();
        let files: Vec<_> = (0..12)
            .map(|n| {
                let session = format!("queued-{n:02}");
                let first = claude_user(&format!("first-{n:02}"), &session, "original body");
                let tail = claude_user(&format!("tail-{n:02}"), &session, "captured body");
                let name = format!("{session}.jsonl");
                home.append(&name, &first);
                if retained {
                    home.append(&name, &tail);
                }
                (session, name, first, tail)
            })
            .collect();
        let (mut engine, mut lock) = crate::import::tests::engine(&home, 4096);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        if retained {
            drop(engine);
            drop(lock);
            polylogue_testkit::delete_index(&home.root);
            for (_, name, _, _) in &files {
                std::fs::remove_file(home.file(name)).unwrap();
            }
            (engine, lock) = crate::import::tests::engine(&home, 4096);
            engine.config.reads_sources = false;
            engine.load().unwrap();
        } else {
            for (_, name, _, tail) in &files {
                home.append(name, tail);
            }
        }
        engine.config.workers = 1;
        engine.config.live_every = Duration::from_millis(10);
        let inputs: Vec<_> = files
            .into_iter()
            .map(|(session, name, first, tail)| {
                let path = home.file(&name);
                let state = engine.inputs.get_mut(&path).unwrap();
                state.origin = held_origin();
                let id = state.input_id.clone();
                Input {
                    session,
                    path,
                    before: source_rows(&engine, &id),
                    id,
                    captured_before: if retained {
                        first.len() + tail.len()
                    } else {
                        first.len()
                    } as u64,
                    bytes: first + &tail,
                }
            })
            .collect();
        let import = home.dir.path().join("queue-request.json");
        std::fs::write(
            &import,
            serde_json::to_vec(&serde_json::json!({
                "uuid":"queue-request", "chat_messages":[{
                    "uuid":"queue-answer", "sender":"human", "text":"ordinary import"
                }]
            }))
            .unwrap(),
        )
        .unwrap();
        READS.lock().unwrap().clear();
        let (entered, waiting) = channel();
        let (release, held) = channel();
        *CONTROL.lock().unwrap() = Some(Control {
            started: entered,
            release: held,
        });
        let stop = engine.stop_request();
        let (tx, rx) = channel();
        let worker_tx = tx.clone();
        let worker = std::thread::spawn(move || {
            let status = Arc::new(Mutex::new(LoopStatus::default()));
            let result = engine.run(worker_tx, rx, &status, Lifetime::UntilIdle, || {});
            (engine, result)
        });
        let entered = waiting.recv_timeout(Duration::from_secs(5));
        let until = Instant::now() + Duration::from_secs(5);
        while queued() < inputs.len() as i64 - 1 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let filled = queued() == inputs.len() as i64 - 1;
        let (reply, answer) = sync_channel(0);
        let sent = tx.send(LoopMsg::Import {
            paths: vec![import],
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
        // Observe the existing queue to release the in-hand parser only after
        // the loop has returned work that no worker read. No elapsed assertion
        // or delay decides whether the import is fair.
        let until = Instant::now() + Duration::from_secs(5);
        while queued() != 0 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let returned_before_release = queued() == 0;
        let _ = release.send(());
        let until = Instant::now() + Duration::from_secs(5);
        while home.count("SELECT count(*) AS n FROM session WHERE native_id='queue-request'") != 1
            && Instant::now() < until
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        // The synchronous reply keeps the loop at its real ACK boundary while
        // the committed capture and untouched queued inputs are inspected.
        let at_reply = READS.lock().unwrap().clone();
        let before_reply = {
            let reads = home.reads();
            let reader = reads.reader().unwrap();
            inputs.iter().map(|input| {
                let claims = reader.query(
                    "SELECT count(*) AS n FROM message_source m JOIN input_ref i ON i.pk=m.input_pk WHERE i.id=?1",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                ).unwrap()[0]["n"].as_u64().unwrap();
                let markers = reader.query(
                    "SELECT count(*) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.stopped_attempt IS NOT NULL",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                ).unwrap()[0]["n"].as_u64().unwrap();
                let captured = reader.query(
                    "SELECT max(length) AS n FROM a.revision WHERE input=?1",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                ).unwrap()[0]["n"].as_u64().unwrap();
                (input.path.clone(), claims, markers, captured)
            }).collect::<Vec<_>>()
        };
        let response = answer.recv_timeout(Duration::from_secs(5));
        drop(answer);
        if entered.is_err() || sent.is_err() || response.is_err() {
            stop.request();
            let _ = tx.send(LoopMsg::Stop);
        }
        let (mut engine, result) = worker.join().unwrap();
        assert!(entered.is_ok());
        assert!(filled, "the held worker must have queued siblings");
        assert!(sent.is_ok());
        assert!(
            returned_before_release,
            "unstarted jobs stayed ahead of the request"
        );
        assert_eq!(
            at_reply.len(),
            1,
            "the import waited for unstarted reads: {at_reply:?}"
        );
        let first = &at_reply[0];
        for (path, claims, markers, captured) in before_reply {
            assert_eq!(markers, 0, "a request is not a stopped reading: {path:?}");
            assert_eq!(
                claims,
                if &path == first {
                    2
                } else {
                    u64::from(!retained)
                }
            );
            let input = inputs.iter().find(|input| input.path == path).unwrap();
            assert_eq!(
                captured,
                if &path == first {
                    input.bytes.len() as u64
                } else {
                    input.captured_before
                }
            );
        }
        let report = response.unwrap().unwrap();
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert!(report.refused.is_empty(), "{report:?}");
        let summary = result.unwrap();
        assert!(summary.work.errors.is_empty(), "{summary:?}");
        assert_eq!(summary.imports, 1);
        assert!(!stop.requested());
        for input in &inputs {
            assert_eq!(
                home.texts(&input.session),
                ["original body", "captured body"]
            );
            let reader = engine.reads.reader().unwrap();
            let progress = reader.input_progress(&input.id).unwrap().unwrap();
            assert_eq!(progress.parsed_upto, input.bytes.len() as u64);
            let stored = reader
                .stored_chain(&input.id, progress.chain, progress.length)
                .unwrap();
            assert_eq!(
                stored
                    .read_range(&engine.blobs, 0, progress.length)
                    .unwrap(),
                input.bytes.as_bytes()
            );
            drop(reader);
            let captured = source_rows(&engine, &input.id);
            if retained {
                assert_eq!(captured, input.before);
            } else {
                assert_eq!(
                    captured.iter().filter(|r| r["row"] == "revision").count(),
                    2
                );
                assert_eq!(captured.iter().filter(|r| r["row"] == "extent").count(), 2);
            }
        }
        let before_repeat: Vec<_> = inputs
            .iter()
            .map(|input| source_rows(&engine, &input.id))
            .collect();
        let repeat = engine.run_pass().unwrap();
        assert!(repeat.errors.is_empty(), "{repeat:?}");
        assert_eq!((repeat.bytes_read, repeat.bytes_stored), (0, 0));
        for (input, captured) in inputs.iter().zip(before_repeat) {
            assert_eq!(source_rows(&engine, &input.id), captured);
        }
        drop(engine);
        drop(lock);
    }
}

#[cfg(feature = "faults")]
#[test]
fn stopping_at_chain_planning_cancels_the_engine_read_and_keeps_committed_progress() {
    use polylogue_store::fault::{self, Action};
    use polylogue_store::read::limit::ReadLimit;

    let home = ClaudeHome::new();
    home.append(
        "planning-stop.jsonl",
        &claude_user("one", "planning-stop", "retained body"),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&home.file("planning-stop.jsonl")]
        .input_id
        .clone();
    let before = source_rows(&engine, &input);
    let checkpoint = |engine: &Engine| {
        engine.reads.reader().unwrap().query(
            "SELECT p.chain,p.reading_version,p.status,p.parsed_upto,p.at_length,p.records,p.used,p.skipped,p.unreadable,CASE WHEN p.checkpoint IS NULL THEN NULL ELSE hex(p.checkpoint) END AS checkpoint,p.reason FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 ORDER BY p.chain",
            &[polylogue_store::SqlValue::Text(input.clone())],
        ).unwrap()
    };
    let before_checkpoint = checkpoint(&engine);
    let parent = ReadLimit::without_deadline(Default::default());
    let stop = engine.stop_request();
    fault::clear();
    fault::arm("read.chains_behind_index", "", Action::Hold, 0, 1);
    let worker_parent = parent.clone();
    let work = std::thread::spawn(move || {
        let result = worker_parent.scope(|| {
            let result = engine.run_pass();
            assert!(ReadLimit::current().unwrap().check().is_ok());
            result
        });
        (engine, result)
    });
    let until = Instant::now() + Duration::from_secs(10);
    while !fault::held() && !work.is_finished() && Instant::now() < until {
        std::thread::yield_now();
    }
    let reached = fault::held();
    stop.request();
    fault::release();
    let (engine, result) = work.join().unwrap();
    fault::clear();
    assert!(
        reached,
        "pass did not reach the retained-chain planning reader"
    );
    assert!(matches!(result, Err(EngineError::Stopped)), "{result:?}");
    assert!(
        parent.check().is_ok(),
        "engine stop cancelled an independent caller"
    );
    assert_eq!(source_rows(&engine, &input), before);
    assert_eq!(checkpoint(&engine), before_checkpoint);
    assert!(
        engine
            .reads
            .reader()
            .unwrap()
            .query("SELECT 1", &[])
            .is_ok()
    );
}

#[test]
fn engine_stop_interrupts_a_running_statement_without_closing_other_readers() {
    use polylogue_store::read::limit::ReadLimit;
    let home = ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 4096);
    let stop = engine.stop_request();
    let limit = ReadLimit::without_deadline(stop.0.reads.clone());
    let reads = &engine.reads;
    std::thread::scope(|threads| {
        let read_limit = &limit;
        let (started, ready) = channel();
        let work = threads.spawn(move || read_limit.scope(|| {
            let reader = reads.reader()?;
            reader.query("SELECT 1", &[])?;
            started.send(read_limit.sqlite_steps()).unwrap();
            reader.query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n", &[])
        }));
        let before = ready.recv().unwrap();
        let until = Instant::now() + Duration::from_secs(10);
        while limit.sqlite_steps() == before && Instant::now() < until {
            std::thread::yield_now();
        }
        let running = limit.sqlite_steps() > before;
        stop.request();
        let result = work.join().unwrap();
        assert!(running, "statement did not reach SQLite");
        assert!(
            matches!(result, Err(polylogue_store::StoreError::ReadCancelled)),
            "{result:?}"
        );
    });
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .query("SELECT 7 AS n", &[])
            .unwrap()[0]["n"],
        7
    );
}
