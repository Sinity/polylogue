use super::*;
use polylogue_model::Json;
use polylogue_sources::{Arrival, Capture, OriginSpec, Parse, Root, Step, StepParser};
use std::sync::mpsc::Sender;

fn source_rows(engine: &Engine, input: &str) -> Vec<Json> {
    engine.reads.reader().unwrap().query(
        "SELECT 'revision' AS row,number AS n,chain,length,produced AS detail FROM a.revision WHERE input=?1
         UNION ALL SELECT 'extent',offset,chain,length,hex(blob) FROM a.extent WHERE input=?1 ORDER BY row,n",
        &[polylogue_store::SqlValue::Text(input.to_owned())],
    ).unwrap()
}

#[test]
fn a_completed_scope_reset_reloads_confirmed_state_without_recapturing() {
    let home = Home::new();
    let text = claude_user("one", "scope", "original confirmed body");
    home.append("scope.jsonl", &text);
    let path = home.file("scope.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let chain = engine.inputs[&path].chain.unwrap();
    let captured = source_rows(&engine, &input);
    let checkpoint = engine
        .reads
        .reader()
        .unwrap()
        .parser_checkpoint(&input, chain, &engine.inputs[&path].origin.parse_version())
        .unwrap();
    assert!(
        checkpoint
            .as_ref()
            .and_then(|stored| stored.checkpoint.as_ref())
            .and_then(|checkpoint| checkpoint.parser_state.as_ref())
            .is_some(),
        "Claude continuation is committed with its cursor"
    );
    engine.archive.advance_generation_for_test().unwrap();
    let (resumed, _) = engine.one_pass(false, &Default::default()).unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!(resumed.full_scans, 1, "the new scope repeats discovery");
    assert_eq!((resumed.bytes_read, resumed.bytes_stored), (0, 0));
    assert_eq!(engine.inputs[&path].input_id, input);
    assert_eq!(engine.inputs[&path].chain, Some(chain));
    assert_eq!(engine.inputs[&path].parsed_upto, text.len() as u64);
    assert_eq!(source_rows(&engine, &input), captured);
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .parser_checkpoint(&input, chain, &engine.inputs[&path].origin.parse_version())
            .unwrap(),
        checkpoint
    );
    home.append(
        "scope.jsonl",
        &claude_user("two", "scope", "body after reset"),
    );
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let session = polylogue_model::SessionId::parse("claude-code:scope").unwrap();
    let messages = engine
        .reads
        .reader()
        .unwrap()
        .messages(&session, false)
        .unwrap();
    assert_eq!(
        messages
            .iter()
            .map(|message| message.text())
            .collect::<Vec<_>>(),
        ["original confirmed body", "body after reset"]
    );
}

#[test]
fn a_cached_scope_cannot_be_rebound_after_its_entry_check() {
    let home = Home::new();
    home.append(
        "scope-race.jsonl",
        &claude_user("one", "scope-race", "committed before the reset"),
    );
    let path = home.file("scope-race.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let input = engine.inputs[&path].input_id.clone();
    let captured = source_rows(&engine, &input);
    // This is the valid entry check, followed by a complete reset before
    // the shared load/turn step that takes the token runs.
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
    ) -> Result<(Step, Json), polylogue_sources::ParseError> {
        READS.lock().unwrap().push(self.1.clone());
        let control = CONTROL.lock().unwrap().take();
        if let Some(control) = control {
            control.started.send(()).unwrap();
            control.release.recv().unwrap();
        }
        self.0.step(state, bytes, base, finished)
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
    let claude_code = polylogue_sources::origin::by_name("claude-code").unwrap();
    Box::leak(Box::new(OriginSpec {
        name: claude_code.name,
        harness_prefixes: claude_code.harness_prefixes,
        modules: claude_code.modules,
        support: Support::Parsed,
        arrival: Arrival::Live,
        route: polylogue_model::Route::Live,
        input_identity: polylogue_sources::InputIdentity::Location,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: claude_code.recognize,
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
    let home = Home::new();
    let first = claude_user("one", "scope-session", "first confirmed body");
    home.append("scope-session.jsonl", &first);
    let path = home.file("scope-session.jsonl");
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.live_every = Duration::ZERO;
    assert!(engine.run_pass().unwrap().errors.is_empty());
    // Pause the real Claude parser without changing its version or output.
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
            let (tx, rx) = crossbeam_channel::unbounded();
            let status = Arc::new(SharedLoopStatus::default());
            let result = engine.run(tx, rx, &status, Lifetime::UntilIdle, || {});
            (engine, result)
        });
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("real parser entered after reading the suffix");
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

#[cfg(feature = "faults")]
#[test]
fn queued_import_yields_then_resumes_from_saved_positions() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    use polylogue_store::fault::{self, Action};
    use std::sync::mpsc::sync_channel;

    for (kept, reads_sources) in [(false, true), (true, false), (true, true)] {
        for stop_after_import in [false, true] {
            let home = Home::new();
            let first = claude_user("first", "request-turn", "first stored record");
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
            if kept {
                home.append("request-turn.jsonl", &suffix);
            }
            let path = home.file("request-turn.jsonl");
            let (mut engine, mut lock) = crate::import::tests::engine(&home, 256);
            assert!(engine.run_pass().unwrap().errors.is_empty());
            let input = engine.inputs[&path].input_id.clone();
            if kept {
                drop(engine);
                drop(lock);
                // Synthetic cold replay: every original byte stays in the
                // register and the file reader cannot reopen the source.
                polylogue_testkit::delete_index(&home.root);
                std::fs::remove_file(&path).unwrap();
                (engine, lock) = crate::import::tests::engine(&home, 256);
                engine.config.reads_sources = reads_sources;
                engine.load().unwrap();
            } else {
                home.append("request-turn.jsonl", &suffix);
            }
            let undiscovered = kept && reads_sources;
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
            fault::clear();
            fault::arm("parse", "request-turn.jsonl", Action::Hold, 0, 1);
            let (tx, rx) = crossbeam_channel::unbounded();
            let worker_tx = tx.clone();
            let status = Arc::new(SharedLoopStatus::default());
            let worker = std::thread::spawn(move || {
                let result = engine.run(worker_tx, rx, &status, Lifetime::UntilIdle, || {});
                (engine, result)
            });
            let until = Instant::now() + Duration::from_secs(5);
            while !fault::held() && !worker.is_finished() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(5));
            }
            let entered = fault::held();
            // Rendezvous keeps the engine in the real import response until
            // the test checks the committed background prefix.
            let (reply, answer) = sync_channel(0);
            let queued = tx.send(LoopMsg::Import {
                paths: vec![import],
                format: None,
                dry_run: false,
                authorship: Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                },
                parent: None,
                reply,
            });
            fault::release();
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
            let cursor = home.read(|reader| reader.query(
                "SELECT COALESCE(max(p.parsed_upto),0) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1",
                &[polylogue_store::SqlValue::Text(input.clone())],
            ))[0]["n"].as_u64().unwrap();
            let captured = home.read(|reader| {
                reader.query(
                    "SELECT max(length) AS n FROM a.revision WHERE input=?1",
                    &[polylogue_store::SqlValue::Text(input.clone())],
                )
            })[0]["n"]
                .as_u64()
                .unwrap();
            let response = answer.recv_timeout(Duration::from_secs(5));
            drop(answer);
            if stop_after_import || response.is_err() || !imported || !entered {
                let _ = tx.send(LoopMsg::Stop);
            }
            let (mut engine, result) = worker.join().unwrap();
            fault::clear();
            assert!(entered, "the parser did not reach its held chunk");
            assert!(queued.is_ok());
            assert!(
                imported,
                "a request waited behind the rest of the stored history"
            );
            assert!(
                cursor < complete.len() as u64,
                "the import answered only after the whole input: {cursor}"
            );
            if kept {
                assert_eq!(captured, complete.len() as u64);
                assert_eq!(source_rows(&engine, &input), captured_before);
            } else {
                assert!(
                    captured > first.len() as u64,
                    "the committed capture chunk must commit before yielding"
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
            assert_eq!(
                home.texts("request-turn"),
                std::iter::once("first stored record".to_owned())
                    .chain((0..48).map(|n| format!("tail {n} {}", "x".repeat(512))))
                    .collect::<Vec<_>>()
            );
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
fn a_stop_keeps_the_bytes_in_hand_without_reading_the_whole_input() {
    let home = Home::new();
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
    let stop = engine.archive.stop().clone();
    let (entered, waiting) = channel();
    let (release, held) = channel();
    *CONTROL.lock().unwrap() = Some(Control {
        started: entered,
        release: held,
    });
    let (tx, rx) = crossbeam_channel::unbounded();
    let worker_tx = tx.clone();
    let worker = std::thread::spawn(move || {
        let status = Arc::new(SharedLoopStatus::default());
        let result = engine.run(worker_tx, rx, &status, Lifetime::UntilStopped, || {});
        (engine, result)
    });
    // The stop is made while the parser holds its first chunk.
    let entered = waiting.recv_timeout(Duration::from_secs(5));
    stop.cancel();
    let _ = release.send(());
    let (engine, result) = worker.join().unwrap();
    assert!(entered.is_ok());
    assert!(result.unwrap().work.errors.is_empty());
    let reads = polylogue_store::ReadPool::open(&engine.archive.reopened()).unwrap();
    let reader = reads.reader().unwrap();
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
    drop(reads);
    drop(engine);
    drop(lock);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    let resumed = engine.run_pass().unwrap();
    assert!(resumed.errors.is_empty(), "{resumed:?}");
    assert_eq!(
        home.texts("stop-request"),
        std::iter::once("initial body".to_owned())
            .chain(std::iter::repeat_n("x".repeat(512), 32))
            .collect::<Vec<_>>()
    );
}

#[cfg(feature = "faults")]
#[test]
fn an_import_answers_before_queued_captures_start_and_the_next_turn_finishes_them() {
    use polylogue_model::authorship::{Author, Authorship, Channel};
    use polylogue_store::fault::{self, Action};
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
    // Stored chains read again after the index is deleted are the build's
    // work, which yields to a request between ranges
    // (queued_import_yields_then_resumes_from_saved_positions).
    {
        let kept = false;
        let home = Home::new();
        let files: Vec<_> = (0..12)
            .map(|n| {
                let session = format!("queued-{n:02}");
                let first = claude_user(&format!("first-{n:02}"), &session, "original body");
                let tail = claude_user(&format!("tail-{n:02}"), &session, "captured body");
                let name = format!("{session}.jsonl");
                home.append(&name, &first);
                if kept {
                    home.append(&name, &tail);
                }
                (session, name, first, tail)
            })
            .collect();
        let (mut engine, mut lock) = crate::import::tests::engine(&home, 4096);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let ids: HashMap<_, _> = engine
            .inputs
            .iter()
            .map(|(path, state)| (path.clone(), state.input_id.clone()))
            .collect();
        if kept {
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
                let id = ids[&path].clone();
                Input {
                    session,
                    path,
                    before: source_rows(&engine, &id),
                    id,
                    captured_before: if kept {
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
        fault::clear();
        fault::arm("parse", "queued-", Action::Hold, 0, 1);
        let stop = engine.archive.stop().clone();
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker_tx = tx.clone();
        let worker = std::thread::spawn(move || {
            let status = Arc::new(SharedLoopStatus::default());
            let result = engine.run(worker_tx, rx, &status, Lifetime::UntilIdle, || {});
            (engine, result)
        });
        let until = Instant::now() + Duration::from_secs(5);
        while !fault::held() && !worker.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let entered = fault::held();
        let until = Instant::now() + Duration::from_secs(5);
        while queued() < inputs.len() as i64 - 1 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let filled = queued() == inputs.len() as i64 - 1;
        let (reply, answer) = sync_channel(0);
        let sent = tx.send(LoopMsg::Import {
            paths: vec![import],
            format: None,
            dry_run: false,
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            parent: None,
            reply,
        });
        // Observe the existing queue to release the in-hand parser only after
        // the loop has returned work that no worker read. No elapsed annotation
        // or delay decides whether the import is fair.
        let until = Instant::now() + Duration::from_secs(5);
        while queued() != 0 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        let returned_before_release = queued() == 0;
        fault::release();
        let until = Instant::now() + Duration::from_secs(5);
        while home.count("SELECT count(*) AS n FROM session WHERE native_id='queue-request'") != 1
            && Instant::now() < until
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        // The synchronous reply keeps the loop at its real ACK boundary while
        // the committed capture and untouched queued inputs are inspected.
        // A read that meets the first build's install starts again on the
        // installed index.
        let before_reply = home.read(|reader| {
            inputs.iter().map(|input| {
                let message_parses = reader.query(
                    "SELECT count(*) AS n FROM message_parse m JOIN input_ref i ON i.pk=m.input_pk WHERE i.id=?1",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                )?[0]["n"].as_u64().unwrap();
                let markers = reader.query(
                    "SELECT count(*) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1 AND p.stopped_attempt IS NOT NULL",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                )?[0]["n"].as_u64().unwrap();
                let captured = reader.query(
                    "SELECT max(length) AS n FROM a.revision WHERE input=?1",
                    &[polylogue_store::SqlValue::Text(input.id.clone())],
                )?[0]["n"].as_u64().unwrap();
                Ok((input.path.clone(), message_parses, markers, captured))
            }).collect::<polylogue_store::Result<Vec<_>>>()
        });
        let response = answer.recv_timeout(Duration::from_secs(5));
        drop(answer);
        if !entered || sent.is_err() || response.is_err() {
            let _ = tx.send(LoopMsg::Stop);
        }
        let (mut engine, result) = worker.join().unwrap();
        fault::clear();
        assert!(entered);
        assert!(filled, "the held worker must have queued siblings");
        assert!(sent.is_ok());
        assert!(
            returned_before_release,
            "unstarted jobs stayed ahead of the request"
        );
        // One worker was held while every sibling left the queue. The
        // committed message sources independently identify the completed job.
        let completed: Vec<_> = before_reply
            .iter()
            .filter(|(_, message_parses, _, _)| *message_parses == 2)
            .collect();
        assert_eq!(
            completed.len(),
            1,
            "unstarted reads ran before the import: {before_reply:?}"
        );
        let first = completed[0].0.clone();
        for (path, message_parses, markers, captured) in before_reply {
            assert_eq!(markers, 0, "a request is not a stopped reading: {path:?}");
            assert_eq!(
                message_parses,
                if path == first { 2 } else { u64::from(!kept) }
            );
            let input = inputs.iter().find(|input| input.path == path).unwrap();
            assert_eq!(
                captured,
                if path == first {
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
        assert!(!stop.is_cancelled());
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
            if kept {
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
