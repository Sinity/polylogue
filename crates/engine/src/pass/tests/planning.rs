use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use polylogue_store::fault::{self, Action};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel, sync_channel};

fn histories(n: usize) -> ClaudeHome {
    let home = ClaudeHome::new();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    for body in ["first", "other", "first"] {
        for i in 0..n {
            let session = format!("scan-{i:04}");
            std::fs::write(
                home.file(&format!("{session}.jsonl")),
                claude_user(&format!("u-{i:04}"), &session, &format!("{body} {i}")),
            )
            .unwrap();
        }
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    drop(engine);
    home
}

fn kept_request(engine: &mut Engine, home: &ClaudeHome) -> PathBuf {
    let request = home.dir.path().join("request.json");
    std::fs::write(
        &request,
        serde_json::to_vec(&serde_json::json!({
            "uuid": "queued-import", "chat_messages": [{
                "uuid": "imported-message", "sender": "human", "text": "explicit request body"
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let kept = engine
        .import_with_options(std::slice::from_ref(&request), &ImportOptions::default())
        .unwrap();
    assert!(kept.pass.errors.is_empty(), "{kept:?}");
    assert!(kept.refused.is_empty(), "{kept:?}");
    assert_eq!(kept.imported.len(), 1, "{kept:?}");
    assert!(kept.pass.bytes_stored > 0, "{kept:?}");
    request
}

fn pending_current_chains(engine: &Engine, inputs: &[String]) -> usize {
    engine
        .reads
        .reader()
        .unwrap()
        .chains_behind_index(
            &[(
                "claude-code".into(),
                polylogue_sources::origin::by_name("claude-code")
                    .unwrap()
                    .reading_version(),
            )],
            polylogue_store::StoppedChains::Finished,
            RevisionChains::Newest,
            Some(inputs),
        )
        .unwrap()
        .len()
}

/// A real import arrives while the next planning statement is held. Joining the
/// enqueueing thread before rearming the fault makes both interruptions exact.
fn interrupt(engine: &mut Engine, send: Sender<LoopMsg>, request: PathBuf) -> Pass {
    engine.load().unwrap();
    fault::arm("read.chains_behind_index", "", Action::Hold, 1, 1);
    let finished = Arc::new(AtomicBool::new(false));
    let done = Arc::clone(&finished);
    let incoming = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(30);
        while !fault::held() && !done.load(Ordering::Acquire) && Instant::now() < until {
            std::thread::yield_now();
        }
        if !fault::held() {
            return None;
        }
        let (reply, wait) = sync_channel(1);
        send.send(LoopMsg::Import {
            paths: vec![request],
            format: None,
            from_archive: None,
            dry_run: false,
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            parent: None,
            reply,
        })
        .unwrap();
        fault::release();
        Some(wait.recv_timeout(Duration::from_secs(5)).unwrap())
    });
    let mut pass = Pass::default();
    let result = engine.reparse_behind(RevisionChains::Newest, &mut pass);
    finished.store(true, Ordering::Release);
    if let Some(message) = engine.requests.as_mut().unwrap().queued() {
        engine.answer(message, &|_| {}, true, &mut RunSummary::default());
    }
    let reply = incoming.join().unwrap();
    result.unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reply = reply
        .expect("catch-up reached another bounded planning page")
        .unwrap();
    assert!(reply.pass.errors.is_empty(), "{reply:?}");
    assert!(reply.refused.is_empty(), "{reply:?}");
    assert_eq!(reply.already_stored.len(), 1, "{reply:?}");
    assert_eq!((reply.pass.bytes_read, reply.pass.bytes_stored), (0, 0));
    pass
}

#[test]
fn queued_imports_resume_catchup_without_restarting_completed_pages() {
    fault::clear();
    let n = 2 * IN_POOL_PER_WORKER + 9;
    let home = histories(n);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.load().unwrap();
    let ids: Vec<_> = engine
        .inputs
        .values()
        .map(|input| input.input_id.clone())
        .collect();
    assert_eq!(ids.len(), n);
    let request = kept_request(&mut engine, &home);
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    engine.discard_confirmed_state();
    engine.config.reads_sources = false;
    let first_page = engine
        .reads
        .reader()
        .unwrap()
        .input_ids_after(None, IN_POOL_PER_WORKER)
        .unwrap();
    let (send, receive) = channel();
    engine.requests = Some(Requests::new(receive));
    fault::arm(
        "read.chains_behind_index",
        "all inputs",
        Action::Fail,
        0,
        u32::MAX,
    );
    let first = interrupt(&mut engine, send.clone(), request.clone());
    assert!(
        first.bytes_read > 0,
        "first page parsed its actual stored bytes"
    );
    fault::clear();
    // Confirm a source below the completed page without local discovery. Its
    // bytes must be read in the next finite scan, rather than losing it behind
    // the bookmark or restarting the current scan at the first key.
    let late = (0..1000)
        .map(|i| home.dir.path().join(format!("late-{i}.jsonl")))
        .find(|path| engine.location_id(path) < *first_page.last().unwrap())
        .unwrap();
    let late_id = engine.location_id(&late);
    let bytes = claude_user("late", "late", "arrived behind the planning bookmark");
    let mut raw = polylogue_store::fixture::writer::step(
        &late_id,
        "claude-code",
        late.to_str().unwrap(),
        Default::default(),
        0,
        0,
    );
    raw.parse.clear();
    raw.length = bytes.len() as u64;
    raw.observation = Some(engine.observation(&late, None));
    let prepared = engine.archive.prepare_bytes(bytes.as_bytes()).unwrap();
    engine.writer.store_bytes(prepared, raw).unwrap();
    assert!(
        !late.exists(),
        "only confirmed original bytes are available"
    );
    fault::arm(
        "read.chains_behind_index",
        "all inputs",
        Action::Fail,
        0,
        u32::MAX,
    );
    fault::arm(
        "read.chains_behind_index",
        &first_page[0],
        Action::Fail,
        0,
        u32::MAX,
    );
    let second = interrupt(&mut engine, send, request);
    assert!(
        second.bytes_read > 0,
        "a later page made progress after the import"
    );
    fault::clear();
    fault::arm(
        "read.chains_behind_index",
        "all inputs",
        Action::Fail,
        0,
        u32::MAX,
    );
    fault::arm(
        "read.chains_behind_index",
        &first_page[0],
        Action::Fail,
        0,
        u32::MAX,
    );
    let mut last = Pass::default();
    let result = engine.reparse_behind(RevisionChains::Newest, &mut last);
    fault::clear();
    result.unwrap();
    assert!(last.errors.is_empty(), "{last:?}");
    assert_eq!(pending_current_chains(&engine, &ids), 0);
    assert_eq!(
        pending_current_chains(&engine, std::slice::from_ref(&late_id)),
        1
    );
    // The ordinary pass resumes Older, then completes every retained chain.
    let complete = engine.run_pass().unwrap();
    assert!(complete.errors.is_empty(), "{complete:?}");
    assert!(
        complete.capture_pending,
        "a source changed behind this scan"
    );
    let late_pass = engine.run_pass().unwrap();
    assert!(late_pass.errors.is_empty(), "{late_pass:?}");
    assert!(!late_pass.capture_pending, "{late_pass:?}");
    assert_eq!(pending_current_chains(&engine, &[late_id]), 0);
    assert_eq!(home.texts("late"), ["arrived behind the planning bookmark"]);
    for i in 0..n {
        assert_eq!(home.texts(&format!("scan-{i:04}")), [format!("first {i}")]);
    }
}

#[test]
fn a_failed_planning_page_retries_without_repeating_completed_pages() {
    fault::clear();
    let home = histories(IN_POOL_PER_WORKER + 2);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.config.reads_sources = false;
    engine.load().unwrap();
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let ids = engine
        .reads
        .reader()
        .unwrap()
        .input_ids_after(None, 2 * IN_POOL_PER_WORKER)
        .unwrap();
    fault::arm("read.chains_behind_index", "", Action::Fail, 1, 1);
    let mut pass = Pass::default();
    let failed = engine.reparse_behind(RevisionChains::Newest, &mut pass);
    fault::clear();
    assert!(failed.is_err());
    assert!(pass.bytes_read > 0);
    assert_eq!(pending_current_chains(&engine, &ids), 2);
    fault::arm(
        "read.chains_behind_index",
        &ids[0],
        Action::Fail,
        0,
        u32::MAX,
    );
    let result = engine.reparse_behind(RevisionChains::Newest, &mut pass);
    fault::clear();
    result.unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pending_current_chains(&engine, &ids), 0);
}

#[test]
fn a_scope_reset_revisits_readings_before_the_planning_bookmark() {
    fault::clear();
    let home = histories(IN_POOL_PER_WORKER + 2);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.config.reads_sources = false;
    let request = kept_request(&mut engine, &home);
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let ids = engine
        .reads
        .reader()
        .unwrap()
        .input_ids_after(None, 2 * IN_POOL_PER_WORKER)
        .unwrap();
    let (send, receive) = channel();
    engine.requests = Some(Requests::new(receive));
    interrupt(&mut engine, send, request);
    fault::clear();
    assert!(pending_current_chains(&engine, &ids) > 0);
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    engine.archive.advance_generation_for_test().unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(!pass.capture_pending, "{pass:?}");
    assert_eq!(pending_current_chains(&engine, &ids), 0);
}

#[test]
fn the_daemon_loop_keeps_its_planning_page_when_refreshing_committed_progress() {
    fault::clear();
    let home = histories(IN_POOL_PER_WORKER + 2);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.config.reads_sources = false;
    let request = kept_request(&mut engine, &home);
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let first = engine
        .reads
        .reader()
        .unwrap()
        .input_ids_after(None, IN_POOL_PER_WORKER)
        .unwrap();
    let completed_page = format!("Newest:{}", serde_json::to_string(&first).unwrap());
    fault::arm(
        "read.chains_behind_index",
        "all inputs",
        Action::Fail,
        0,
        u32::MAX,
    );
    fault::arm("read.chains_behind_index", "Newest:", Action::Hold, 1, 1);
    let (send, receive) = channel();
    let incoming = send.clone();
    let enqueue = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(30);
        while !fault::held() {
            assert!(Instant::now() < until);
            std::thread::yield_now();
        }
        fault::arm(
            "read.chains_behind_index",
            &completed_page,
            Action::Fail,
            0,
            u32::MAX,
        );
        let (reply, wait) = sync_channel(1);
        incoming
            .send(LoopMsg::Import {
                paths: vec![request],
                format: None,
                from_archive: None,
                dry_run: false,
                authorship: Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                },
                parent: None,
                reply,
            })
            .unwrap();
        fault::release();
        wait.recv_timeout(Duration::from_secs(5)).unwrap().unwrap()
    });
    let result = engine.run(send, receive, &Arc::default(), Lifetime::UntilIdle, || {});
    let reply = enqueue.join().unwrap();
    fault::clear();
    let summary = result.unwrap();
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    assert_eq!(summary.imports, 1);
    assert!(reply.refused.is_empty(), "{reply:?}");
    assert_eq!(reply.already_stored.len(), 1, "{reply:?}");
    assert_eq!((reply.pass.bytes_read, reply.pass.bytes_stored), (0, 0));
    let ids = engine
        .reads
        .reader()
        .unwrap()
        .input_ids_after(None, 2 * IN_POOL_PER_WORKER)
        .unwrap();
    assert_eq!(pending_current_chains(&engine, &ids), 0);
}

#[test]
fn older_reading_copies_are_drained_across_multiple_planning_pages() {
    let n = IN_POOL_PER_WORKER + 2;
    let home = histories(n);
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.config.reads_sources = false;
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.readings_copied, n as u64, "{pass:?}");
    assert!(!pass.capture_pending, "{pass:?}");
    for i in 0..n {
        assert_eq!(home.texts(&format!("scan-{i:04}")), [format!("first {i}")]);
    }
}
