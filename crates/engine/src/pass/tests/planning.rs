use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use polylogue_store::fault::{self, Action};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;

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
                    .parse_version(),
            )],
            RevisionChains::Newest,
            Some(inputs),
        )
        .unwrap()
        .len()
}

/// A real import arrives while the next planning statement is held. Joining the
/// enqueueing thread before rearming the fault makes both interruptions exact.
fn interrupt(
    engine: &mut Engine,
    send: crossbeam_channel::Sender<LoopMsg>,
    request: PathBuf,
) -> Pass {
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
    let (send, receive) = crossbeam_channel::unbounded();
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
    let (send, receive) = crossbeam_channel::unbounded();
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
    let (send, receive) = crossbeam_channel::unbounded();
    let incoming = send.clone();
    let enqueue = std::thread::spawn(move || {
        assert!(
            fault::wait_held_within(std::time::Duration::from_secs(30)),
            "no thread reached the held point"
        );
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

/// A pass whose stored chains must all be read again still stores a new
/// file first: storing never waits behind that reading.
#[test]
fn a_new_file_is_stored_before_the_stored_backlog_is_read() {
    fault::clear();
    let home = ClaudeHome::new();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    let mut backlog = Vec::new();
    for i in 0..6 {
        let session = format!("backlog-{i}");
        std::fs::write(
            home.file(&format!("{session}.jsonl")),
            claude_user(&format!("u-{i}"), &session, &format!("stored {i}")),
        )
        .unwrap();
        backlog.push(crate::location_id(
            "test-machine",
            &home.file(&format!("{session}.jsonl")),
        ));
    }
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    // Every stored chain now needs reading again.
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    engine.discard_confirmed_state();
    // The new file's input sorts after every stored one, where reading by
    // input alone would put it.
    let (path, bytes) = (0..)
        .map(|n| home.file(&format!("fresh-{n}.jsonl")))
        .find(|path| {
            backlog
                .iter()
                .all(|id| crate::location_id("test-machine", path) > *id)
        })
        .map(|path| {
            let bytes = claude_user("u-fresh", "fresh", "stored while the backlog waits");
            std::fs::write(&path, &bytes).unwrap();
            (path, bytes)
        })
        .unwrap();

    fault::arm("parse", "backlog-", Action::Hold, 0, 1);
    let archive = engine.archive.clone();
    let watched = path.clone();
    let observer = std::thread::spawn(move || {
        fault::wait_held();
        let pool = polylogue_store::ReadPool::open(&archive).unwrap();
        let stored = pool
            .reader()
            .unwrap()
            .known_input_at("test-machine", watched.as_os_str().as_encoded_bytes())
            .unwrap()
            .map(|known| known.stored);
        fault::release();
        stored
    });
    let pass = engine.run_pass();
    let stored = observer.join().unwrap();
    fault::clear();
    let pass = pass.unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        stored,
        Some(bytes.len() as u64),
        "the new file was not stored before the stored chains were read"
    );
    assert_eq!(home.texts("fresh"), ["stored while the backlog waits"]);
    for i in 0..6 {
        assert_eq!(home.texts(&format!("backlog-{i}")), [format!("stored {i}")]);
    }
}

/// A pass whose stored chains must all be read again reads a session file
/// that grew before that backlog: a live session never waits behind it
/// (docs/architecture.md, live inputs go before bulk work).
#[test]
fn a_grown_file_is_read_before_the_stored_backlog() {
    fault::clear();
    let home = ClaudeHome::new();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    for i in 0..6 {
        let session = format!("backlog-{i}");
        std::fs::write(
            home.file(&format!("{session}.jsonl")),
            claude_user(&format!("u-{i}"), &session, &format!("stored {i}")),
        )
        .unwrap();
    }
    home.append("live.jsonl", &claude_user("l1", "live", "first record"));
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    // Every stored chain now needs reading again, and the session grows.
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    engine.discard_confirmed_state();
    home.append("live.jsonl", &claude_user("l2", "live", "written since"));

    fault::arm("parse", "backlog-", Action::Hold, 0, 1);
    let archive = engine.archive.clone();
    let observer = std::thread::spawn(move || {
        fault::wait_held();
        let pool = polylogue_store::ReadPool::open(&archive).unwrap();
        let texts: Vec<String> = pool
            .reader()
            .unwrap()
            .messages(
                &polylogue_model::SessionId::parse("claude-code:live").unwrap(),
                false,
            )
            .unwrap()
            .iter()
            .map(|message| message.text())
            .collect();
        fault::release();
        texts
    });
    let pass = engine.run_pass();
    let seen_during_backlog = observer.join().unwrap();
    fault::clear();
    let pass = pass.unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        seen_during_backlog,
        ["first record", "written since"],
        "the grown file was not read before the stored chains"
    );
    for i in 0..6 {
        assert_eq!(home.texts(&format!("backlog-{i}")), [format!("stored {i}")]);
    }
}

/// The catch-up of stored chains stops at the end of a page when a live
/// session changed, so the pass ends and the next one reads that session
/// first; the catch-up then resumes after the page it finished.
#[test]
fn the_stored_backlog_stops_at_a_page_for_a_changed_live_session() {
    fault::clear();
    let home = histories(IN_POOL_PER_WORKER + 2);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.workers = 1;
    engine.load().unwrap();
    let ids: Vec<_> = engine
        .inputs
        .values()
        .map(|input| input.input_id.clone())
        .collect();
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    engine.discard_confirmed_state();
    let live = home.file("scan-0000.jsonl");
    fault::arm("read.chains_behind_index", "Newest:[", Action::Hold, 0, 1);
    let pass = std::thread::scope(|scope| {
        let running = scope.spawn(|| engine.run_pass());
        assert!(
            fault::wait_held_within(Duration::from_secs(60)),
            "the catch-up planned its first page"
        );
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&live)
            .unwrap();
        std::io::Write::write_all(
            &mut file,
            claude_user("u-live", "scan-0000", "written during the backlog").as_bytes(),
        )
        .unwrap();
        fault::release();
        running.join().unwrap()
    });
    fault::clear();
    let pass = pass.unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.capture_pending, "the pass ended for the live session");
    assert!(
        pending_current_chains(&engine, &ids) > 0,
        "the catch-up stopped after its first page"
    );
    let next = engine.run_pass().unwrap();
    assert!(next.errors.is_empty(), "{next:?}");
    assert_eq!(
        home.texts("scan-0000"),
        ["first 0", "written during the backlog"]
    );
    assert_eq!(pending_current_chains(&engine, &ids), 0);
    assert_eq!(home.texts("scan-0001"), ["first 1"]);
}

#[test]
fn a_later_rebuild_page_does_not_include_an_append_after_the_cut() {
    fault::clear();
    let home = ClaudeHome::new();
    let n = crate::build::PLANNED_INPUTS + 1;
    for i in 0..n {
        let session = format!("cut-{i:04}");
        home.append(
            &format!("{session}.jsonl"),
            &claude_user(&format!("initial-{i}"), &session, "before the cut"),
        );
    }
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let report = engine.run_pass().unwrap();
        assert!(report.errors.is_empty(), "{report:?}");
    }
    polylogue_testkit::delete_index(&home.root);

    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.reads_sources = false;
    let (session, path) = (0..n)
        .map(|i| {
            let session = format!("cut-{i:04}");
            let path = home.file(&format!("{session}.jsonl"));
            (session, path)
        })
        .max_by_key(|(_, path)| engine.location_id(path))
        .unwrap();
    let id = engine.location_id(&path);
    let from = std::fs::metadata(&path).unwrap().len();
    let append = claude_user("after-cut", &session, "arrived while rebuilding");
    let mut raw = polylogue_store::fixture::writer::step(
        &id,
        "claude-code",
        path.to_str().unwrap(),
        Default::default(),
        from,
        0,
    );
    raw.parse.clear();
    raw.length = from + append.len() as u64;
    raw.observation = Some(engine.observation(&path, None));
    let prepared = engine.archive.prepare_bytes(append.as_bytes()).unwrap();
    let writer = engine.writer();
    fault::arm("read.chains_behind_index", "All:[", Action::Hold, 0, 1);
    let report = std::thread::scope(|scope| {
        let building = scope.spawn(|| engine.build_index());
        assert!(
            fault::wait_held_within(Duration::from_secs(60)),
            "the rebuild reached its first planning page"
        );
        let stored = writer.store_bytes(prepared, raw);
        fault::release();
        stored.unwrap();
        building.join().unwrap()
    });
    fault::clear();
    let (built, _) = report.unwrap();
    assert!(built.errors.is_empty(), "{built:?}");
    assert_eq!(home.texts(&session), ["before the cut"]);

    let caught_up = engine.run_pass().unwrap();
    assert!(caught_up.errors.is_empty(), "{caught_up:?}");
    assert_eq!(
        home.texts(&session),
        ["before the cut", "arrived while rebuilding"]
    );
}

#[test]
fn first_replay_reads_recent_durable_bytes_before_backfill_and_resumes_after_reopen() {
    let home = ClaudeHome::new();
    {
        let (engine, _lock) = crate::import::tests::engine(&home, 4096);
        for (input, seen_at_ms) in [("aaa-old", 10), ("xxx-middle", 20), ("zzz-new", 30)] {
            let path = home.file(&format!("{input}.jsonl"));
            let mut raw = polylogue_store::fixture::writer::step(
                input,
                "claude-code",
                path.to_str().unwrap(),
                Default::default(),
                0,
                0,
            );
            let bytes = claude_user(&format!("u-{input}"), input, input);
            raw.parse.clear();
            raw.observation.as_mut().unwrap().seen_at_ms = seen_at_ms;
            raw.length = bytes.len() as u64;
            engine
                .writer
                .store_bytes(engine.archive.prepare_bytes(bytes.as_bytes()).unwrap(), raw)
                .unwrap();
            assert!(!path.exists(), "only the archive holds the source");
        }
    }

    for session in ["zzz-new", "xxx-middle"] {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        engine.config.workers = 1;
        engine.load().unwrap();
        fault::arm("read.chains_behind_index", "aaa-old", Action::Fail, 0, 1);
        let mut pass = Pass::default();
        let result = engine.reparse_behind(RevisionChains::Newest, &mut pass);
        fault::clear();
        assert!(
            result.is_err(),
            "the ordinary backfill reaches the held page"
        );
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(home.texts(session), [session]);
        assert!(home.texts("aaa-old").is_empty());
    }
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    for session in ["zzz-new", "xxx-middle", "aaa-old"] {
        assert_eq!(home.texts(session), [session]);
    }
}
