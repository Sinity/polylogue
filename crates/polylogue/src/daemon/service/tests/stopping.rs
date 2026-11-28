//! Stopping the daemon while its loop is in the middle of a pass.

use polylogue_model::SessionId;
use polylogue_store::fault::{self, Action};
use std::time::{Duration, Instant};

/// Six sessions written together reach one pass, whose one parse worker
/// takes them largest first. A stop before the second provider call discards
/// its private reading, keeps its acquired bytes and starts no queued work.
#[test]
fn stop_during_a_pass_keeps_committed_rows() {
    let mut daemon = crate::test_daemon::start_with_workers("before the pass", 1);
    let project = daemon.home.join(".claude/projects/-p");
    let names = ["stop-1", "stop-2", "stop-3", "stop-4", "stop-5", "stop-6"];
    fault::clear();
    fault::arm("parse", "stop-2", Action::Hold, 0, 1);
    for (i, name) in names.iter().enumerate() {
        // Each later session is shorter, so the worker takes them in order.
        let text = format!("{name} ").repeat(60 - 5 * i);
        let records: String = (0..3)
            .map(|n| polylogue_testkit::claude_user(&format!("{name}-{n}"), name, &text))
            .collect();
        std::fs::write(project.join(format!("{name}.jsonl")), records).unwrap();
    }
    assert!(
        fault::wait_held_within(std::time::Duration::from_secs(30)),
        "no thread reached the held point"
    );
    // The held parser returns once the stop is made; the stop does not wait
    // for the queued sessions.
    let _release = daemon.daemon.archive.stop().on_cancel(fault::release);
    daemon.stop().unwrap();
    fault::clear();

    let (archive, _lock) =
        polylogue_store::Archive::open_or_create(&daemon._dir.path().join("archive")).unwrap();
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    for (i, name) in names.iter().enumerate() {
        let session = SessionId::parse(&format!("claude-code:{name}")).unwrap();
        let messages = reader.messages(&session, false).unwrap();
        let expected = if i == 0 { 3 } else { 0 };
        assert_eq!(
            messages.len(),
            expected,
            "{name}: only the earlier committed parse is shown"
        );
    }
    let held_path = project.join("stop-2.jsonl");
    let held = reader
        .known_input_at(
            "synthetic-machine",
            held_path.as_os_str().as_encoded_bytes(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        held.state,
        polylogue_model::chain::ChainState::Unparsed,
        "private parsing did not advance"
    );
    let original = std::fs::read(&held_path).unwrap();
    assert_eq!(held.stored, original.len() as u64);
    let source = reader
        .stored_chain(&held.input_id, held.chain.unwrap(), held.stored)
        .unwrap();
    assert_eq!(
        source
            .read_range(&archive.blobs().unwrap(), 0, held.stored)
            .unwrap(),
        original,
        "the held parser's read bytes stay stored unchanged"
    );
    drop(reader);
    drop(reads);
    drop(_lock);
    let mut resumed = crate::test_daemon::start_runtime(
        &daemon._dir.path().join("archive"),
        daemon.config.clone(),
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let until = Instant::now() + Duration::from_secs(30);
    loop {
        let reader = resumed.daemon.reads.reader().unwrap();
        let all_complete = names.iter().enumerate().all(|(i, name)| {
            let session = SessionId::parse(&format!("claude-code:{name}")).unwrap();
            let messages = reader.messages(&session, false).unwrap();
            let text = format!("{name} ").repeat(60 - 5 * i);
            messages.len() == 3 && messages.iter().all(|message| message.text() == text)
        });
        if all_complete {
            break;
        }
        assert!(
            Instant::now() < until,
            "restart did not complete each source exactly once"
        );
        std::thread::yield_now();
    }
    resumed.shutdown().unwrap();
}

#[test]
fn stop_during_index_rebuild_keeps_stored_bytes() {
    for point in ["index.build.prepare", "index.deferred"] {
        stop_at_rebuild_stage(point);
    }
}

/// The daemon can read a committed part of a populated archive while
/// other recorded chains are due, and retains that part across a stop.
#[test]
fn missing_index_serves_committed_rows_during_replay_and_resumes_them() {
    use crate::test_daemon;
    use polylogue_model::freshness::CompletenessReason;
    use serde_json::json;

    let mut original = test_daemon::start_with_sources_and_workers(
        "first needle",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/second.jsonl"),
                polylogue_testkit::claude_user("second-message", "second", "second needle"),
            )
            .unwrap();
        },
        1,
        1,
    );
    original.stop().unwrap();
    let root = original._dir.path().join("archive");
    polylogue_testkit::delete_index(&root);
    let mut config = original.config.clone();
    config.reads_sources = false;

    fault::clear();
    fault::arm("parse", "", Action::Hold, 1, 1);
    let mut runtime = crate::daemon::service::start(
        crate::config::Config::for_test(&config, &root),
        config.clone(),
        crate::daemon::service::ArchiveMode::Standalone,
        polylogue_engine::Lifetime::UntilStopped,
        &Default::default(),
    )
    .unwrap();
    let _release = runtime.daemon.archive.stop().on_cancel(fault::release);
    assert!(fault::wait_held_within(Duration::from_secs(30)));
    let ids = ["claude-code:s1", "claude-code:second"].map(|id| SessionId::parse(id).unwrap());
    let deadline = Instant::now() + Duration::from_secs(30);
    let (instance, visible) = loop {
        let reader = runtime.daemon.reads.reader().unwrap();
        let visible: Vec<_> = ids
            .iter()
            .filter(|id| !reader.messages(id, false).unwrap().is_empty())
            .collect();
        if let [visible] = visible.as_slice() {
            let freshness = runtime.daemon.freshness(&reader).unwrap();
            assert!(
                freshness.local.parsing.reasons().iter().any(|reason| {
                    reason.reason == CompletenessReason::MissingCheckpoint
                        && reason.origin.as_deref() == Some("claude-code")
                        && reason.count == Some(1)
                }),
                "{freshness:?}"
            );
            drop(reader);
            let reply = test_daemon::call(
                &runtime.daemon,
                "rows",
                json!({"selection": "sessions where needle since:2026-03-01"}),
            );
            if let Some(data) = reply.value()
                && data["reply"]["rows"]
                    .as_array()
                    .is_some_and(|rows| rows.len() == 1)
            {
                assert_eq!(reply.exit_code(), 5, "{reply:?}");
                let reader = runtime.daemon.reads.reader().unwrap();
                break (reader.change_position().unwrap().index_instance, *visible);
            }
        }
        assert!(
            Instant::now() < deadline,
            "committed rows did not appear in the date-filtered search: {visible:?}"
        );
        std::thread::yield_now();
    };
    runtime.shutdown().unwrap();
    fault::clear();
    let mut resumed = test_daemon::start_runtime(
        &root,
        config,
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let reader = resumed.daemon.reads.reader().unwrap();
        assert_eq!(reader.change_position().unwrap().index_instance, instance);
        assert_eq!(reader.messages(visible, false).unwrap().len(), 1);
        if ids
            .iter()
            .all(|id| reader.messages(id, false).unwrap().len() == 1)
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the remaining reading did not resume"
        );
        std::thread::yield_now();
    }
    resumed.shutdown().unwrap();
}

fn stop_at_rebuild_stage(point: &str) {
    let records: String = (0..48)
        .map(|n| {
            polylogue_testkit::claude_user(&format!("rebuild-{n}"), "rebuild", "stored source")
        })
        .collect();
    let mut original = crate::test_daemon::start_with_sources_and_workers(
        "already captured",
        &|home| {
            std::fs::write(home.join(".claude/projects/-p/rebuild.jsonl"), &records).unwrap();
        },
        48,
        1,
    );
    original.stop().unwrap();
    let root = original._dir.path().join("archive");
    // An extra read index keeps this file usable and makes the daemon
    // rebuild it privately. With no usable index the daemon catches up
    // incrementally instead.
    polylogue_store::archive::open_write(&root.join("index.db"), 8)
        .unwrap()
        .execute_batch("CREATE INDEX test_rebuild_extra ON message(pk)")
        .unwrap();
    let mut config = original.config.clone();
    config.reads_sources = false;
    fault::clear();
    fault::arm(point, "", Action::Hold, 0, 1);
    let mut runtime = crate::daemon::service::start(
        crate::config::Config::for_test(&config, &root),
        config.clone(),
        crate::daemon::service::ArchiveMode::Standalone,
        polylogue_engine::Lifetime::UntilIdle,
        &Default::default(),
    )
    .unwrap();
    assert!(
        fault::wait_held_within(std::time::Duration::from_secs(30)),
        "no thread reached {point}"
    );
    let mut idle = runtime.idle();
    runtime.stop_accepting();
    fault::release();
    let summary = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(idle.wait_for(Option::is_some))
        .unwrap()
        .clone()
        .unwrap();
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    runtime.shutdown().unwrap();
    fault::clear();
    let mut runtime = crate::test_daemon::start_runtime(
        &root,
        config,
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let started = Instant::now();
    loop {
        let reader = runtime.daemon.reads.reader().unwrap();
        let messages = reader
            .messages(&SessionId::parse("claude-code:rebuild").unwrap(), false)
            .unwrap();
        drop(reader);
        if runtime.daemon.archive.index_build().is_none() && messages.len() == 48 {
            assert!(
                messages
                    .iter()
                    .all(|message| message.text() == "stored source")
            );
            break;
        }
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "restart did not replay stored bytes"
        );
        std::thread::yield_now();
    }
    runtime.shutdown().unwrap();
}

fn due_derivations(root: &std::path::Path) -> i64 {
    rusqlite::Connection::open(root.join("index.db"))
        .unwrap()
        .query_row("SELECT count(*) FROM summary_due", [], |row| row.get(0))
        .unwrap()
}

/// A stop while the derive thread holds a computed wave drops that wave:
/// the session stays due, and the next start derives it.
#[test]
fn a_stop_during_derivation_leaves_the_session_due_and_restart_derives_it() {
    let mut daemon = crate::test_daemon::start_with_workers("before derivation", 1);
    let root = daemon._dir.path().join("archive");
    let pause = daemon
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .pause_derive_for_test();
    std::fs::write(
        daemon.home.join(".claude/projects/-p/derive.jsonl"),
        polylogue_testkit::claude_user("d1", "derive", "a session to derive"),
    )
    .unwrap();
    while pause.wait().is_err() {}
    let pause = std::sync::Mutex::new(Some(pause));
    let _release = daemon.daemon.archive.stop().on_cancel(move || {
        if let Some(pause) = pause.lock().unwrap().take() {
            pause.release();
        }
    });
    daemon.stop().unwrap();
    assert_eq!(due_derivations(&root), 1, "the dropped wave stays due");

    let mut resumed = crate::test_daemon::start_runtime(
        &root,
        daemon.config.clone(),
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let until = Instant::now() + Duration::from_secs(30);
    while due_derivations(&root) != 0 {
        assert!(Instant::now() < until, "restart did not derive the session");
        std::thread::sleep(Duration::from_millis(10));
    }
    resumed.shutdown().unwrap();
}

/// A stop while an import parses a stored export ends it between items:
/// the import is not reported done, no later item is parsed, and importing
/// it again after a restart finishes the export, each item once.
#[test]
fn a_stop_during_an_import_is_not_reported_done_and_importing_again_finishes_it() {
    let mut daemon = crate::test_daemon::start_with_workers("before the import", 1);
    let root = daemon._dir.path().join("archive");
    let export = daemon._dir.path().join("conversations.json");
    let conversations: Vec<_> = (0..3)
        .map(|n| {
            serde_json::json!({
                "uuid": format!("imported-{n}"),
                "chat_messages": [{
                    "uuid": format!("imported-message-{n}"),
                    "sender": "human",
                    "text": format!("imported words {n}"),
                }],
            })
        })
        .collect();
    std::fs::write(&export, serde_json::to_vec(&conversations).unwrap()).unwrap();
    fault::clear();
    // The second item holds until the stop is made.
    fault::arm("parse", "conversations.json", Action::Hold, 1, 1);
    let _release = daemon.daemon.archive.stop().on_cancel(fault::release);
    let mut client =
        std::os::unix::net::UnixStream::connect(daemon.daemon.archive.socket()).unwrap();
    let request = command_surface::Request {
        operation: "import".into(),
        arguments: serde_json::json!({"paths": [export]}),
        request_id: "import-during-stop".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    use std::io::Write;
    writeln!(client, "{}", serde_json::to_string(&request).unwrap()).unwrap();
    assert!(
        fault::wait_held_within(std::time::Duration::from_secs(30)),
        "no thread reached the held point"
    );
    daemon.stop().unwrap();
    fault::clear();
    let answer = crate::client::receive(&mut client, None, |_| Ok(false))
        .unwrap()
        .expect("the import answers");
    assert!(
        answer.value().is_none(),
        "a stopped import is not reported done: {answer:?}"
    );

    let session = |n: usize| SessionId::parse(&format!("claude-ai:imported-{n}")).unwrap();
    {
        let (archive, _lock) = polylogue_store::Archive::open_or_create(&root).unwrap();
        let reads = polylogue_store::ReadPool::open(&archive).unwrap();
        let reader = reads.reader().unwrap();
        assert!(
            reader.messages(&session(2), false).unwrap().is_empty(),
            "no item after the stop was parsed"
        );
    }
    let mut resumed = crate::test_daemon::start_runtime(
        &root,
        daemon.config.clone(),
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    crate::test_answers::ok(
        crate::test_daemon::Call::new("import", serde_json::json!({"paths": [export]}))
            .answer(&resumed.daemon),
    );
    let reader = resumed.daemon.reads.reader().unwrap();
    let texts: Vec<String> = (0..3)
        .flat_map(|n| reader.messages(&session(n), false).unwrap())
        .map(|message| message.text())
        .collect();
    assert_eq!(
        texts,
        ["imported words 0", "imported words 1", "imported words 2"]
    );
    drop(reader);
    resumed.shutdown().unwrap();
}

#[test]
fn status_reports_a_source_walk_and_the_finished_pass() {
    use crate::test_daemon;
    use serde_json::json;

    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let source = home.join(".claude");
    let project = source.join("projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    for n in 0..50 {
        let id = format!("pass-source-{n}");
        std::fs::write(
            project.join(format!("{id}.jsonl")),
            polylogue_testkit::claude_user(&id, &id, "one message"),
        )
        .unwrap();
    }
    let archive = dir.path().join("archive");
    let config = test_daemon::engine_config(home);
    fault::arm(
        "source.walk",
        &source.to_string_lossy(),
        Action::Hold,
        49,
        1,
    );
    let mut runtime = crate::daemon::service::start(
        crate::config::Config::for_test(&config, &archive),
        config,
        crate::daemon::service::ArchiveMode::Standalone,
        polylogue_engine::Lifetime::UntilIdle,
        &Default::default(),
    )
    .unwrap();
    let release_on_stop = runtime.daemon.archive.stop().on_cancel(fault::release);
    assert!(
        fault::wait_held_within(Duration::from_secs(30)),
        "the walk did not reach the last file"
    );
    let read_status = || {
        let answer = test_daemon::call(&runtime.daemon, "status", json!({}));
        answer
            .value()
            .unwrap_or_else(|| panic!("status: {answer:?}"))
            .clone()
    };
    let held = read_status();
    let partial = test_daemon::call(&runtime.daemon, "rows", json!({"selection": "sessions"}));
    fault::release();
    assert_eq!(partial.exit_code(), 5, "{partial:?}");
    assert_eq!(
        partial.value().unwrap()["reply"]["read_position"]["freshness"]["local"]["parsing"],
        json!({"state": "incomplete", "reasons": [
            {"reason": "capture_pending", "waiting_ends": true}
        ]})
    );
    let threads: Vec<polylogue_store::activity::Thread> =
        serde_json::from_value(held["threads"].clone()).unwrap();
    let walking = threads
        .iter()
        .find(|thread| {
            thread
                .doing
                .first()
                .is_some_and(|work| work.what == "running a pass")
        })
        .expect("the pass is still running");
    let work: Vec<_> = walking
        .doing
        .iter()
        .map(|work| work.what.as_str())
        .collect();
    assert_eq!(
        work,
        [
            "running a pass",
            "finding source files",
            "walking source files"
        ]
    );
    let root = walking.doing.last().unwrap();
    assert_eq!(root.subject.as_deref(), source.to_str());
    assert!(walking.seconds_since_progress.unwrap() < root.seconds);

    let mut idle = runtime.idle();
    let summary = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(idle.wait_for(Option::is_some))
        .unwrap()
        .clone()
        .unwrap();
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    let finished = read_status();
    let pass: polylogue_engine::Pass =
        serde_json::from_value(finished["last_pass"].clone()).unwrap();
    assert_eq!(pass.inputs_seen, 50, "{pass:?}");
    let reader = runtime.daemon.reads.reader().unwrap();
    for n in 0..50 {
        let session = SessionId::parse(&format!("claude-code:pass-source-{n}")).unwrap();
        assert_eq!(
            reader.messages(&session, false).unwrap().len(),
            1,
            "{session}"
        );
    }
    drop(reader);
    assert!(
        pass.phases
            .iter()
            .any(|(phase, _)| phase == "finding source files")
    );
    assert_eq!(
        pass.seconds,
        pass.phases.iter().map(|(_, seconds)| seconds).sum::<f64>()
    );
    drop(release_on_stop);
    runtime.shutdown().unwrap();
    fault::clear();
}
