mod coverage;
mod export_inputs;
mod idle;
mod no_sources;
mod parent_check;
mod readings;
mod returning;
mod scope_lifecycle;
mod tool_captures;
mod unreadable;
use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::io::Read;
use std::os::unix::fs::{FileExt, MetadataExt};

#[cfg(feature = "faults")]
mod build_failures;
#[cfg(feature = "faults")]
mod build_memory;
#[cfg(feature = "faults")]
mod changed_files;
#[cfg(feature = "faults")]
mod live_queue;
#[cfg(feature = "faults")]
mod planning;
#[cfg(feature = "faults")]
mod sampling;
#[cfg(feature = "faults")]
mod standings;

#[test]
fn finished_pass_times_its_work_in_execution_order() {
    let home = ClaudeHome::new();
    home.append("timed.jsonl", &claude_user("u1", "timed", "one message"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.messages_written, 1, "{pass:?}");
    let work: Vec<_> = pass
        .phases
        .iter()
        .map(|(phase, _)| phase.as_str())
        .collect();
    assert_eq!(
        work,
        [
            "preparing the pass",
            "loading source state",
            "collecting source data",
            "finding source files",
            "checking database captures",
            "planning file reads",
            "storing and reading new files",
            "committing changes",
            "computing session figures",
        ]
    );
    assert_eq!(
        pass.seconds,
        pass.phases.iter().map(|(_, seconds)| seconds).sum::<f64>()
    );
}

#[test]
fn extent_checkpoint_survives_a_rewrite_read_failure_before_same_inode_growth() {
    failed_rewrite_then_growth(false);
}

#[test]
fn extent_checkpoint_keeps_tail_detection_after_a_rewrite_read_failure() {
    failed_rewrite_then_growth(true);
}

fn failed_rewrite_then_growth(edit_tail: bool) {
    let home = ClaudeHome::new();
    let old = "original body ".repeat(40);
    home.append("checkpoint.jsonl", &claude_user("one", "session", &old));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    let seeded = engine.run_pass().unwrap();
    assert!(seeded.errors.is_empty(), "initial capture: {seeded:?}");
    let path = home.file("checkpoint.jsonl");
    let selected = engine
        .reads
        .reader()
        .unwrap()
        .latest_chain(&engine.inputs[&path].input_id)
        .unwrap()
        .unwrap()
        .chain;
    let cached_chain = engine.inputs[&path].chain;
    home.append(
        "checkpoint.jsonl",
        &claude_user("two", "session", "second body"),
    );
    let appended = engine.run_pass().unwrap();
    assert!(
        appended.errors.is_empty(),
        "append before rewrite: {appended:?}"
    );
    assert_eq!(
        cached_chain,
        Some(selected),
        "the next append keeps the acknowledged source chain"
    );
    let inode = std::fs::metadata(&path).unwrap().ino();
    // Change a stored byte in place so planning selects a rewrite (a touch
    // that keeps every stored byte is observed back, docs/archive.md), then
    // restore it and make only that planned read fail. Moving the file aside
    // preserves its inode and bytes.
    let first = std::fs::read(&path).unwrap()[0];
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.write_at(b" ", 0).unwrap();
    let now = FileFacts::of(&std::fs::metadata(&path).unwrap());
    let plan = engine.plan(&path, now).unwrap().unwrap();
    assert_eq!(plan.change, Change::Rewritten);
    file.write_at(&[first], 0).unwrap();
    drop(file);
    let aside = home.dir.path().join("held-source");
    std::fs::rename(&path, &aside).unwrap();
    let mut failed = Pass::default();
    engine
        .execute_pool(vec![plan], Vec::new(), &[], &mut failed)
        .unwrap();
    assert_eq!(failed.errors.len(), 1, "{failed:?}");
    assert_eq!(
        failed.errors[0].key(),
        (
            path.display().to_string().as_str(),
            polylogue_model::Cause::Environment,
            polylogue_model::Code::Io
        ),
        "{failed:?}"
    );
    std::fs::rename(&aside, &path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    let second = if edit_tail {
        let source = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, source.replace("second body", "edited body")).unwrap();
        "edited body"
    } else {
        "second body"
    };
    home.append(
        "checkpoint.jsonl",
        &claude_user("three", "session", "third body"),
    );
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        home.texts("session"),
        [old, second.into(), "third body".into()]
    );
    let reader = engine.reads.reader().unwrap();
    let known = reader
        .known_input_at("test-machine", path.as_os_str().as_encoded_bytes())
        .unwrap()
        .unwrap();
    let chain = reader
        .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
        .expect("a failed read must not let the next append overwrite the stored prefix");
    let mut actual = Vec::new();
    chain
        .reader(Arc::clone(&engine.blobs))
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, std::fs::read(&path).unwrap());
    assert_eq!(known.state.parsed_to(), known.stored);
    assert_eq!(
        reader
            .query("SELECT count(DISTINCT chain) AS n FROM a.revision", &[])
            .unwrap()[0]["n"],
        if edit_tail { 2 } else { 1 },
    );
}

// Unit tests need this crate's Engine rather than the testkit's dependency.
fn open_engine(home: &ClaudeHome) -> (Engine, polylogue_store::ArchiveLock) {
    let mut config = crate::test_config();
    config.home = home.home.clone();
    config.tool_dirs.clear();
    config.workers = 3;
    config.machine = "test-machine".into();
    config.host = "test".into();
    let (archive, lock) = Archive::open_or_create(&home.root).unwrap();
    (Engine::new(config, archive).unwrap(), lock)
}

/// An engine whose clock reads `ahead` past the real time the test files get.
fn open_engine_ahead(
    home: &ClaudeHome,
    ahead: jiff::SignedDuration,
) -> (Engine, polylogue_store::ArchiveLock) {
    let mut config = crate::test_config();
    config.home = home.home.clone();
    config.tool_dirs.clear();
    config.machine = "test-machine".into();
    config.host = "test".into();
    config.clock = polylogue_model::WallClock::fixed(jiff::Timestamp::now() + ahead);
    let (archive, lock) = Archive::open_or_create(&home.root).unwrap();
    (Engine::new(config, archive).unwrap(), lock)
}

#[test]
fn finished_inputs_restore_their_continuation_when_they_grow() {
    for ahead in [
        jiff::SignedDuration::from_hours(24),
        jiff::SignedDuration::from_mins(1),
    ] {
        let home = ClaudeHome::new();
        home.append("growing.jsonl", &claude_user("one", "growing", "first"));
        let (mut engine, _lock) = open_engine_ahead(&home, ahead);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        home.append("growing.jsonl", &claude_user("two", "growing", "second"));
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        assert_eq!(home.texts("growing"), ["first", "second"]);
    }
}

#[test]
fn changed_paths_do_not_run_a_full_scan_and_a_rescan_finds_missed_files() {
    let home = ClaudeHome::new();
    home.append("first.jsonl", &claude_user("one", "first", "first"));
    let (mut engine, _lock) = open_engine(&home);
    assert_eq!(engine.run_pass().unwrap().full_scans, 1);
    home.append("changed.jsonl", &claude_user("two", "changed", "changed"));
    home.append("missed.jsonl", &claude_user("three", "missed", "missed"));
    let changed = [home.file("changed.jsonl")].into_iter().collect();
    let (pass, _) = engine.one_pass(false, &changed).unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.full_scans, 0);
    assert_eq!(home.texts("changed"), ["changed"]);
    assert!(home.texts("missed").is_empty());
    let changes = crate::watch::Changes {
        full: true,
        ..Default::default()
    };
    let (pass, _) = engine.one_pass(changes.full, &changes.paths).unwrap();
    assert_eq!(pass.full_scans, 1);
    assert_eq!(home.texts("missed"), ["missed"]);
}

#[test]
fn unrelated_file_events_do_not_make_a_busy_database_appear_quiet() {
    let home = ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let source = rusqlite::Connection::open(&db).unwrap();
    source.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE kept(id INTEGER PRIMARY KEY,n INTEGER); INSERT INTO kept VALUES(1,0);").unwrap();
    let (mut engine, _lock) = open_engine(&home);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine.config.live_every = Duration::from_secs(3600);
    let length = |engine: &Engine| {
        engine
            .reads
            .reader()
            .unwrap()
            .latest_chain(&engine.location_id(&db))
            .unwrap()
            .unwrap()
            .stored
    };
    let before = length(&engine);
    let changed = [db.clone()].into_iter().collect();
    source.execute("UPDATE kept SET n=1", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed).unwrap();
    engine.captures.completed(true);
    home.append("other.jsonl", &claude_user("one", "other", "other"));
    let other = [home.file("other.jsonl")].into_iter().collect();
    engine.one_pass(false, &other).unwrap();
    assert_eq!(length(&engine), before);
    source.execute("UPDATE kept SET n=2", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed).unwrap();
    engine.captures.completed(true);
    assert_eq!(length(&engine), before);
    engine.capture_checked = None;
    engine.one_pass(false, &Default::default()).unwrap();
    let completed = engine.captures.completed(true);
    assert!(completed.errors.is_empty(), "{:?}", completed.errors);
    assert!(length(&engine) > before);
}

mod capture_only;
mod live_inputs;

#[test]
fn idle_loop_turns_do_not_repeat_the_startup_discovery_walk() {
    let home = ClaudeHome::new();
    let (mut engine, _lock) = open_engine(&home);
    engine.config.live_every = Duration::from_millis(1);
    let status = Arc::new(SharedLoopStatus::default());
    let shared = Arc::clone(&status);
    let (tx, rx) = crossbeam_channel::unbounded();
    let worker_tx = tx.clone();
    let worker = std::thread::spawn(move || {
        engine.run(worker_tx, rx, &shared, Lifetime::UntilStopped, || {})
    });
    let observed = status.wait_for(Instant::now() + Duration::from_secs(10), |now| {
        now.passes >= 3
    });
    tx.send(LoopMsg::Stop).unwrap();
    let summary = worker.join().unwrap().unwrap();
    assert!(observed.is_some(), "{:?}", status.get());
    assert_eq!(summary.work.full_scans, 1);
    assert!(summary.work.errors.is_empty(), "{summary:?}");
}

#[test]
fn first_byte_read_failure_retries_after_restart() {
    let home = ClaudeHome::new();
    home.append(
        "first-byte.jsonl",
        &claude_user("one", "first-byte", "read after restart"),
    );
    let (mut engine, lock) = open_engine(&home);
    engine.load().unwrap();
    let path = home.file("first-byte.jsonl");
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    engine.track(&collect::Found {
        path: path.clone(),
        origin,
    });
    let now = FileFacts::of(&std::fs::metadata(&path).unwrap());
    let plan = engine.plan(&path, now).unwrap().unwrap();
    let held = home.dir.path().join("held-original");
    std::fs::rename(&path, &held).unwrap();
    let mut failed = Pass::default();
    let executed = engine.execute_pool(vec![plan], Vec::new(), &[], &mut failed);
    std::fs::rename(&held, &path).unwrap();
    executed.unwrap();
    assert!(!failed.errors.is_empty(), "{failed:?}");
    assert_eq!(failed.bytes_stored, 0, "{failed:?}");
    assert_eq!(home.count("SELECT count(*) AS n FROM parse_state"), 0);
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    drop(engine);
    let mut engine = Engine::new(config, archive).unwrap();
    let retried = engine.run_pass().unwrap();
    assert!(retried.errors.is_empty(), "{retried:?}");
    assert!(retried.bytes_stored > 0, "{retried:?}");
    assert_eq!(home.texts("first-byte"), ["read after restart"]);
    drop(engine);
    drop(lock);
}

#[test]
fn unparseable_stored_input_refused_and_pass_continues() {
    let home = ClaudeHome::new();
    let outside = home.dir.path().join("export");
    std::fs::create_dir_all(&outside).unwrap();
    let kept = outside.join("kept.jsonl");
    let odd = outside.join("odd.jsonl");
    std::fs::write(&kept, claude_user("k1", "kept", "still indexed")).unwrap();
    std::fs::write(&odd, claude_user("o1", "odd", "refused")).unwrap();
    let options = crate::ImportOptions {
        origin: Some("claude-code".into()),
        ..Default::default()
    };
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
        let report = engine
            .import_with_options(&[kept, odd.clone()], &options)
            .unwrap();
        assert!(report.pass.errors.is_empty(), "{report:?}");
    }
    // An older archive recorded this import at a path no parser reads.
    let unreadable = outside.join("odd.txt");
    let archive = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
    let moved = archive
        .execute(
            "UPDATE observation SET path_bytes = CAST(?1 AS BLOB) WHERE path_bytes = CAST(?2 AS BLOB)",
            [unreadable.to_str().unwrap(), odd.to_str().unwrap()],
        )
        .unwrap();
    assert_eq!(moved, 1);
    drop(archive);
    polylogue_testkit::delete_index(&home.root);

    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    let pass = engine.run_pass().unwrap();
    assert_eq!(home.texts("kept"), ["still indexed"]);
    assert!(home.texts("odd").is_empty());
    let reason = format!(
        "no claude-code parser reads the recorded path {}",
        unreadable.display()
    );
    assert_eq!(pass.errors.len(), 1, "{:?}", pass.errors);
    assert!(
        pass.errors[0].to_string().ends_with(&reason),
        "{:?}",
        pass.errors
    );
    let refused = home
        .reads()
        .reader()
        .unwrap()
        .query(
            "SELECT failure_detail AS reason FROM parse_state WHERE failure_kind='refused'",
            &[],
        )
        .unwrap();
    assert_eq!(refused.len(), 1, "{refused:?}");
    assert_eq!(refused[0]["reason"], reason.as_str());
    let again = engine.run_pass().unwrap();
    assert!(
        again.errors.is_empty(),
        "a recorded refusal finishes: {again:?}"
    );
}

/// A watched input whose recorded path its origin's parser does not read (an
/// earlier build captured a file this build's declaration no longer lists)
/// is refused with its reason at its stored length when the index is built
/// again; it is not read range by range with nothing recorded, and the
/// refusal finishes.
#[test]
fn unparsed_watched_input_is_refused_and_finishes() {
    let home = ClaudeHome::new();
    home.append("kept.jsonl", &claude_user("k1", "kept", "still indexed"));
    let odd_text = claude_user("o1", "odd", "refused");
    home.append("odd.jsonl", &odd_text);
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let odd = home.file("odd.jsonl");
    let unreadable = odd.with_extension("txt");
    let archive = rusqlite::Connection::open(home.root.join("archive.db")).unwrap();
    let moved = archive
        .execute(
            "UPDATE observation SET path_bytes = CAST(?1 AS BLOB) WHERE path_bytes = CAST(?2 AS BLOB)",
            [unreadable.to_str().unwrap(), odd.to_str().unwrap()],
        )
        .unwrap();
    assert!(moved > 0);
    drop(archive);
    polylogue_testkit::delete_index(&home.root);

    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    engine.config.reads_sources = false;
    let pass = engine.run_pass().unwrap();
    assert_eq!(home.texts("kept"), ["still indexed"]);
    assert!(home.texts("odd").is_empty());
    let reason = format!(
        "no claude-code parser reads the recorded path {}",
        unreadable.display()
    );
    assert_eq!(pass.errors.len(), 1, "{:?}", pass.errors);
    assert!(
        pass.errors[0].to_string().ends_with(&reason),
        "{:?}",
        pass.errors
    );
    let refused = home
        .reads()
        .reader()
        .unwrap()
        .query(
            "SELECT failure_detail AS reason, (parsed_to+waiting) AS at_length FROM parse_state WHERE failure_kind='refused'",
            &[],
        )
        .unwrap();
    assert_eq!(
        refused,
        [serde_json::json!({"reason": reason, "at_length": odd_text.len()})]
    );
    let again = engine.run_pass().unwrap();
    assert!(
        again.errors.is_empty(),
        "a recorded refusal finishes: {again:?}"
    );
    assert_eq!(again.bytes_read, 0);
}

#[test]
fn a_stopped_engine_reports_cancelled_derivation() {
    let home = ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    engine.archive.stop().cancel();
    assert!(matches!(engine.derive(), Err(EngineError::Stopped)));
}

/// An append at an unchanged location writes no observation; the file's
/// facts at that read are kept with its revision, so a restarted engine
/// finds the grown file unchanged by a look and reads none of its bytes.
#[test]
fn a_restart_reads_no_bytes_of_files_appended_at_their_location() {
    let home = ClaudeHome::new();
    home.append("grown.jsonl", &claude_user("one", "grown", "first record"));
    {
        let (mut engine, _lock) = open_engine(&home);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        home.append("grown.jsonl", &claude_user("two", "grown", "second record"));
        assert!(engine.run_pass().unwrap().errors.is_empty());
        home.append(
            "grown.jsonl",
            &claude_user("three", "grown", "third record"),
        );
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    assert_eq!(home.count("SELECT count(*) AS n FROM a.observation"), 1);
    let (mut engine, _lock) = open_engine(&home);
    let again = engine.run_pass().unwrap();
    assert!(again.errors.is_empty(), "{:?}", again.errors);
    assert_eq!(again.bytes_read, 0);
}
