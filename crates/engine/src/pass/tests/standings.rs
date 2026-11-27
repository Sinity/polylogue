use super::*;
use polylogue_store::fault::{self, Action};

/// The one claude-code chain's stored state word, failure sentence and the
/// stored length its parser saw.
fn standing(engine: &Engine) -> (String, Option<String>, Option<i64>) {
    let rows = engine
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT p.state,p.failure_detail,p.parsed_to+p.waiting AS seen_to FROM parse_state p
             JOIN input_ref i ON i.pk=p.input_pk WHERE i.origin='claude-code'",
            &[],
        )
        .unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    (
        rows[0]["state"].as_str().unwrap().to_owned(),
        rows[0]["failure_detail"].as_str().map(str::to_owned),
        rows[0]["seen_to"].as_i64(),
    )
}

#[test]
fn a_parser_failure_is_the_chains_failed_parse_state_until_the_bytes_change() {
    let home = ClaudeHome::new();
    home.append(
        "broken-standing.jsonl",
        &claude_user("one", "session", "first body"),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    fault::clear();
    fault::arm("parse", "broken-standing", Action::Panic, 0, u32::MAX);
    let failed = engine.run_pass().unwrap();
    fault::clear();
    assert!(!failed.errors.is_empty(), "{failed:?}");
    let length = std::fs::metadata(home.file("broken-standing.jsonl"))
        .unwrap()
        .len() as i64;
    let (status, reason, at) = standing(&engine);
    assert_eq!(status, "failed");
    assert!(reason.is_some_and(|reason| !reason.is_empty()));
    assert_eq!(at, Some(length));
    // Unchanged bytes are not parsed again: the recorded failure stands.
    let unchanged = engine.run_pass().unwrap();
    assert!(unchanged.errors.is_empty(), "{unchanged:?}");
    assert_eq!(standing(&engine).0, "failed");
    // Growth reads the chain again.
    home.append(
        "broken-standing.jsonl",
        &claude_user("two", "session", "second body"),
    );
    let grown = engine.run_pass().unwrap();
    assert!(grown.errors.is_empty(), "{grown:?}");
    let length = std::fs::metadata(home.file("broken-standing.jsonl"))
        .unwrap()
        .len() as i64;
    let (status, reason, at) = standing(&engine);
    assert_eq!(
        (status.as_str(), reason, at),
        ("parsed", None, Some(length))
    );
}

#[test]
fn a_failed_startup_read_keeps_discovery_work_for_the_next_turn() {
    let home = ClaudeHome::new();
    home.append(
        "after-read-error.jsonl",
        &claude_user("u1", "resumed", "kept discovery"),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.live_every = Duration::from_millis(1);
    let (tx, rx) = crossbeam_channel::unbounded();
    let status = Arc::new(SharedLoopStatus::default());
    fault::clear();
    fault::arm("read.watched_inputs", "test-machine", Action::Fail, 0, 1);
    let result = engine.run(tx, rx, &status, Lifetime::UntilIdle, || {});
    fault::clear();
    let summary = result.unwrap();
    assert!(summary.passes >= 2, "{summary:?}");
    assert!(
        summary.work.errors.iter().any(|error| error
            .to_string()
            .contains("injected fault at read.watched_inputs")),
        "{summary:?}"
    );
    assert!(!summary.work.capture_pending, "{summary:?}");
    assert_eq!(home.texts("resumed"), ["kept discovery"]);
}

fn stored_state(engine: &Engine, path: &Path) -> polylogue_model::chain::ChainState {
    let input = &engine.inputs[path];
    engine
        .reads
        .reader()
        .unwrap()
        .latest_chain(&input.input_id)
        .unwrap()
        .unwrap()
        .state
}

/// A chain that parsed one record and failed at the next.
fn failed_after_a_prefix(home: &ClaudeHome, name: &str) -> (Engine, polylogue_store::ArchiveLock) {
    let first = claude_user("one", "dependency-stop", "first stored body");
    let second = claude_user("two", "dependency-stop", "second stored body");
    home.append(name, &first);
    let (mut engine, lock) = crate::import::tests::engine(home, first.len() as u64);
    let seeded = engine.run_pass().unwrap();
    assert!(seeded.errors.is_empty(), "{seeded:?}");
    let path = home.file(name);
    let prefix = stored_state(&engine, &path);
    assert!(
        prefix.resume().is_some(),
        "the successful prefix keeps what its parser carries"
    );
    home.append(name, &second);
    engine.config.chunk = second.len() as u64;
    fault::arm("parse", "dependency-stop", Action::Panic, 0, u32::MAX);
    let failed = engine.run_pass().unwrap();
    fault::clear();
    assert!(!failed.errors.is_empty(), "{failed:?}");
    let stopped = stored_state(&engine, &path);
    let stored_length = std::fs::metadata(&path).unwrap().len();
    let polylogue_model::chain::ChainState::Failed {
        parsed_to,
        seen_to,
        failure,
        ..
    } = &stopped
    else {
        panic!("the chain failed: {stopped:?}");
    };
    assert_eq!(
        *parsed_to,
        prefix.parsed_to(),
        "the failure follows the parsed prefix"
    );
    assert_eq!(
        *seen_to, stored_length,
        "the failure names every stored byte"
    );
    assert_eq!(
        failure.kind,
        polylogue_model::ChainFailureKind::ParserPanicked
    );
    (engine, lock)
}

#[test]
fn a_recorded_failure_stays_stopped_across_restart_until_the_reading_version_changes() {
    let home = ClaudeHome::new();
    let name = "dependency-stop.jsonl";
    let (engine, lock) = failed_after_a_prefix(&home, name);
    let path = home.file(name);
    let stopped = stored_state(&engine, &path);
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    let revisions = home.count("SELECT count(*) AS n FROM a.revision");
    drop(engine);
    let mut engine = Engine::new(config, archive).unwrap();
    let unchanged = engine.run_pass().unwrap();
    assert!(unchanged.errors.is_empty(), "{unchanged:?}");
    assert_eq!(unchanged.bytes_read, 0, "{unchanged:?}");
    assert_eq!(stored_state(&engine, &path), stopped);
    // A new parser version reads the chain again.
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let retried = engine.run_pass().unwrap();
    assert!(retried.errors.is_empty(), "{retried:?}");
    assert!(retried.bytes_read > 0, "{retried:?}");
    assert_eq!(retried.bytes_stored, 0, "{retried:?}");
    assert_eq!(standing(&engine).0, "parsed");
    assert_eq!(
        home.texts("dependency-stop"),
        ["first stored body", "second stored body"]
    );
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.revision"),
        revisions
    );
    drop(engine);
    drop(lock);
}

#[test]
fn a_new_executable_does_not_reread_failed_chains() {
    let home = ClaudeHome::new();
    let name = "dependency-stop.jsonl";
    let (engine, lock) = failed_after_a_prefix(&home, name);
    let path = home.file(name);
    let stopped = stored_state(&engine, &path);
    let (config, archive) = (engine.config.clone(), engine.archive.clone());
    drop(engine);
    // Nothing of the program that recorded the failure is kept: any other
    // build that starts on the archive reads the same state.
    for _ in 0..2 {
        let mut engine = Engine::new(config.clone(), archive.clone()).unwrap();
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(pass.bytes_read, 0, "{pass:?}");
        assert_eq!(stored_state(&engine, &path), stopped);
    }
    drop(lock);
}
