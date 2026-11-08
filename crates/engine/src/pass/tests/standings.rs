use super::*;
use polylogue_store::fault::{self, Action};

fn standing(engine: &Engine) -> (String, Option<String>, Option<i64>) {
    let rows = engine
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT p.status,p.reason,p.at_length FROM parse_state p
             JOIN input_ref i ON i.pk=p.input_pk WHERE i.format='claude-code'",
            &[],
        )
        .unwrap();
    assert_eq!(rows.len(), 1, "{rows:?}");
    (
        rows[0]["status"].as_str().unwrap().to_owned(),
        rows[0]["reason"].as_str().map(str::to_owned),
        rows[0]["at_length"].as_i64(),
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
    fault::arm("parse", "broken-standing", Action::Fail, 0, u32::MAX);
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
    // Growth reads the chain again from where it stopped.
    home.append(
        "broken-standing.jsonl",
        &claude_user("two", "session", "second body"),
    );
    let grown = engine.run_pass().unwrap();
    assert!(grown.errors.is_empty(), "{grown:?}");
    let (status, reason, at) = standing(&engine);
    assert_eq!((status.as_str(), reason, at), ("done", None, None));
}

#[test]
fn a_failed_startup_read_keeps_discovery_work_for_the_next_turn() {
    let home = ClaudeHome::new();
    home.append(
        "after-read-error.jsonl",
        &claude_user("u1", "resumed", "retained discovery"),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    engine.config.live_every = Duration::from_millis(1);
    let (tx, rx) = channel();
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    fault::clear();
    fault::arm("read.watched_inputs", "test-machine", Action::Fail, 0, 1);
    let result = engine.run(tx, &rx, &status, Lifetime::UntilIdle, || {});
    fault::clear();
    let summary = result.unwrap();
    assert!(summary.passes >= 2, "{summary:?}");
    assert!(
        summary
            .work
            .errors
            .iter()
            .any(|error| error.contains("injected fault at read.watched_inputs")),
        "{summary:?}"
    );
    assert!(!summary.work.capture_pending, "{summary:?}");
    assert_eq!(home.texts("resumed"), ["retained discovery"]);
}

fn stopped_reading(engine: &Engine, path: &Path) -> polylogue_store::read::StoredParserCheckpoint {
    let input = &engine.inputs[path];
    let reader = engine.reads.reader().unwrap();
    let progress = reader.input_progress(&input.input_id).unwrap().unwrap();
    reader
        .parser_checkpoint(
            &input.input_id,
            progress.chain,
            &input.origin.reading_version(),
        )
        .unwrap()
        .unwrap()
}

#[test]
fn a_recorded_failure_stays_stopped_across_restart_until_the_executable_changes() {
    let home = ClaudeHome::new();
    let name = "dependency-stop.jsonl";
    let first = claude_user("one", "dependency-stop", "first retained body");
    let second = claude_user("two", "dependency-stop", "second retained body");
    home.append(name, &(first.clone() + &second));
    let (mut engine, lock) = crate::import::tests::engine(&home, first.len() as u64);
    fault::arm("parse", "dependency-stop", Action::Fail, 1, u32::MAX);
    let failed = engine.run_pass().unwrap();
    fault::clear();
    assert!(!failed.errors.is_empty(), "{failed:?}");
    let path = home.file(name);
    let attempt = engine.stopped_attempt;
    let stopped = stopped_reading(&engine, &path);
    assert!(stopped.progress.stopped());
    assert!(
        stopped.parsed_upto > 0,
        "the failure follows a real parsed prefix"
    );
    assert_eq!(stopped.stopped_attempt, Some(attempt));
    assert!(
        stopped.checkpoint.is_none(),
        "a failure invents no continuation"
    );
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    let revisions = home.count("SELECT count(*) AS n FROM a.revision");
    drop(engine);
    let mut engine = Engine::new(config.clone(), archive.clone(), attempt).unwrap();
    let unchanged = engine.run_pass().unwrap();
    assert!(unchanged.errors.is_empty(), "{unchanged:?}");
    assert_eq!(unchanged.bytes_read, 0, "{unchanged:?}");
    assert_eq!(
        stopped_reading(&engine, &path).stopped_attempt,
        Some(attempt)
    );
    assert_eq!(standing(&engine).0, "failed");
    drop(engine);

    let changed = StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"different actual executable bytes"),
        parser_settings: None,
    };
    let mut engine = Engine::new(config, archive, changed).unwrap();
    let retried = engine.run_pass().unwrap();
    assert!(retried.errors.is_empty(), "{retried:?}");
    assert!(retried.bytes_read > 0, "{retried:?}");
    assert_eq!(retried.bytes_stored, 0, "{retried:?}");
    let completed = stopped_reading(&engine, &path);
    assert!(!completed.progress.stopped());
    assert_eq!(completed.stopped_attempt, None);
    assert_eq!(
        home.texts("dependency-stop"),
        ["first retained body", "second retained body"]
    );
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.revision"),
        revisions
    );
    drop(engine);
    drop(lock);
}

#[test]
fn an_unrecorded_failure_gets_one_real_attempt_then_stops_across_restart() {
    let home = ClaudeHome::new();
    let name = "unrecorded-stop.jsonl";
    home.append(
        name,
        &claude_user("one", "unrecorded-stop", "stored source"),
    );
    let (mut engine, lock) = crate::import::tests::engine(&home, 4096);
    fault::arm("parse", "unrecorded-stop", Action::Fail, 0, u32::MAX);
    let failed = engine.run_pass().unwrap();
    fault::clear();
    assert!(!failed.errors.is_empty(), "{failed:?}");
    let path = home.file(name);
    let state = &engine.inputs[&path];
    let progress = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&state.input_id)
        .unwrap()
        .unwrap();
    // A historical progress payload has the real failure and no attempt metadata.
    let mut step =
        polylogue_store::ParseStep::new(Default::default(), Default::default(), 0..0, false);
    step.progress = polylogue_model::batch::StepProgress::Failed {
        error: "unrecorded synthetic failure".into(),
    };
    step.reading_version = state.origin.reading_version();
    engine
        .writer
        .send(polylogue_store::Batch {
            required_disposition: None,
            input: InputKey {
                id: state.input_id.clone(),
                format: state.origin.name.into(),
                arrival_rank: 0,
            },
            change: Change::Reparse {
                chain: progress.chain,
            },
            observation: None,
            continues: false,
            new_blob: None,
            blobs: vec![],
            length: progress.length,
            parse: vec![step],
            acquisition: None,
            basis: None,
            release: Default::default(),
        })
        .unwrap();
    assert!(engine.writer.flush().unwrap().errors.is_empty());
    assert_eq!(stopped_reading(&engine, &path).stopped_attempt, None);
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    let attempt = engine.stopped_attempt;
    drop(engine);
    let mut engine = Engine::new(config.clone(), archive.clone(), attempt).unwrap();
    fault::arm("parse", "unrecorded-stop", Action::Fail, 0, u32::MAX);
    let retry = engine.run_pass().unwrap();
    fault::clear();
    assert!(!retry.errors.is_empty(), "{retry:?}");
    assert_eq!(retry.bytes_stored, 0, "{retry:?}");
    assert_eq!(
        stopped_reading(&engine, &path).stopped_attempt,
        Some(attempt)
    );
    drop(engine);
    let mut engine = Engine::new(config, archive, attempt).unwrap();
    let settled = engine.run_pass().unwrap();
    assert!(settled.errors.is_empty(), "{settled:?}");
    assert_eq!(settled.bytes_read, 0, "{settled:?}");
    assert_eq!(standing(&engine).0, "failed");
    // Reading-version change also invalidates the stop, even with the same binary.
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let reread = engine.run_pass().unwrap();
    assert!(reread.errors.is_empty(), "{reread:?}");
    assert!(reread.bytes_read > 0, "{reread:?}");
    assert_eq!(stopped_reading(&engine, &path).stopped_attempt, None);
    assert_eq!(standing(&engine).0, "done");
    drop(engine);
    drop(lock);
}
