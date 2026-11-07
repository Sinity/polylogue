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
