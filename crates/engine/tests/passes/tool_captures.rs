use super::*;

/// A database of `rows` rows where Codex keeps one.
fn database(w: &ClaudeHome, rows: u32) -> std::path::PathBuf {
    let db = w.home.join(".codex/state_5.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute_batch(&format!(
            "CREATE TABLE kept_rows(id INTEGER PRIMARY KEY,n INTEGER);
             WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<{rows})
             INSERT INTO kept_rows SELECT n,10 FROM ids;"
        ))
        .unwrap();
    db
}

#[test]
fn a_first_build_captures_a_database_whole_in_its_pass() {
    let w = ClaudeHome::new();
    database(&w, 3000);
    w.append("live.jsonl", &user("u1", "live", "beside the capture"));
    let (mut engine, _lock) = w.engine(64 << 20);
    let first = engine.run_turn().unwrap();
    assert!(!first.capture_pending, "{first:?}");
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(w.texts("live"), ["beside the capture"]);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM input_ref WHERE format='codex-database'"),
        1
    );
}

#[test]
fn ordinary_inputs_are_serviced_between_database_capture_turns() {
    let w = ClaudeHome::new();
    w.append("live.jsonl", &user("u1", "live", "before capture"));
    let (mut engine, _lock) = w.engine(64 << 20);
    // A first build captures whole; later passes capture in turns.
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let db = database(&w, 3000);
    let first = engine.run_turn().unwrap();
    assert!(first.capture_pending);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(w.texts("live"), ["before capture"]);
    w.append("live.jsonl", &user("u2", "live", "during capture"));
    let second = engine.run_turn().unwrap();
    assert!(second.capture_pending);
    assert!(second.errors.is_empty());
    assert_eq!(w.texts("live"), ["before capture", "during capture"]);
    // The active batch is independent of later discovery results.
    fs::remove_file(&db).unwrap();
    let complete = engine.run_pass().unwrap();
    assert!(!complete.capture_pending);
    assert!(complete.errors.is_empty(), "{:?}", complete.errors);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM input_ref WHERE format='codex-database'"),
        1
    );
    let out = fs::read_dir(w.root.join("exports/codex-database"))
        .unwrap()
        .map(|entry| entry.unwrap().path().join("state_5.sqlite.jsonl"))
        .find(|path| path.is_file())
        .unwrap();
    assert_eq!(fs::read_to_string(out).unwrap().lines().count(), 3001);
}

/// With no change reported and no rescan due, only the loop's own
/// continuation can finish a capture larger than one turn; a loop that
/// stopped after the first turn would leave the test waiting.
#[test]
fn daemon_continues_capture_without_a_watcher_event() {
    use polylogue_engine::{LoopMsg, LoopStatus};
    use std::sync::{Arc, Mutex, mpsc::channel};
    use std::time::Duration;
    let w = ClaudeHome::new();
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(64 << 20);
    config.rescan_every = Duration::MAX;
    let mut engine = Engine::new(config, archive).unwrap();
    // A first build captures whole; this one has nothing to capture.
    assert!(engine.run_pass().unwrap().errors.is_empty());
    database(&w, 4000);
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    let shared = status.clone();
    let (tx, rx) = channel();
    let (ready, started) = channel();
    let worker_tx = tx.clone();
    let thread = std::thread::spawn(move || {
        let _lock = lock;
        engine.run(worker_tx, &rx, &shared, || ready.send(()).unwrap())
    });
    started.recv().unwrap();
    let finished = loop {
        let now = status.lock().unwrap().clone();
        if now.passes > 0 && !now.busy {
            break now;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    tx.send(LoopMsg::Stop).unwrap();
    thread.join().unwrap().unwrap();
    assert!(finished.passes >= 1, "{finished:?}");
    assert_eq!(finished.full_scans, 1);
    assert_eq!(finished.last_pass.inputs_seen, 1, "{finished:?}");
    assert!(finished.last_pass.errors.is_empty(), "{finished:?}");
    assert!(!finished.last_pass.capture_pending, "{finished:?}");
    assert_eq!(
        w.count("SELECT count(*) AS n FROM input_ref WHERE format='codex-database'"),
        1
    );
}
