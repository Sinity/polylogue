use super::*;

#[test]
fn ordinary_inputs_are_serviced_between_database_capture_turns() {
    let w = ClaudeHome::new();
    let db = w.home.join(".codex/state_5.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    let source = rusqlite::Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE kept_rows(id INTEGER PRIMARY KEY,n INTEGER);
        WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<3000)
        INSERT INTO kept_rows SELECT n,10 FROM ids;",
        )
        .unwrap();
    w.append("live.jsonl", &user("u1", "live", "before capture"));
    let (mut engine, _lock) = w.engine(64 << 20);
    let first = engine.converge_turn().unwrap();
    assert!(first.capture_pending);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(w.texts("live"), ["before capture"]);
    w.append("live.jsonl", &user("u2", "live", "during capture"));
    let second = engine.converge_turn().unwrap();
    assert!(second.capture_pending);
    assert!(second.errors.is_empty());
    assert_eq!(w.texts("live"), ["before capture", "during capture"]);
    // The active batch is independent of later discovery results.
    fs::remove_file(&db).unwrap();
    let complete = engine.converge().unwrap();
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

#[test]
fn daemon_continues_capture_without_a_watcher_event() {
    use polylogue_engine::{LoopMsg, Pass};
    use std::sync::{Arc, Mutex, mpsc::channel};
    use std::time::{Duration, Instant};
    let w = ClaudeHome::new();
    let db = w.home.join(".codex/state_5.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    rusqlite::Connection::open(db)
        .unwrap()
        .execute_batch(
            "CREATE TABLE kept_rows(id INTEGER PRIMARY KEY);
         WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<4000)
         INSERT INTO kept_rows SELECT n FROM ids;",
        )
        .unwrap();
    let (mut engine, lock) = w.engine(64 << 20);
    let status = Arc::new(Mutex::new(Pass::default()));
    let shared = status.clone();
    let (tx, rx) = channel();
    let (ready, started) = channel();
    let worker_tx = tx.clone();
    let thread = std::thread::spawn(move || {
        let _lock = lock;
        engine.run(worker_tx, &rx, &shared, || ready.send(()).unwrap())
    });
    started.recv_timeout(Duration::from_secs(5)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let complete = loop {
        let pass = status.lock().unwrap().clone();
        if pass.inputs_seen == 1 && !pass.capture_pending {
            break true;
        }
        if !pass.errors.is_empty() || Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    tx.send(LoopMsg::Stop).unwrap();
    thread.join().unwrap().unwrap();
    assert!(complete, "{:?}", status.lock().unwrap());
    assert_eq!(
        w.count("SELECT count(*) AS n FROM input_ref WHERE format='codex-database'"),
        1
    );
}
