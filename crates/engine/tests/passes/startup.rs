use super::*;
use polylogue_engine::{LoopMsg, LoopStatus};
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::{Duration, Instant};

#[test]
fn restart_keeps_the_index_and_reads_no_unchanged_input() {
    let home = ClaudeHome::new();
    home.append("s1.jsonl", &user("u1", "s1", "stored before restart"));
    let generation;
    {
        let (mut engine, _lock) = home.engine(64 << 20);
        engine.run_pass().unwrap();
        assert_eq!(
            home.count("SELECT count(*) AS n FROM meta WHERE key='inputs_checked'"),
            1
        );
        generation = ReadPool::new(&engine.archive, 1)
            .unwrap()
            .reader()
            .unwrap()
            .change_position()
            .unwrap()
            .generation;
    }
    let before = home.export("claude-code:s1");
    let (mut engine, _lock) = home.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert_eq!(
        generation,
        ReadPool::new(&engine.archive, 1)
            .unwrap()
            .reader()
            .unwrap()
            .change_position()
            .unwrap()
            .generation
    );
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_read, 0);
    assert_eq!(pass.inputs_changed, 0);
    assert_eq!(pass.sessions_derived, 0);
    assert_eq!(before, home.export("claude-code:s1"));
    // A changed input invalidates the completed revision check.
    home.append("s1.jsonl", &user("u2", "s1", "written after restart"));
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert_eq!(
        home.texts("s1"),
        ["stored before restart", "written after restart"]
    );
}

fn wait_idle(status: &Mutex<LoopStatus>, after: u64) -> LoopStatus {
    let until = Instant::now() + Duration::from_secs(15);
    loop {
        let state = status.lock().unwrap().clone();
        if state.passes > after && !state.busy {
            return state;
        }
        assert!(Instant::now() < until, "loop did not finish: {state:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn repeated_restarts_resume_appends_without_reading_the_history() {
    let home = ClaudeHome::new();
    let original = user("u1", "s1", &"stored history ".repeat(1000));
    home.append("s1.jsonl", &original);
    {
        let (mut engine, _lock) = home.engine(1024);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let mut all = original;
    for n in 2..=4 {
        let append = user(&format!("u{n}"), "s1", &format!("new prompt {n}"));
        home.append("s1.jsonl", &append);
        all.push_str(&append);
        let (mut engine, _lock) = home.engine(1024);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(
            pass.bytes_read,
            append.len() as u64,
            "restart read old bytes: {pass:?}"
        );
    }
    let whole = ClaudeHome::new();
    whole.append("s1.jsonl", &all);
    let (mut engine, _lock) = whole.engine(1024);
    engine.run_pass().unwrap();
    assert_eq!(
        home.export("claude-code:s1"),
        whole.export("claude-code:s1")
    );
}

#[test]
fn a_file_event_visits_only_its_input_and_idle_time_runs_no_pass() {
    let home = ClaudeHome::new();
    for n in 0..20 {
        home.append(
            &format!("s{n}.jsonl"),
            &user("u1", &format!("s{n}"), "first"),
        );
    }
    let (mut engine, lock) = home.engine(64 << 20);
    engine.run_pass().unwrap();
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    let (tx, rx) = channel();
    let worker_tx = tx.clone();
    let shared = Arc::clone(&status);
    let thread = std::thread::spawn(move || {
        let _lock = lock;
        engine.run(worker_tx, &rx, &shared, || {})
    });
    let first = wait_idle(&status, 0);
    // Let queued creation notifications finish before the file under test changes.
    std::thread::sleep(Duration::from_millis(500));
    let passes = status.lock().unwrap().passes;
    home.append("s3.jsonl", &user("u2", "s3", "second"));
    let changed = wait_idle(&status, passes);
    assert!(changed.last_pass.inputs_seen <= 2, "{changed:?}");
    assert_eq!(home.texts("s3"), ["first", "second"]);
    assert!(first.last_pass.errors.is_empty());
    std::thread::sleep(Duration::from_millis(500));
    let passes = status.lock().unwrap().passes;
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(status.lock().unwrap().passes, passes);
    tx.send(LoopMsg::Stop).unwrap();
    thread.join().unwrap().unwrap();
}

#[test]
fn deleted_file_cursors_stay_on_disk_until_a_rebuild_needs_their_bytes() {
    let home = ClaudeHome::new();
    home.append("s1.jsonl", &user("u1", "s1", "kept without its source"));
    {
        let (mut engine, _lock) = home.engine(64 << 20);
        engine.run_pass().unwrap();
        fs::remove_file(home.file("s1.jsonl")).unwrap();
        engine.run_pass().unwrap();
        let reads = ReadPool::new(&engine.archive, 1).unwrap();
        let reader = reads.reader().unwrap();
        assert!(
            reader
                .resume_watched_inputs(&engine.config.machine)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            reader
                .known_watched_inputs(&engine.config.machine)
                .unwrap()
                .len(),
            1
        );
    }
    delete_index(&home.root);
    let (mut engine, _lock) = home.engine(64 << 20);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.bytes_read > 0);
    assert_eq!(home.texts("s1"), ["kept without its source"]);
}
