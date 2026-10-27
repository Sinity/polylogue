use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

#[test]
fn changed_paths_do_not_run_a_full_scan_and_a_rescan_finds_missed_files() {
    let home = ClaudeHome::new();
    home.append("first.jsonl", &claude_user("one", "first", "first"));
    let (mut engine, _lock) = home.engine(64 << 20);
    assert_eq!(engine.run_pass().unwrap().full_scans, 1);
    home.append("changed.jsonl", &claude_user("two", "changed", "changed"));
    home.append("missed.jsonl", &claude_user("three", "missed", "missed"));
    let changed = [home.file("changed.jsonl")].into_iter().collect();
    let pass = engine.one_pass(false, &changed, true).unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.full_scans, 0);
    assert_eq!(home.texts("changed"), ["changed"]);
    assert!(home.texts("missed").is_empty());
    let changes = crate::watch::Changes {
        full: true,
        ..Default::default()
    };
    let pass = engine.one_pass(changes.full, &changes.paths, true).unwrap();
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
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine.config.live_every = Duration::from_secs(3600);
    let out = engine.capture_outputs().pop().unwrap().path;
    let before = std::fs::read(&out).unwrap();
    let changed = [db.clone()].into_iter().collect();
    source.execute("UPDATE kept SET n=1", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed, true).unwrap();
    // Another file changes between the database's writes. Its wake must not
    // be a second quiet look at the database.
    home.append("other.jsonl", &claude_user("one", "other", "other"));
    let other = [home.file("other.jsonl")].into_iter().collect();
    engine.one_pass(false, &other, true).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), before);
    source.execute("UPDATE kept SET n=2", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed, true).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), before);
    // One interval with no database change makes the final value due.
    engine.capture_checked = None;
    let pass = engine.one_pass(false, &Default::default(), true).unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(
        std::fs::read_to_string(out).unwrap().lines().any(|line| {
            serde_json::from_str::<serde_json::Value>(line).unwrap()["r"]["n"] == 2
        })
    );
}
