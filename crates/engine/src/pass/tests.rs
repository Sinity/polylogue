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
