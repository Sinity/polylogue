use super::*;
use polylogue_model::capture::{Accessibility, Classification};
#[test]
fn status_keeps_the_last_completed_full_walk_across_incremental_events() {
    let home = ClaudeHome::new();
    let root = home.home.join(".codex");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(home.home.join(".grok")).unwrap();
    std::fs::write(root.join("unrecognized.dat"), b"first").unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let first = engine
        .reads
        .reader()
        .unwrap()
        .capture_coverage(&engine.config.machine)
        .unwrap()
        .unwrap();
    first.validate().unwrap();
    let codex = first
        .roots
        .iter()
        .find(|r| r.path_bytes == root.as_os_str().as_encoded_bytes())
        .unwrap();
    assert_eq!(
        codex
            .entries
            .iter()
            .filter(|e| e.class == Classification::Unrecognized)
            .map(|e| e.files)
            .sum::<u64>(),
        1
    );
    let grok = first
        .roots
        .iter()
        .find(|r| r.path_bytes == home.home.join(".grok").as_os_str().as_encoded_bytes())
        .unwrap();
    assert_eq!(grok.status, Accessibility::Accessible);
    assert!(grok.entries.is_empty());
    let extra = root.join("another.dat");
    std::fs::write(&extra, b"second").unwrap();
    engine
        .one_pass(false, &std::collections::BTreeSet::from([extra]))
        .unwrap();
    let reader = engine.reads.reader().unwrap();
    assert_eq!(
        reader
            .capture_coverage(&engine.config.machine)
            .unwrap()
            .unwrap(),
        first
    );
    assert!(reader.status().unwrap().capture_coverage.contains(&first));
    drop(reader);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let next = engine
        .reads
        .reader()
        .unwrap()
        .capture_coverage(&engine.config.machine)
        .unwrap()
        .unwrap();
    let codex = next
        .roots
        .iter()
        .find(|r| r.path_bytes == root.as_os_str().as_encoded_bytes())
        .unwrap();
    assert_eq!(
        codex
            .entries
            .iter()
            .filter(|e| e.class == Classification::Unrecognized)
            .map(|e| e.files)
            .sum::<u64>(),
        2
    );
}
