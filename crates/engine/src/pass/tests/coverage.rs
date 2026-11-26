use super::*;
use polylogue_model::capture::{Accessibility, Classification};
#[test]
fn status_keeps_the_last_completed_full_walk_across_incremental_events() {
    let home = Home::new();
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
    assert_eq!(
        reader.status().unwrap().capture_coverage,
        vec![first.clone()]
    );
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

/// A tool home holding a symbolic link, a socket and a user project gives one
/// coverage report, with an ignored `**` entry for each reason.
#[test]
fn ignored_files_matching_any_path_are_one_entry_per_reason() {
    let home = Home::new();
    let claude = home.home.join(".claude");
    let project = claude.join("work");
    std::fs::create_dir_all(project.join(".git")).unwrap();
    std::fs::write(project.join("notes.txt"), b"a user file").unwrap();
    std::os::unix::fs::symlink(home.file("elsewhere.jsonl"), claude.join("link")).unwrap();
    let _socket = std::os::unix::net::UnixListener::bind(claude.join("agent.sock")).unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let coverage = engine
        .reads
        .reader()
        .unwrap()
        .capture_coverage(&engine.config.machine)
        .unwrap()
        .unwrap();
    let root = coverage
        .roots
        .iter()
        .find(|r| r.path_bytes == claude.as_os_str().as_encoded_bytes())
        .unwrap();
    let mut reasons: Vec<_> = root
        .entries
        .iter()
        .filter(|e| e.class == Classification::Ignored && e.pattern == "**")
        .map(|e| e.reason.clone().unwrap())
        .collect();
    reasons.sort();
    assert_eq!(
        reasons,
        [
            "special filesystem entry is not read",
            "symbolic link is not followed",
            "user project under tool home",
        ]
    );
}

/// A refused coverage report is the pass's error; the files the walk found
/// are still stored and read.
#[cfg(feature = "faults")]
#[test]
fn a_refused_coverage_report_does_not_stop_reading() {
    use polylogue_store::fault::{self, Action};

    let home = Home::new();
    home.append("kept.jsonl", &claude_user("one", "kept", "kept words"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    fault::clear();
    fault::arm("write.coverage", "test-machine", Action::Fail, 0, u32::MAX);
    let pass = engine.run_pass();
    fault::clear();
    let pass = pass.unwrap();
    assert_eq!(pass.errors.len(), 1, "{pass:?}");
    assert!(
        pass.errors[0]
            .to_string()
            .contains("injected fault at write.coverage (test-machine)"),
        "{pass:?}"
    );
    assert_eq!(home.texts("kept"), ["kept words"]);
}
