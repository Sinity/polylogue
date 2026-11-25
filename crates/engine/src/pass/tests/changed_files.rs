use super::*;
use polylogue_store::fault::{self, Action};

/// Runs a pass on its own thread, holds it as it admits the plan of
/// `grows.jsonl`, appends `record` to that file, and lets the pass go on.
fn pass_while_a_file_grows(
    home: &ClaudeHome,
    engine: Engine,
    lock: polylogue_store::ArchiveLock,
    record: &str,
) -> Pass {
    fault::arm("pass.admit", "grows.jsonl", Action::Hold, 0, 1);
    let mut engine = engine;
    let running = std::thread::spawn(move || {
        let result = engine.run_pass();
        drop(engine);
        drop(lock);
        result
    });
    while !fault::held() && !running.is_finished() {
        std::thread::yield_now();
    }
    let held = fault::held();
    home.append("grows.jsonl", record);
    fault::release();
    let pass = running.join().unwrap();
    fault::clear();
    assert!(held, "the pass admitted the planned file");
    pass.expect("a file that changed after it was planned is that file's condition, not the pass's")
}

/// A new archive's first pass stores every file found before its build
/// (docs/installing.md). A session file written to after the storing turn
/// planned it is planned again from what it holds then, and is stored with
/// every other file.
#[test]
fn a_file_growing_in_the_first_build_is_stored_with_every_other_file() {
    fault::clear();
    let home = ClaudeHome::new();
    home.append(
        "before.jsonl",
        &claude_user("b1", "before", "beside it, before"),
    );
    home.append("grows.jsonl", &claude_user("g1", "grows", "first record"));
    home.append(
        "later.jsonl",
        &claude_user("l1", "later", "beside it, after"),
    );
    let (mut engine, lock) = home.engine(1 << 20);
    engine.config.workers = 1;
    assert!(
        engine.archive.index_build().is_some(),
        "the first pass is a build"
    );
    let pass = pass_while_a_file_grows(
        &home,
        engine,
        lock,
        &claude_user("g2", "grows", "written while planned"),
    );
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.texts("before"), ["beside it, before"]);
    assert_eq!(
        home.texts("grows"),
        ["first record", "written while planned"]
    );
    assert_eq!(home.texts("later"), ["beside it, after"]);
}

/// A stored session appended to again after its pass planned the append
/// (a live transcript) is planned again; the pass stores and reads it and
/// every other file.
#[test]
fn a_file_growing_after_its_append_was_planned_is_read_with_every_other_file() {
    fault::clear();
    let home = ClaudeHome::new();
    home.append("steady.jsonl", &claude_user("s1", "steady", "unchanged"));
    home.append("grows.jsonl", &claude_user("g1", "grows", "first record"));
    let (mut engine, lock) = crate::import::tests::engine(&home, 1 << 20);
    engine.config.workers = 1;
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    home.append("grows.jsonl", &claude_user("g2", "grows", "second record"));
    home.append("fresh.jsonl", &claude_user("f1", "fresh", "new beside it"));
    let pass = pass_while_a_file_grows(
        &home,
        engine,
        lock,
        &claude_user("g3", "grows", "written while planned"),
    );
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.texts("steady"), ["unchanged"]);
    assert_eq!(home.texts("fresh"), ["new beside it"]);
    assert_eq!(
        home.texts("grows"),
        ["first record", "second record", "written while planned"]
    );
}
