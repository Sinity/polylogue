use super::*;
use polylogue_store::fault::{self, Action};

/// Runs `pass` on its own thread, holds it as it admits the plan of
/// `grows.jsonl`, appends `record` to that file, and lets the pass go on.
/// The pass's errors, or the error that ended it.
fn pass_while_a_file_grows(
    home: &ClaudeHome,
    record: &str,
    pass: impl FnOnce() -> Result<Vec<polylogue_model::Failure>, String> + Send + 'static,
) -> Vec<polylogue_model::Failure> {
    fault::arm("pass.admit", "grows.jsonl", Action::Hold, 0, 1);
    let running = std::thread::spawn(pass);
    while !fault::held() && !running.is_finished() {
        std::thread::yield_now();
    }
    let held = fault::held();
    home.append("grows.jsonl", record);
    fault::release();
    let errors = running.join().unwrap();
    fault::clear();
    assert!(held, "the pass admitted the planned file");
    errors
        .expect("a file that changed after it was planned is that file's condition, not the pass's")
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
    let errors = pass_while_a_file_grows(
        &home,
        &claude_user("g2", "grows", "written while planned"),
        move || {
            let result = engine.run_pass();
            drop(engine);
            drop(lock);
            result
                .map(|pass| pass.errors)
                .map_err(|error| error.to_string())
        },
    );
    assert!(errors.is_empty(), "{errors:?}");
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
    let errors = pass_while_a_file_grows(
        &home,
        &claude_user("g3", "grows", "written while planned"),
        move || {
            let result = engine.run_pass();
            drop(engine);
            drop(lock);
            result
                .map(|pass| pass.errors)
                .map_err(|error| error.to_string())
        },
    );
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(home.texts("steady"), ["unchanged"]);
    assert_eq!(home.texts("fresh"), ["new beside it"]);
    assert_eq!(
        home.texts("grows"),
        ["first record", "second record", "written while planned"]
    );
}

/// A write concurrent with a planned read invalidates that read. The
/// next pass compares the committed prefix before continuing the chain.
#[test]
fn a_file_appended_to_while_its_range_is_read_is_retried() {
    fault::clear();
    let home = ClaudeHome::new();
    home.append("grows.jsonl", &claude_user("g1", "grows", "first record"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    engine.config.workers = 1;
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let path = home.file("grows.jsonl");
    let committed = std::fs::metadata(&path).unwrap().len();
    home.append("grows.jsonl", &claude_user("g2", "grows", "second record"));
    fault::arm("read.source_range", "grows.jsonl", Action::Hold, 0, 1);
    let pass = std::thread::scope(|scope| {
        let running = scope.spawn(|| engine.run_pass());
        assert!(
            fault::wait_held_within(Duration::from_secs(60)),
            "the pass read the appended range"
        );
        home.append(
            "grows.jsonl",
            &claude_user("g3", "grows", "written while read"),
        );
        fault::release();
        running.join().unwrap()
    });
    fault::clear();
    let pass = pass.unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(pass.capture_pending);
    assert_eq!(home.texts("grows"), ["first record"]);
    assert_eq!(engine.inputs[&path].stored, committed);
    let next = engine.run_pass().unwrap();
    assert!(next.errors.is_empty(), "{:?}", next.errors);
    assert_eq!(
        home.texts("grows"),
        ["first record", "second record", "written while read"]
    );
}
