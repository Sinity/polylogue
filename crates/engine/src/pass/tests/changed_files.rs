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

/// A live transcript that grows while its range is being read stores the
/// planned range after comparing the committed prefix with the newer file.
#[test]
fn a_file_appended_to_while_its_range_is_read_stores_that_range() {
    fault::clear();
    let home = ClaudeHome::new();
    home.append("grows.jsonl", &claude_user("g1", "grows", "first record"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    engine.config.workers = 1;
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let path = home.file("grows.jsonl");
    home.append("grows.jsonl", &claude_user("g2", "grows", "second record"));
    let planned = std::fs::metadata(&path).unwrap().len();
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
    assert_eq!(home.texts("grows"), ["first record", "second record"]);
    assert_eq!(engine.inputs[&path].stored, planned);
    let next = engine.run_pass().unwrap();
    assert!(next.errors.is_empty(), "{:?}", next.errors);
    assert_eq!(
        home.texts("grows"),
        ["first record", "second record", "written while read"]
    );
}

/// A growing file that also edited the already captured prefix cannot
/// continue the old chain from the concurrent range.
#[test]
fn early_edit_and_growth_during_a_read_leave_the_committed_chain_intact() {
    fault::clear();
    let home = ClaudeHome::new();
    let path = home.file("edited-during-read.jsonl");
    let first = claude_user("u1", "edited", "old word")
        + &claude_user("u2", "edited", &"same suffix ".repeat(450));
    home.append("edited-during-read.jsonl", &first);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    engine.config.workers = 1;
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let later = claude_user("u3", "edited", "planned append");
    home.append("edited-during-read.jsonl", &later);
    let replacement = first.replacen("old word", "new word", 1)
        + &later
        + &claude_user("u4", "edited", "new suffix");
    fault::arm(
        "read.source_range",
        "edited-during-read.jsonl",
        Action::Hold,
        0,
        1,
    );
    let pass = std::thread::scope(|scope| {
        let running = scope.spawn(|| engine.run_pass());
        assert!(fault::wait_held_within(Duration::from_secs(60)));
        std::fs::write(&path, &replacement).unwrap();
        fault::release();
        running.join().unwrap().unwrap()
    });
    fault::clear();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.capture_pending);
    assert_eq!(pass.bytes_stored, 0);
    let input = engine.location_id(&path);
    let committed = engine
        .reads
        .reader()
        .unwrap()
        .latest_chain(&input)
        .unwrap()
        .unwrap();
    assert_eq!((committed.chain, committed.stored), (1, first.len() as u64));
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let rewritten = reader.latest_chain(&input).unwrap().unwrap();
    assert_eq!(rewritten.chain, 2);
    assert_eq!(
        reader
            .stored_range(&input, 1, 0, first.len() as u64, &engine.blobs)
            .unwrap(),
        first.as_bytes()
    );
    assert_eq!(
        reader
            .stored_range(&input, 2, 0, replacement.len() as u64, &engine.blobs)
            .unwrap(),
        replacement.as_bytes()
    );
}

#[test]
fn a_device_number_change_reuses_only_compared_bytes() {
    let home = ClaudeHome::new();
    let first = claude_user("u1", "remount", "before remount");
    let second = claude_user("u2", "remount", "after remount");
    let path = home.file("remount.jsonl");
    home.append("remount.jsonl", &first);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());

    let change_device = |engine: &mut Engine| {
        let state = engine.inputs.get_mut(&path).unwrap();
        state.facts.as_mut().unwrap().device ^= 1;
    };
    change_device(&mut engine);
    let same = engine.run_pass().unwrap();
    assert!(same.errors.is_empty(), "{same:?}");
    assert_eq!(
        (same.bytes_read, same.bytes_stored),
        (first.len() as u64, 0)
    );
    assert_eq!(engine.inputs[&path].chain, Some(1));

    change_device(&mut engine);
    home.append("remount.jsonl", &second);
    let appended = engine.run_pass().unwrap();
    assert!(appended.errors.is_empty(), "{appended:?}");
    assert_eq!(appended.bytes_stored, second.len() as u64);
    assert_eq!(engine.inputs[&path].chain, Some(1));
    assert_eq!(home.texts("remount"), ["before remount", "after remount"]);
}
