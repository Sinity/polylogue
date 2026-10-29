mod coverage;
use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::io::Read;
use std::os::unix::fs::MetadataExt;

#[cfg(feature = "faults")]
mod live_queue;

#[test]
fn extent_checkpoint_survives_a_rewrite_read_failure_before_same_inode_growth() {
    failed_rewrite_then_growth(false);
}

#[test]
fn extent_checkpoint_keeps_tail_detection_after_a_rewrite_read_failure() {
    failed_rewrite_then_growth(true);
}

fn failed_rewrite_then_growth(edit_tail: bool) {
    let home = ClaudeHome::new();
    let old = "original body ".repeat(40);
    home.append("checkpoint.jsonl", &claude_user("one", "session", &old));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 256);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    home.append(
        "checkpoint.jsonl",
        &claude_user("two", "session", "second body"),
    );
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let path = home.file("checkpoint.jsonl");
    let inode = std::fs::metadata(&path).unwrap().ino();
    // Touch the same file so planning selects a rewrite, then make only that
    // planned read fail. Moving it aside preserves its inode and bytes.
    std::fs::File::open(&path)
        .unwrap()
        .set_modified(std::time::UNIX_EPOCH)
        .unwrap();
    let now = FileFacts::of(&std::fs::metadata(&path).unwrap());
    let plan = engine.plan(&path, now).unwrap().unwrap();
    assert_eq!(plan.change, Change::Rewritten);
    let aside = home.dir.path().join("held-source");
    std::fs::rename(&path, &aside).unwrap();
    let mut failed = Pass::default();
    engine.execute_pool(vec![plan], &mut failed).unwrap();
    assert_eq!(failed.errors.len(), 1, "{failed:?}");
    assert!(failed.errors[0].starts_with("read "), "{failed:?}");
    std::fs::rename(&aside, &path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    let second = if edit_tail {
        let source = std::fs::read_to_string(&path).unwrap();
        std::fs::write(&path, source.replace("second body", "edited body")).unwrap();
        "edited body"
    } else {
        "second body"
    };
    home.append(
        "checkpoint.jsonl",
        &claude_user("three", "session", "third body"),
    );
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        home.texts("session"),
        [old, second.into(), "third body".into()]
    );
    let reader = engine.reads.reader().unwrap();
    let known = reader
        .known_input_at("test-machine", &path.to_string_lossy())
        .unwrap()
        .unwrap();
    let chain = reader
        .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
        .expect("a failed read must not let the next append overwrite the stored prefix");
    let mut actual = Vec::new();
    chain
        .reader(Arc::clone(&engine.blobs))
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, std::fs::read(&path).unwrap());
    assert_eq!(known.parsed_upto, known.stored);
    assert_eq!(
        reader
            .query("SELECT count(DISTINCT chain) AS n FROM a.revision", &[])
            .unwrap()[0]["n"],
        if edit_tail { 2 } else { 1 },
    );
}

// Unit tests need this crate's Engine rather than the testkit's dependency.
fn open_engine(home: &ClaudeHome) -> (Engine, polylogue_store::ArchiveLock) {
    let mut config = Config::from_env();
    config.home = home.home.clone();
    config.tool_dirs.clear();
    config.workers = 3;
    config.machine = "test-machine".into();
    config.host = "test".into();
    let (archive, lock) = Archive::open_or_create(&home.root).unwrap();
    (Engine::new(config, archive).unwrap(), lock)
}

#[test]
fn changed_paths_do_not_run_a_full_scan_and_a_rescan_finds_missed_files() {
    let home = ClaudeHome::new();
    home.append("first.jsonl", &claude_user("one", "first", "first"));
    let (mut engine, _lock) = open_engine(&home);
    assert_eq!(engine.run_pass().unwrap().full_scans, 1);
    home.append("changed.jsonl", &claude_user("two", "changed", "changed"));
    home.append("missed.jsonl", &claude_user("three", "missed", "missed"));
    let changed = [home.file("changed.jsonl")].into_iter().collect();
    let pass = engine.one_pass(false, &changed).unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.full_scans, 0);
    assert_eq!(home.texts("changed"), ["changed"]);
    assert!(home.texts("missed").is_empty());
    let changes = crate::watch::Changes {
        full: true,
        ..Default::default()
    };
    let pass = engine.one_pass(changes.full, &changes.paths).unwrap();
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
    let (mut engine, _lock) = open_engine(&home);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine.config.live_every = Duration::from_secs(3600);
    let length = |engine: &Engine| {
        engine
            .reads
            .reader()
            .unwrap()
            .input_progress(&engine.location_id(&db))
            .unwrap()
            .unwrap()
            .length
    };
    let before = length(&engine);
    let changed = [db.clone()].into_iter().collect();
    source.execute("UPDATE kept SET n=1", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed).unwrap();
    engine.captures.completed(true);
    home.append("other.jsonl", &claude_user("one", "other", "other"));
    let other = [home.file("other.jsonl")].into_iter().collect();
    engine.one_pass(false, &other).unwrap();
    assert_eq!(length(&engine), before);
    source.execute("UPDATE kept SET n=2", []).unwrap();
    engine.capture_checked = None;
    engine.one_pass(false, &changed).unwrap();
    engine.captures.completed(true);
    assert_eq!(length(&engine), before);
    engine.capture_checked = None;
    engine.one_pass(false, &Default::default()).unwrap();
    let completed = engine.captures.completed(true);
    assert!(completed.errors.is_empty(), "{:?}", completed.errors);
    assert!(length(&engine) > before);
}

mod capture_only;
mod live_inputs;
