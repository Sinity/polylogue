use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use std::sync::mpsc::{channel, sync_channel};

/// Runs the loop until it is idle, with whatever was queued before it started.
fn until_idle(engine: &mut Engine, queued: Vec<LoopMsg>) -> RunSummary {
    let (tx, rx) = channel();
    for message in queued {
        tx.send(message).unwrap();
    }
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    engine
        .run(tx, &rx, &status, Lifetime::UntilIdle, || {})
        .unwrap()
}

fn rows(home: &ClaudeHome) -> [i64; 3] {
    [
        home.count("SELECT count(*) AS n FROM a.input"),
        home.count("SELECT count(*) AS n FROM a.observation"),
        home.count("SELECT count(*) AS n FROM a.revision"),
    ]
}

/// Every file under `dir`, relative to it.
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(next) = pending.pop() {
        for entry in std::fs::read_dir(&next).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.push(path.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    found.sort();
    found
}

/// law: until-idle-ends-after-work
#[test]
fn a_loop_run_until_idle_stores_what_it_finds_then_returns() {
    let home = ClaudeHome::new();
    home.append("first.jsonl", &claude_user("one", "first", "first words"));
    let (mut engine, _lock) = open_engine(&home);
    let summary = until_idle(&mut engine, Vec::new());
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    assert!(summary.passes >= 1);
    assert_eq!(summary.imports, 0);
    assert_eq!(summary.work.full_scans, 1);
    assert!(summary.work.inputs_changed >= 1, "{summary:?}");
    assert_eq!(home.texts("first"), ["first words"]);
}

/// law: no-sources-reads-no-source
#[test]
fn without_sources_a_fresh_archive_records_no_input_and_writes_nothing_under_home() {
    let home = ClaudeHome::new();
    home.append("present.jsonl", &claude_user("one", "present", "on disk"));
    // A hook spool beside the transcript: a spool is a source too.
    let spool = crate::test_config().spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let spool = home
        .home
        .join(spool.strip_prefix("/synthetic/home").unwrap());
    std::fs::create_dir_all(&spool).unwrap();
    std::fs::write(
        spool.join("4242.ndjson"),
        "{\"at_ns\":\"1767225600000000000\",\"event\":\"Stop\",\"provider\":\"claude-code\",\"session_id\":\"present\",\"payload\":{}}\n",
    )
    .unwrap();
    let before = files(&home.home);
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = false;
    let summary = until_idle(&mut engine, Vec::new());
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    assert_eq!(summary.work.full_scans, 0);
    assert_eq!(summary.work.inputs_seen, 0);
    assert_eq!(rows(&home), [0, 0, 0]);
    assert_eq!(
        files(&home.home),
        before,
        "a sourceless loop wrote under home"
    );
}

/// law: no-sources-parses-what-is-stored
#[test]
fn without_sources_the_index_is_rebuilt_from_stored_bytes_and_a_gone_file_is_not_observed() {
    let home = ClaudeHome::new();
    home.append("kept.jsonl", &claude_user("one", "kept", "stored words"));
    {
        let (mut engine, _lock) = open_engine(&home);
        let summary = until_idle(&mut engine, Vec::new());
        assert!(summary.work.errors.is_empty(), "{summary:?}");
    }
    assert_eq!(home.texts("kept"), ["stored words"]);
    let stored = rows(&home);
    polylogue_testkit::delete_index(&home.root);
    std::fs::remove_file(home.file("kept.jsonl")).unwrap();
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = false;
    let summary = until_idle(&mut engine, Vec::new());
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    assert_eq!(home.texts("kept"), ["stored words"]);
    // The file is gone, but nothing looked: no `deleted` observation.
    assert_eq!(rows(&home), stored);
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.observation WHERE class = 'deleted'"),
        0
    );
}

/// law: no-sources-takes-imports
#[test]
fn without_sources_an_import_is_stored_and_the_loop_then_becomes_idle() {
    let home = ClaudeHome::new();
    home.append("unread.jsonl", &claude_user("one", "unread", "never read"));
    let path = home.dir.path().join("conversation.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&serde_json::json!({
            "uuid": "imported-session", "chat_messages": [{
                "uuid": "imported-message", "sender": "human", "text": "imported words"
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = false;
    let (reply, answer) = sync_channel(1);
    let summary = until_idle(
        &mut engine,
        vec![LoopMsg::Import {
            paths: vec![path],
            from_archive: None,
            dry_run: false,
            authorship: Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            },
            parent: None,
            reply,
        }],
    );
    let report = answer.recv().unwrap().unwrap();
    assert_eq!(report.refused.len(), 0, "{report:?}");
    assert_eq!(report.imported.len(), 1, "{report:?}");
    assert_eq!(summary.imports, 1);
    assert!(
        summary.passes >= 2,
        "the import is followed by a pass: {summary:?}"
    );
    // The import is the archive's only input: the transcript was never read.
    assert_eq!(home.count("SELECT count(*) AS n FROM a.input"), 1);
    assert!(home.texts("unread").is_empty());
}
