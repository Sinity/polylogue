use super::*;
use polylogue_model::authorship::{Author, Authorship, Channel};
use std::sync::mpsc::sync_channel;

/// Runs the loop until it is idle, with whatever was queued before it started.
fn until_idle(engine: &mut Engine, queued: Vec<LoopMsg>) -> RunSummary {
    let (tx, rx) = crossbeam_channel::unbounded();
    for message in queued {
        tx.send(message).unwrap();
    }
    let status = Arc::new(SharedLoopStatus::default());
    engine
        .run(tx, rx, &status, Lifetime::UntilIdle, || {})
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
            format: None,
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
    // The first pass answered the import while it built the new archive's
    // index, and went on to install it: the import is read.
    assert!(summary.passes >= 1, "{summary:?}");
    assert_eq!(home.count("SELECT count(*) AS n FROM message"), 1);
    // The import is the archive's only input: the transcript was never read.
    assert_eq!(home.count("SELECT count(*) AS n FROM a.input"), 1);
    assert!(home.texts("unread").is_empty());
}

#[test]
fn replay_uses_recorded_raw_paths() {
    use std::os::unix::ffi::OsStringExt;

    let mut home = ClaudeHome::new();
    // Fixed roots may contain raw path bytes; the provider's wildcard path
    // components and transcript stem still use their supported spelling.
    let raw_home = home
        .home
        .with_file_name(std::ffi::OsString::from_vec(b"home-\xff".to_vec()));
    std::fs::rename(&home.home, &raw_home).unwrap();
    home.home = raw_home;
    let path = home.file("native-kept.jsonl");
    std::fs::write(
        &path,
        claude_user("one", "native-kept", "native kept words"),
    )
    .unwrap();
    {
        let (mut engine, _lock) = open_engine(&home);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let reader = engine.reads.reader().unwrap();
        let known = reader
            .known_watched_input_at("test-machine", path.as_os_str().as_encoded_bytes())
            .unwrap()
            .expect("the fixed raw root was captured");
        assert_eq!(
            known.path_bytes.as_slice(),
            path.as_os_str().as_encoded_bytes()
        );
        let plans = reader
            .chains_behind_index(
                &[("claude-code".into(), "synthetic-previous-reading".into())],
                RevisionChains::Newest,
                None,
            )
            .unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].input, known.input_id);
        assert_eq!(
            plans[0].source.path_bytes.as_deref(),
            Some(path.as_os_str().as_encoded_bytes())
        );
    }
    assert_eq!(home.texts("native-kept"), ["native kept words"]);
    let stored = rows(&home);
    assert!(
        stored.iter().all(|count| *count > 0),
        "the fixture must capture original bytes and their observation"
    );
    std::fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = false;
    // Stored identity and parser selection do not depend on this host having
    // observed the input. No raw path is used to reopen the gone file.
    engine.config.machine = "another-machine".into();
    let summary = until_idle(&mut engine, Vec::new());
    assert!(summary.work.errors.is_empty(), "{summary:?}");
    assert_eq!(home.texts("native-kept"), ["native kept words"]);
    assert_eq!(rows(&home), stored);
    assert!(engine.inputs.is_empty());
    let idle = engine.run_pass().unwrap();
    assert!(idle.errors.is_empty(), "{idle:?}");
    assert_eq!(
        (idle.bytes_read, idle.bytes_stored, idle.messages_written),
        (0, 0, 0)
    );
}

#[cfg(feature = "faults")]
#[test]
fn stored_planning_skips_watched_history() {
    use polylogue_store::fault::{self, Action};

    let home = ClaudeHome::new();
    home.append("stored.jsonl", &claude_user("one", "stored", "stored body"));
    {
        let (mut engine, _lock) = open_engine(&home);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    let stored = rows(&home);
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = false;
    // Refuse the entire observation-history lookup, not just one result row.
    // Successful replay shows that this work is absent from the real route.
    fault::clear();
    fault::arm(
        "read.watched_inputs",
        "test-machine",
        Action::Fail,
        0,
        u32::MAX,
    );
    let replay = engine.run_pass();
    let skipped = engine.inputs.is_empty() && !engine.watched_loaded;
    engine.config.reads_sources = true;
    let restoration = engine.load();
    fault::clear();
    let replay = replay.unwrap();
    assert!(replay.errors.is_empty(), "{replay:?}");
    assert!(skipped);
    assert!(
        restoration
            .unwrap_err()
            .to_string()
            .contains("read.watched_inputs")
    );
    engine.load().unwrap();
    assert!(engine.watched_loaded);
    assert_eq!(engine.inputs.len(), 1);
    assert_eq!(
        engine.inputs[&home.file("stored.jsonl")].parsed_upto,
        std::fs::metadata(home.file("stored.jsonl")).unwrap().len()
    );
    assert_eq!(home.texts("stored"), ["stored body"]);
    assert_eq!(rows(&home), stored);
}

/// A run until idle whose turns fail the same way returns after the repeat,
/// with the failure reported, instead of retrying it.
#[cfg(feature = "faults")]
#[test]
fn a_turn_failing_as_the_one_before_ends_an_idle_run() {
    use polylogue_store::fault::{self, Action};

    let home = ClaudeHome::new();
    home.append("stored.jsonl", &claude_user("one", "stored", "stored body"));
    let (mut engine, _lock) = open_engine(&home);
    engine.config.reads_sources = true;
    engine.config.live_every = std::time::Duration::from_millis(1);
    fault::clear();
    fault::arm(
        "read.watched_inputs",
        "test-machine",
        Action::Fail,
        0,
        u32::MAX,
    );
    let summary = until_idle(&mut engine, Vec::new());
    fault::clear();
    assert_eq!(summary.passes, 2, "{summary:?}");
    assert!(
        summary
            .work
            .errors
            .iter()
            .any(|e| e.to_string().contains("read.watched_inputs")),
        "{summary:?}"
    );
}

#[test]
fn a_refused_reading_keeps_its_rows_and_is_read_again_under_a_new_parser_version() {
    for old_version in [false, true] {
        let home = ClaudeHome::new();
        home.append(
            "upgrade.jsonl",
            &claude_user("one", "upgrade", "preserved reading"),
        );
        let (mut engine, _lock) = open_engine(&home);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        assert_eq!(home.texts("upgrade"), ["preserved reading"]);
        let state = &engine.inputs[&home.file("upgrade.jsonl")];
        let input = state.input_id.clone();
        let length = state.stored;
        let version = state.origin.parse_version();
        // A refusal after a committed prefix changes neither that prefix's
        // rows nor its cursor. A resetting step would retract the reading.
        let mut step = polylogue_store::ParseStep::failed(
            length..length,
            polylogue_model::ChainFailure::refused("recorded synthetic refusal"),
        );
        step.parse_version = version.clone();
        engine
            .writer
            .send(polylogue_store::Batch {
                input: InputKey {
                    id: input.clone(),
                    format: state.origin.name.into(),
                    route: state.origin.route,
                },
                change: Change::Reparse {
                    chain: state.chain.unwrap(),
                },
                observation: None,
                continues: false,
                new_blob: None,
                blobs: vec![],
                length,
                parse: vec![step],
                capture_read: None,
                release: Default::default(),
            })
            .unwrap();
        assert!(engine.writer.flush().unwrap().errors.is_empty());
        assert_eq!(
            home.texts("upgrade"),
            ["preserved reading"],
            "a refusal at the committed cursor keeps its prior rows"
        );
        if old_version {
            polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
        }
        let stored = rows(&home);
        engine.config.reads_sources = false;
        std::fs::remove_file(home.file("upgrade.jsonl")).unwrap();
        engine.archive.advance_generation_for_test().unwrap();
        let replay = engine.run_pass().unwrap();
        assert!(replay.errors.is_empty(), "{replay:?}");
        let progress = engine.reads.reader().unwrap().query(
            "SELECT state,parser_version,parsed_to FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1",
            &[polylogue_store::SqlValue::Text(input)],
        ).unwrap();
        assert_eq!(
            progress[0]["state"],
            if old_version { "parsed" } else { "failed" }
        );
        assert_eq!(progress[0]["parser_version"], version);
        assert_eq!(progress[0]["parsed_to"].as_u64(), Some(length));
        assert_eq!(rows(&home), stored);
        assert_eq!(home.texts("upgrade"), ["preserved reading"]);
    }
}
