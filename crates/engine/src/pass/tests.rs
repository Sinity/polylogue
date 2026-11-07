mod coverage;
mod no_sources;
mod parent_admission;
use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};
use std::io::Read;
use std::os::unix::fs::MetadataExt;

#[cfg(feature = "faults")]
mod live_queue;
#[cfg(feature = "faults")]
mod standings;

#[test]
fn confirmed_capture_resumes_in_the_same_engine_without_reacquiring_provider_bytes() {
    for codex in [false, true] {
        let home = ClaudeHome::new();
        let (path, origin, bytes, session) = if codex {
            let path = home.home.join(".codex/sessions/rollout-delayed.jsonl");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let bytes = "{\"type\":\"session_meta\",\"payload\":{\"id\":\"delayed\"}}\n{\"type\":\"response_item\",\"payload\":{\"id\":\"m1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"independent payload\"}]}}\n".to_owned();
            std::fs::write(&path, &bytes).unwrap();
            (path, "codex", bytes, "codex:delayed")
        } else {
            let bytes = claude_user("u1", "delayed", "confirmed once");
            home.append("delayed.jsonl", &bytes);
            (
                home.file("delayed.jsonl"),
                "claude-code",
                bytes,
                "claude-code:delayed",
            )
        };
        let origin = polylogue_sources::origin::by_name(origin).unwrap();
        let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
        engine.load().unwrap();
        assert!(engine.track(&collect::Found {
            path: path.clone(),
            origin
        }));
        let now = FileFacts::of(&std::fs::metadata(&path).unwrap());
        let mut plan = engine.plan(&path, now).unwrap().unwrap();
        if codex {
            plan.input.id = "codex:rollout:delayed".into();
        }
        plan.parser_deferred = true;
        let mut job = plan.next_job();
        job.end = (bytes.len() / 2) as u64;
        job.length = job.end;
        job.at_end = false;
        let mut context = engine.sampling_context(&path, origin.name, "").unwrap();
        context.facts = Some(now);
        job.acquisition = Some(context);
        let (prepared, done) = crate::pool::run(job, &engine.archive, &engine.blobs, true);
        assert!(done.error.is_none(), "{:?}", done.error);
        // A cursorless raw attempt is still unsupported before confirmation.
        engine.apply_raw_result(&mut plan, &done).unwrap();
        let mut unconfirmed = Pass::default();
        engine
            .record_raw_status(&plan, done.raw_acquisition.as_ref(), &mut unconfirmed)
            .unwrap();
        assert_eq!(unconfirmed.errors.len(), 1);
        let results = engine
            .writer
            .sender()
            .unwrap()
            .store_group(vec![prepared.unwrap()], Arc::clone(&engine.blobs))
            .unwrap();
        assert!(results.into_iter().all(|result| result.is_ok()));
        // Confirmation arrives after the initial load, without finishing a
        // logical plan. A rollout may not have been tracked at all yet.
        if codex {
            engine.inputs.remove(&path);
        }
        assert!(
            engine
                .inputs
                .get(&path)
                .is_none_or(|state| state.chain.is_none())
        );
        // A retired producer cursor is unavailable, while the exact captured
        // input is now confirmed. Mismatched identity, coordinates and file
        // evidence must still report the missing continuation proof.
        let original_id = plan.input.id.clone();
        let original_format = plan.input.format.clone();
        let original_facts = plan.facts;
        let original_size = plan.size;
        let mut not_covered = done.raw_acquisition.clone().unwrap();
        not_covered.range.end += 1;
        let mut refused = Pass::default();
        engine
            .record_raw_status(&plan, Some(&not_covered), &mut refused)
            .unwrap();
        assert_eq!(refused.errors.len(), 1);
        let mut no_receipt = Pass::default();
        engine
            .record_raw_status(&plan, None, &mut no_receipt)
            .unwrap();
        assert_eq!(no_receipt.errors.len(), 1);
        for mismatch in 0..4 {
            match mismatch {
                0 => plan.input.id = "another-input".into(),
                1 => plan.input.format = "another-format".into(),
                2 => plan.facts = None,
                _ => plan.size += 1,
            }
            let mut refused = Pass::default();
            engine
                .record_raw_status(&plan, done.raw_acquisition.as_ref(), &mut refused)
                .unwrap();
            assert_eq!(refused.errors.len(), 1, "{refused:?}");
            plan.input.id = original_id.clone();
            plan.input.format = original_format.clone();
            plan.facts = original_facts;
            plan.size = original_size;
        }
        plan.raw.as_mut().unwrap().custody = Some(polylogue_store::primary::RawCustodyReceipt {
            identity: "unresolved-acquisition".into(),
            count: 1,
        });
        let mut unresolved = Pass::default();
        engine
            .record_raw_status(&plan, done.raw_acquisition.as_ref(), &mut unresolved)
            .unwrap();
        assert_eq!(unresolved.errors.len(), 1);
        plan.raw.as_mut().unwrap().custody = None;
        let mut confirmed = Pass::default();
        engine
            .record_raw_status(&plan, done.raw_acquisition.as_ref(), &mut confirmed)
            .unwrap();
        assert!(confirmed.errors.is_empty(), "{confirmed:?}");
        assert!(
            confirmed.capture_pending,
            "confirmed bytes still need their reading"
        );
        assert!(plan.raw.as_ref().unwrap().cursor.is_none());
        assert_eq!(engine.inputs[&path].stored, done.end);
        assert!(engine.inputs[&path].stored < bytes.len() as u64);
        // The next ordinary plan reads only the missing suffix from the exact
        // confirmed prefix. No retired private cursor is fabricated.
        let mut tail = engine.plan(&path, now).unwrap().unwrap();
        assert_eq!(tail.stored_upto, done.end);
        assert!(matches!(tail.change, Change::Appended { from } if from == done.end));
        let mut job = tail.next_job();
        assert_eq!(job.start, done.end);
        let mut context = engine.sampling_context(&path, origin.name, "").unwrap();
        context.facts = Some(now);
        job.acquisition = Some(context);
        let (prepared, last) = crate::pool::run(job, &engine.archive, &engine.blobs, true);
        assert!(last.error.is_none(), "{:?}", last.error);
        let results = engine
            .writer
            .sender()
            .unwrap()
            .store_group(vec![prepared.unwrap()], Arc::clone(&engine.blobs))
            .unwrap();
        assert!(results.into_iter().all(|result| result.is_ok()));
        engine.refresh_confirmed_input(&path).unwrap();
        assert_eq!(engine.inputs[&path].stored, bytes.len() as u64);
        let parse = engine.stored_plan(&path).unwrap().unwrap();
        assert_eq!(parse.input.id, original_id);
        std::fs::remove_file(&path).unwrap();
        let mut pass = Pass::default();
        engine.execute_pool(vec![parse], &mut pass).unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert_eq!(pass.bytes_read, bytes.len() as u64);
        assert_eq!(pass.bytes_stored, 0);
        let session = polylogue_model::SessionId::parse(session).unwrap();
        let messages = engine
            .reads
            .reader()
            .unwrap()
            .messages(&session, false)
            .unwrap();
        assert_eq!(messages.len(), 1);
        let mut settled = Pass::default();
        engine
            .record_raw_status(&plan, done.raw_acquisition.as_ref(), &mut settled)
            .unwrap();
        assert!(settled.errors.is_empty(), "{settled:?}");
        assert!(
            !settled.capture_pending,
            "a confirmed reading needs no extra pass"
        );
        engine.finish_plan(plan);
        assert_eq!(
            messages[0].text(),
            if codex {
                "independent payload"
            } else {
                "confirmed once"
            }
        );
        let state = engine.inputs.get_mut(&path).unwrap();
        state.parser_state = Some(serde_json::json!({"continuation":"unchanged"}));
        engine.refresh_confirmed_input(&path).unwrap();
        assert_eq!(
            engine.inputs[&path].parser_state,
            Some(serde_json::json!({"continuation":"unchanged"}))
        );
        assert!(engine.stored_plan(&path).unwrap().is_none());
        engine.inputs.get_mut(&path).unwrap().parsed_upto = 0;
        engine.refresh_confirmed_input(&path).unwrap();
        assert!(engine.inputs[&path].parser_state.is_none());
        assert_eq!(engine.inputs[&path].parsed_upto, bytes.len() as u64);
        assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 1);
        assert_eq!(home.count("SELECT count(*) AS n FROM a.extent"), 2);
    }
}

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
        .known_input_at("test-machine", path.as_os_str().as_encoded_bytes())
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
    let mut config = crate::test_config();
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
