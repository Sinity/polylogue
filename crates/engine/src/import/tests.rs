use super::*;

#[test]
fn capture_keeps_original_identity_after_the_source_is_rewritten_or_deleted() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    for delete in [false, true] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("s1.jsonl");
        let original = claude_user("u1", "s1", "captured");
        std::fs::write(&path, &original).unwrap();
        let (mut engine, _lock) = engine(&w, 64);
        let captured =
            Captured::stream(&engine.archive, std::fs::File::open(&path).unwrap()).unwrap();
        if delete {
            std::fs::remove_file(&path).unwrap();
        } else {
            std::fs::write(&path, claude_user("u1", "s1", "replacement")).unwrap();
        }
        let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
        let mut report = ImportReport::default();
        engine
            .import_captured(&path, origin, captured, None, &mut report)
            .unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(w.texts("s1"), ["captured"]);
        let reader = engine.reads.reader().unwrap();
        let stored = reader
            .input_progress(blake3::hash(original.as_bytes()).to_hex().as_ref())
            .unwrap()
            .unwrap();
        assert_eq!(stored.length, original.len() as u64);
        assert_eq!(stored.parsed_upto, original.len() as u64);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 1);
    }
}

#[test]
fn a_registered_import_retries_after_restart_without_storing_part_blobs() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    let w = ClaudeHome::new();
    let path = w.dir.path().join("s1.jsonl");
    let original = claude_user("u1", "s1", "stored before parse");
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&w, 64);
        let captured =
            Captured::stream(&engine.archive, std::fs::File::open(&path).unwrap()).unwrap();
        let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
        let mut report = ImportReport::default();
        // Simulate a stop after committing the source and before the parse job.
        let plan = engine
            .import_plan(&path, origin, captured, None, &mut report)
            .unwrap()
            .unwrap();
        assert_eq!(plan.stored_upto, original.len() as u64);
    }
    let (mut engine, _lock) = engine(&w, 64);
    let retry = engine
        .import_with_options(
            std::slice::from_ref(&path),
            &ImportOptions {
                origin: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(retry.pass.errors.is_empty(), "{:?}", retry.pass.errors);
    assert_eq!(retry.already_stored, [path]);
    assert_eq!(retry.pass.bytes_stored, 0);
    assert_eq!(w.texts("s1"), ["stored before parse"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 1);
}

#[test]
fn a_codex_import_retry_reads_only_after_its_committed_checkpoint() {
    let home = polylogue_testkit::ClaudeHome::new();
    let thread = "019a0000-0000-7000-8000-000000000001";
    let name = format!("rollout-2026-03-01T10-00-00-{thread}.jsonl");
    let record = |kind, payload| {
        format!(
            "{}\n",
            serde_json::json!({"timestamp":"2026-03-01T10:00:01Z","type":kind,"payload":payload})
        )
    };
    let prefix = record(
        "session_meta",
        serde_json::json!({"id":thread,"timestamp":"2026-03-01T10:00:00Z","cwd":"/work/fixture"}),
    ) + &record(
        "turn_context",
        serde_json::json!({"turn_id":"turn-one","model":"checkpoint-model"}),
    ) + &record(
        "response_item",
        serde_json::json!({"type":"message","id":"before","role":"user","content":[{"type":"input_text","text":"original prompt"}]}),
    );
    let suffix = record(
        "response_item",
        serde_json::json!({"type":"message","id":"after","role":"assistant","content":[{"type":"output_text","text":"continued answer"}]}),
    );
    let cursor = prefix.len() as u64;
    let original = prefix + &suffix;
    let path = home.dir.path().join(name);
    std::fs::write(&path, &original).unwrap();
    let options = ImportOptions {
        origin: Some("codex".into()),
        ..Default::default()
    };
    {
        let (mut engine, _lock) = engine(&home, cursor);
        let origin = polylogue_sources::origin::by_name("codex").unwrap();
        let captured = Captured::stream(&engine.archive, original.as_bytes()).unwrap();
        let mut plan = engine
            .import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap()
            .unwrap();
        let job = plan.next_job(&engine.stopped_attempt);
        assert_eq!((job.start, job.end), (0, cursor));
        let (batch, done) = pool::run(job, &engine.archive, &engine.blobs, true, None);
        assert!(done.error.is_none(), "{done:?}");
        assert_eq!(done.parsed_upto, cursor);
        for result in engine.writer.store_group(vec![batch.unwrap()]).unwrap() {
            result.unwrap();
        }
        let reader = engine.reads.reader().unwrap();
        let checkpoint = reader
            .parser_checkpoint(
                &plan.input.id,
                plan.parser_chain.unwrap(),
                &origin.parse_version(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            checkpoint.progress,
            polylogue_model::batch::ParseProgress::Parsing
        );
        assert_eq!(checkpoint.parsed_upto, cursor);
        assert!(checkpoint.checkpoint.is_some());
    }

    let (mut engine, _lock) = engine(&home, cursor);
    engine.stopped_attempt.executable = polylogue_model::BlobHash::of(b"different executable");
    let retry = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    assert!(
        retry.refused.is_empty() && retry.pass.errors.is_empty(),
        "{retry:?}"
    );
    assert_eq!(retry.already_stored.as_slice(), std::slice::from_ref(&path));
    assert_eq!(retry.pass.bytes_stored, 0);
    assert_eq!(retry.pass.bytes_read, suffix.len() as u64);
    let reader = engine.reads.reader().unwrap();
    let session = polylogue_model::SessionId::parse(&format!("codex:{thread}")).unwrap();
    let messages = reader.messages(&session, false).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].blocks[0].text(), Some("original prompt"));
    assert_eq!(messages[1].blocks[0].text(), Some("continued answer"));
    assert_eq!(messages[1].model.as_deref(), Some("checkpoint-model"));
    let input = blake3::hash(original.as_bytes()).to_hex();
    let progress = reader.input_progress(input.as_ref()).unwrap().unwrap();
    assert_eq!(progress.parsed_upto, original.len() as u64);
    assert_eq!(
        reader
            .stored_range(
                input.as_ref(),
                progress.chain,
                0,
                progress.length,
                &engine.blobs
            )
            .unwrap(),
        original.as_bytes()
    );
    let expected = [serde_json::json!({"input": input.to_string()})];
    for table in ["a.revision", "a.extent"] {
        let rows = reader
            .query(&format!("SELECT input FROM {table} ORDER BY input"), &[])
            .unwrap();
        assert_eq!(rows, expected, "{table}");
    }
}

#[test]
fn file_of_declared_origin_without_parser_is_stored_by_import() {
    let w = polylogue_testkit::ClaudeHome::new();
    // Pi's own layout below its root; Pi has no parser yet.
    let path = w
        .dir
        .path()
        .join("staged/.pi/agent/sessions/2026/11/run.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let bytes = b"{\"type\":\"session\",\"id\":\"run\"}\n";
    std::fs::write(&path, bytes).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(
        report.refused.is_empty() && report.pass.errors.is_empty(),
        "{report:?}"
    );
    assert_eq!(report.imported, std::slice::from_ref(&path));
    let [member] = report.members.as_slice() else {
        panic!("one member: {report:?}")
    };
    assert_eq!(
        (member.origin.as_deref(), member.class.as_str()),
        (Some("pi"), "no parser yet")
    );
    assert_eq!(report.packages[0].assets, 0);
    let input = blake3::hash(bytes).to_hex();
    let reader = engine.reads.reader().unwrap();
    let progress = reader.input_progress(input.as_ref()).unwrap().unwrap();
    assert_eq!(
        reader
            .stored_range(
                input.as_ref(),
                progress.chain,
                0,
                progress.length,
                &engine.blobs
            )
            .unwrap(),
        bytes
    );
    drop(reader);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE origin = 'pi'"),
        1
    );
    // Named explicitly, the same bytes are already stored.
    let again = engine
        .import_with_options(
            std::slice::from_ref(&path),
            &ImportOptions {
                origin: Some("pi".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(again.refused.is_empty(), "{again:?}");
    assert_eq!(again.already_stored, [path]);
}

#[test]
fn dry_run_classifies_members_and_stores_nothing() {
    let w = polylogue_testkit::ClaudeHome::new();
    let path = w
        .dir
        .path()
        .join("staged/.pi/agent/sessions/2026/11/run.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"{\"type\":\"session\"}\n").unwrap();
    let unknown = w.dir.path().join("staged/notes.txt");
    std::fs::write(&unknown, b"plain text").unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine
        .import_with_options(
            &[path.clone(), unknown.clone()],
            &ImportOptions {
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
    let classes: Vec<_> = report
        .members
        .iter()
        .map(|member| (member.path.clone(), member.class.as_str()))
        .collect();
    assert_eq!(
        classes,
        [(path, "no parser yet"), (unknown.clone(), "refused")]
    );
    assert_eq!(report.refused.len(), 1);
    assert_eq!(report.refused[0].path, unknown);
    assert!(report.imported.is_empty() && report.already_stored.is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
}

#[test]
fn a_source_changed_during_capture_is_refused_before_registration() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("source");
    std::fs::write(&path, "before").unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let before = FileFacts::of(&file.metadata().unwrap());
    std::fs::write(&path, "after, with another length").unwrap();
    assert!(unchanged(&file, before, before.size).is_err());
    assert!(unchanged(&file, FileFacts::of(&file.metadata().unwrap()), 1).is_err());
}

#[test]
fn capture_failure_leaves_no_catalog_input() {
    struct Failed;
    impl std::io::Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("synthetic interrupted import"))
        }
    }
    let w = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = engine(&w, 64);
    assert!(Captured::stream(&engine.archive, Failed).is_err());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 0);
}

pub(crate) fn engine(
    w: &polylogue_testkit::ClaudeHome,
    chunk: u64,
) -> (Engine, polylogue_store::ArchiveLock) {
    let new = !w.root.exists();
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let config = Config {
        clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
        home: w.home.clone(),
        tool_dirs: Default::default(),
        workers: 3,
        memory_budget: Default::default(),
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "test-machine".into(),
        host: "test".into(),
        chunk,
        live_every: std::time::Duration::from_secs(2),
        reads_sources: true,
    };
    let mut engine = Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap();
    if new {
        // Imports and these tests' passes meet an installed index: the first
        // pass of a loop without sources installs a new archive's empty build
        // (docs/installing.md). Files in the home are left for the tests.
        engine.config.reads_sources = false;
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        assert!(engine.archive.index_build().is_none());
        engine.config.reads_sources = true;
    }
    (engine, lock)
}

#[test]
fn line_import_keeps_a_partial_tail_without_duplicate_parts_on_retry() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    let w = ClaudeHome::new();
    let path = w.dir.path().join("s1.jsonl");
    let original = claude_user("u1", "s1", &"x".repeat(2048)) + "{\"type\":\"user\",\"partial\":";
    std::fs::write(&path, &original).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    for attempt in 0..2 {
        let report = engine
            .import_with_options(
                std::slice::from_ref(&path),
                &ImportOptions {
                    origin: Some("claude-code".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(w.texts("s1"), ["x".repeat(2048)]);
        if attempt == 1 {
            assert_eq!(report.pass.bytes_read, 0);
        }
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
        assert_eq!(
            w.count("SELECT SUM(length) AS n FROM a.extent"),
            original.len() as i64
        );
    }
}

#[test]
fn container_members_keep_their_original_hash_and_duplicate_members_share_one_blob() {
    use std::io::Write;
    let w = polylogue_testkit::ClaudeHome::new();
    let path = w.dir.path().join("export.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
    let bytes = vec![0x55; 256 * 1024];
    for name in ["one.bin", "two.bin"] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(&bytes).unwrap();
    }
    zip.start_file(
        "conversations.json",
        zip::write::SimpleFileOptions::default(),
    )
    .unwrap();
    zip.write_all(br#"[{"conversation_id":"empty","mapping":{},"current_node":null}]"#)
        .unwrap();
    zip.finish().unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(
        report.imported,
        [
            path.clone(),
            PathBuf::from(format!("{}!/one.bin", path.display())),
            PathBuf::from(format!("{}!/conversations.json", path.display()))
        ]
    );
    assert_eq!(
        report.already_stored,
        [PathBuf::from(format!("{}!/two.bin", path.display()))]
    );
    let id = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .input_progress(&id)
            .unwrap()
            .unwrap()
            .length,
        bytes.len() as u64
    );
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 3);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 3);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 3);
}

#[test]
fn import_reads_bounded_frames_and_keeps_only_a_recognition_prefix() {
    struct Generated {
        left: u64,
        max_request: usize,
    }
    impl std::io::Read for Generated {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            self.max_request = self.max_request.max(out.len());
            let n = out.len().min(self.left as usize);
            out[..n].fill(b'x');
            self.left -= n as u64;
            Ok(n)
        }
    }
    let w = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = engine(&w, 1024);
    let size = 3 * hashed_blobs::DEFAULT_FRAME as u64;
    let mut generated = Generated {
        left: size,
        max_request: 0,
    };
    let captured = Captured::stream(&engine.archive, &mut generated).unwrap();
    assert_eq!(captured.prepared.len(), size);
    assert_eq!(captured.head, vec![b'x'; 64]);
    assert_eq!(generated.left, 0);
    assert!(generated.max_request <= hashed_blobs::DEFAULT_FRAME);
}

#[test]
fn restart_replays_a_registered_import_after_the_original_is_deleted() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    let w = ClaudeHome::new();
    let path = w.dir.path().join("s1.jsonl");
    let original = claude_user("u1", "s1", "stored, source gone");
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&w, 64);
        let captured =
            Captured::stream(&engine.archive, std::fs::File::open(&path).unwrap()).unwrap();
        let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
        engine
            .import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap();
    }
    std::fs::remove_file(&path).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(w.texts("s1"), ["stored, source gone"]);
    assert_eq!(w.count(SHOWN_FROM_COPY), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 1);
}

/// Messages whose shown reading is an imported copy of a tool's file.
const SHOWN_FROM_COPY: &str = "SELECT count(*) AS n FROM message m JOIN message_parse s
    ON s.message_id = m.id AND s.input_pk = m.input_pk AND s.chain = m.source_chain
    WHERE s.route = 'copy'";

#[test]
fn replayed_import_keeps_the_captured_winner() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &claude_user("u1", "s1", "native winner"));
    let path = w.dir.path().join("s1.jsonl");
    let original = claude_user("u1", "s1", "imported fallback");
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&w, 64);
        engine.run_pass().unwrap();
        let captured =
            Captured::stream(&engine.archive, std::fs::File::open(&path).unwrap()).unwrap();
        let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
        engine
            .import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap();
    }
    std::fs::remove_file(&path).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(w.texts("s1"), ["native winner"]);
    assert_eq!(w.count(SHOWN_FROM_COPY), 0);
}

mod documents;

#[test]
fn batched_and_reversed_import_groups_select_the_same_overlapping_message() {
    use polylogue_testkit::{ClaudeHome, claude_user};

    let batched = ClaudeHome::new();
    let serial = ClaudeHome::new();
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let inputs = [
        (
            "first.jsonl",
            claude_user("shared", "session", "first reading"),
        ),
        (
            "second.jsonl",
            claude_user("shared", "session", "different reading")
                + &claude_user("next", "session", "another message"),
        ),
    ];
    let group = |engine: &Engine| {
        inputs
            .iter()
            .map(|(path, text)| {
                let captured = Captured::stream(&engine.archive, text.as_bytes()).unwrap();
                (PathBuf::from(path), origin, None, captured)
            })
            .collect::<Vec<_>>()
    };
    let (mut reference, _lock) = engine(&serial, 64);
    let mut report = ImportReport::default();
    for item in group(&reference) {
        reference
            .store_import_group(vec![item], &mut report)
            .unwrap();
    }
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    let expected = serial.texts("session");
    assert_eq!(expected.len(), 2);

    for repeat in [false, true] {
        let (mut receiver, _lock) = engine(&batched, 64);
        let mut items = group(&receiver);
        if repeat {
            items.reverse();
        }
        let mut report = ImportReport::default();
        receiver.store_import_group(items, &mut report).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(batched.texts("session"), expected);
        assert_eq!(batched.count("SELECT COUNT(*) AS n FROM a.revision"), 2);
        assert_eq!(report.pass.bytes_stored == 0, repeat);
    }
}

mod browser_files;
mod replay;

mod assets;
mod derivation;
mod empty_assets;
mod stored_bytes;

mod container_members;

mod strict;

#[cfg(feature = "faults")]
#[test]
fn repeated_explicit_import_keeps_a_stopped_attempt_until_its_executable_changes() {
    use polylogue_store::fault::{self, Action};
    use polylogue_testkit::{ClaudeHome, claude_user};
    let home = ClaudeHome::new();
    let path = home.dir.path().join("stopped-import.jsonl");
    let content = claude_user("one", "stopped-import", "kept original");
    std::fs::write(&path, &content).unwrap();
    let (mut engine, lock) = engine(&home, 4096);
    let options = ImportOptions {
        origin: Some("claude-code".into()),
        ..Default::default()
    };
    fault::arm("parse", "stopped-import", Action::Panic, 0, u32::MAX);
    let failed = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    fault::clear();
    assert!(!failed.pass.errors.is_empty(), "{failed:?}");
    let config = engine.config.clone();
    let archive = engine.archive.clone();
    let attempt = engine.stopped_attempt;
    drop(engine);
    let mut engine = Engine::new(config.clone(), archive.clone(), attempt).unwrap();
    let repeat = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    assert!(repeat.pass.errors.is_empty(), "{repeat:?}");
    assert_eq!(repeat.pass.bytes_read, 0, "{repeat:?}");
    assert_eq!(
        home.count("SELECT count(*) AS n FROM parse_state WHERE status='failed'"),
        1
    );
    drop(engine);
    let changed = StoppedAttempt {
        executable: polylogue_model::BlobHash::of(b"new executable image"),
        ..attempt
    };
    let mut engine = Engine::new(config, archive, changed).unwrap();
    let retried = engine
        .import_with_options(std::slice::from_ref(&path), &options)
        .unwrap();
    assert!(retried.pass.errors.is_empty(), "{retried:?}");
    assert_eq!(retried.pass.bytes_read, content.len() as u64, "{retried:?}");
    assert_eq!(retried.pass.bytes_stored, 0, "{retried:?}");
    assert_eq!(
        home.count(
            "SELECT count(*) AS n FROM parse_state WHERE status='done' AND stopped_attempt IS NULL"
        ),
        1
    );
    assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 1);
    drop(engine);
    drop(lock);
}

mod hook_batches;
