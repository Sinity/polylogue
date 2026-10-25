use super::*;

#[test]
fn an_imported_file_keeps_the_origin_whose_layout_it_sits_in() {
    // Claude Code accepts any JSON-lines file on import; a Grok CLI history
    // copied with its directories still reads as Grok CLI.
    let head = br#"{"type":"system","content":"You are a CLI."}"#;
    let grok = Path::new("/backup/.grok/sessions/%2Fhome/01a0/chat_history.jsonl");
    assert_eq!(recognize_import(grok, head).unwrap().name, "grok-cli");
    let loose = Path::new("/backup/s1.jsonl");
    assert_eq!(recognize_import(loose, head).unwrap().name, "claude-code");
}

#[test]
fn capture_keeps_original_identity_after_the_source_is_rewritten_or_deleted() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    for delete in [false, true] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("s1.jsonl");
        let original = claude_user("u1", "s1", "captured");
        std::fs::write(&path, &original).unwrap();
        let (mut engine, _lock) = engine(&w, 64);
        let captured = Captured::stream(
            &engine.archive,
            &engine.config.blob_roots,
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        if delete {
            std::fs::remove_file(&path).unwrap();
        } else {
            std::fs::write(&path, claude_user("u1", "s1", "replacement")).unwrap();
        }
        let origin = recognize_import(&path, &captured.head).unwrap();
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
        let captured = Captured::stream(
            &engine.archive,
            &engine.config.blob_roots,
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        let origin = recognize_import(&path, &captured.head).unwrap();
        let mut report = ImportReport::default();
        // Simulate a stop after committing the source and before the parse job.
        let plan = engine
            .import_plan(&path, origin, captured, None, &mut report)
            .unwrap()
            .unwrap();
        assert_eq!(plan.piece_pos, original.len() as u64);
    }
    let (mut engine, _lock) = engine(&w, 64);
    let retry = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(retry.pass.errors.is_empty(), "{:?}", retry.pass.errors);
    assert_eq!(retry.already_stored, [path]);
    assert_eq!(retry.pass.bytes_stored, 0);
    assert_eq!(w.texts("s1"), ["stored before parse"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 1);
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
            Err(std::io::Error::other("synthetic interrupted acquisition"))
        }
    }
    let w = polylogue_testkit::ClaudeHome::new();
    let (engine, _lock) = engine(&w, 64);
    assert!(Captured::stream(&engine.archive, &engine.config.blob_roots, Failed).is_err());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 0);
}

pub(super) fn engine(
    w: &polylogue_testkit::ClaudeHome,
    chunk: u64,
) -> (Engine, polylogue_store::ArchiveLock) {
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let config = Config {
        home: w.home.clone(),
        tool_dirs: Default::default(),
        workers: 3,
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "test-machine".into(),
        host: "test".into(),
        chunk,
        blob_roots: Vec::new(),
    };
    (Engine::new(config, archive).unwrap(), lock)
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
        let report = engine.import(std::slice::from_ref(&path)).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
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
fn zip_members_keep_their_original_hash_and_duplicate_members_share_one_blob() {
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
    zip.finish().unwrap();
    let (mut engine, _lock) = engine(&w, 1024);
    let report = engine.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(
        report.imported,
        [
            path.clone(),
            PathBuf::from(format!("{}!/one.bin", path.display()))
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
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 2);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 2);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 2);
}

#[test]
fn acquisition_reads_bounded_frames_and_keeps_only_a_recognition_prefix() {
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
    let captured =
        Captured::stream(&engine.archive, &engine.config.blob_roots, &mut generated).unwrap();
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
        let captured = Captured::stream(
            &engine.archive,
            &engine.config.blob_roots,
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        let origin = recognize_import(&path, &captured.head).unwrap();
        engine
            .import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap();
    }
    std::fs::remove_file(&path).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.converge().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(w.texts("s1"), ["stored, source gone"]);
    assert_eq!(w.count("SELECT route_rank AS n FROM message"), 2);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.blob"), 1);
}

#[test]
fn replayed_import_does_not_replace_the_native_winner() {
    use polylogue_testkit::{ClaudeHome, claude_user};
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &claude_user("u1", "s1", "native winner"));
    let path = w.dir.path().join("s1.jsonl");
    let original = claude_user("u1", "s1", "imported fallback");
    std::fs::write(&path, &original).unwrap();
    {
        let (mut engine, _lock) = engine(&w, 64);
        engine.converge().unwrap();
        let captured = Captured::stream(
            &engine.archive,
            &engine.config.blob_roots,
            std::fs::File::open(&path).unwrap(),
        )
        .unwrap();
        let origin = recognize_import(&path, &captured.head).unwrap();
        engine
            .import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap();
    }
    std::fs::remove_file(&path).unwrap();
    let (mut engine, _lock) = engine(&w, 64);
    let report = engine.converge().unwrap();
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert_eq!(w.texts("s1"), ["native winner"]);
    assert_eq!(w.count("SELECT route_rank AS n FROM message"), 0);
}

mod documents;

#[test]
fn an_export_signature_wins_over_the_loose_jsonl_fallback() {
    let head = br#"{"resourceSpans":[{"scopeSpans":[{"spans":[]}"#;
    for name in ["/backup/traces.jsonl", "/backup/traces.json"] {
        assert_eq!(
            recognize_import(Path::new(name), head).unwrap().name,
            "otel-genai"
        );
    }
    for (name, head) in [
        (
            "/backup/synthetic.atof.jsonl",
            br#"{"atof_version":"0.1","kind":"mark"}"#.as_slice(),
        ),
        (
            "/backup/verification_evidence.db.jsonl",
            br#"{"schema":{}}"#.as_slice(),
        ),
    ] {
        assert_eq!(
            recognize_import(Path::new(name), head).unwrap().name,
            "hermes"
        );
    }
    let claude = br#"{"type":"user","sessionId":"s1"}"#;
    assert_eq!(
        recognize_import(Path::new("/backup/s1.jsonl"), claude)
            .unwrap()
            .name,
        "claude-code"
    );
}

mod replay;

mod assets;
mod stored_bytes;

mod containers;
