use super::*;
const FULL:&[u8]=b"{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"legacy\",\"message\":{\"role\":\"user\",\"content\":\"one\"}}\n";
const APPEND:&[u8]=b"{\"type\":\"assistant\",\"uuid\":\"a1\",\"sessionId\":\"legacy\",\"message\":{\"role\":\"assistant\",\"content\":\"two\"}}\n";
fn fixture(root: &Path) {
    std::fs::create_dir_all(root.join("blob")).unwrap();
    let db = rusqlite::Connection::open(root.join("source.db")).unwrap();
    db.execute_batch("CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY,origin TEXT,source_path TEXT,blob_hash BLOB,blob_size INTEGER,capture_mode TEXT,revision_kind TEXT,predecessor_raw_id TEXT,append_start_offset INTEGER,append_end_offset INTEGER)").unwrap();
    for (id, bytes, predecessor, start) in [
        ("full", FULL, None, None),
        ("next", APPEND, Some("full"), Some(FULL.len() as i64)),
    ] {
        let hash = Sha256::digest(bytes);
        let name = format!("{hash:x}");
        std::fs::create_dir_all(root.join("blob").join(&name[..2])).unwrap();
        std::fs::write(root.join("blob").join(&name[..2]).join(&name[2..]), bytes).unwrap();
        db.execute("INSERT INTO raw_sessions VALUES (?1,'claude-code-session','/home/op/gone/legacy.jsonl',?2,?3,NULL,?4,?5,?6,?7)",rusqlite::params![id,hash.as_slice(),bytes.len() as i64,if predecessor.is_some(){"append"}else{"full"},predecessor,start,start.map(|n|n+bytes.len() as i64)]).unwrap();
    }
}
#[test]
fn predecessor_blobs_keep_original_fragments_and_join_only_recorded_append_offsets() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    fixture(root.path());
    let members = stage(root.path(), scratch.path()).unwrap().unwrap();
    assert_eq!(members.len(), 4);
    assert_eq!(std::fs::read(&members[1].bytes).unwrap(), FULL);
    assert_eq!(std::fs::read(&members[2].bytes).unwrap(), APPEND);
    assert_eq!(
        std::fs::read(&members[3].bytes).unwrap(),
        [FULL, APPEND].concat()
    );
    assert_eq!(members[3].origin.unwrap().name, "claude-code");
    let db = rusqlite::Connection::open(root.path().join("source.db")).unwrap();
    db.execute(
        "UPDATE raw_sessions SET append_start_offset=1 WHERE raw_id='next'",
        [],
    )
    .unwrap();
    assert!(
        stage(root.path(), scratch.path())
            .err()
            .unwrap()
            .to_string()
            .contains("offsets")
    );
}
#[test]
fn missing_or_changed_legacy_blob_refuses_the_whole_import_without_archive_rows() {
    for missing in [false, true] {
        let w = polylogue_testkit::ClaudeHome::new();
        let old = w.dir.path().join("predecessor");
        fixture(&old);
        let hash = format!("{:x}", Sha256::digest(APPEND));
        let path = old.join("blob").join(&hash[..2]).join(&hash[2..]);
        if missing {
            std::fs::remove_file(path).unwrap();
        } else {
            std::fs::write(path, vec![b'x'; APPEND.len()]).unwrap();
        }
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        let report = engine.import(&[old]).unwrap();
        assert_eq!(report.refused.len(), 1);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
    }
}

#[test]
fn predecessor_import_parses_native_sessions_and_rebuilds_after_the_old_archive_is_gone() {
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("predecessor");
    fixture(&old);
    {
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        let report = engine.import(std::slice::from_ref(&old)).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(w.texts("legacy"), ["one", "two"]);
        let id = polylogue_model::BlobHash::of(APPEND).to_string();
        let reader = engine.reads.reader().unwrap();
        let progress = reader.input_progress(&id).unwrap().unwrap();
        assert_eq!(
            reader
                .stored_range(&id, progress.chain, 0, progress.length, &engine.blobs)
                .unwrap(),
            APPEND
        );
    }
    std::fs::remove_dir_all(&old).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("legacy"), ["one", "two"]);
}
