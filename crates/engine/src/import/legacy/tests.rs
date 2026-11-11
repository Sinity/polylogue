use super::*;
fn decoded(member: &package::Member) -> Vec<u8> {
    let mut bytes = Vec::new();
    member
        .bytes
        .reader()
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
}
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
fn archive_format() -> ImportOptions {
    ImportOptions {
        format: Some(crate::import::predecessor::ARCHIVE.into()),
        ..Default::default()
    }
}

#[test]
fn a_predecessor_archive_is_read_only_when_named() {
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("predecessor");
    fixture(&old);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let report = engine.import(std::slice::from_ref(&old)).unwrap();
    assert!(report.imported.is_empty(), "{report:?}");
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
    let empty = w.dir.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let report = engine
        .import_with_options(&[empty], &archive_format())
        .unwrap();
    assert_eq!(report.refused.len(), 1);
    assert!(
        report.refused[0]
            .reason
            .contains("needs source.db and blob/"),
        "{report:?}"
    );
}

#[test]
fn predecessor_blobs_keep_original_fragments_and_join_only_recorded_append_offsets() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    fixture(root.path());
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    assert_eq!(members.len(), 4);
    assert_eq!(decoded(&members[1]), FULL);
    assert_eq!(decoded(&members[2]), APPEND);
    assert_eq!(decoded(&members[3]), [FULL, APPEND].concat());
    assert_eq!(members[3].origin.unwrap().name, "claude-code");
    let db = rusqlite::Connection::open(root.path().join("source.db")).unwrap();
    db.execute(
        "UPDATE raw_sessions SET append_start_offset=1 WHERE raw_id='next'",
        [],
    )
    .unwrap();
    // Only the joined reading is refused; the fragments stay.
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    let refused = members
        .iter()
        .find(|member| member.origin.is_none())
        .unwrap();
    assert!(
        refused
            .path
            .to_string_lossy()
            .contains("raw_sessions/next/")
    );
    assert!(
        refused.refusal.as_deref().unwrap().contains("offsets"),
        "{:?}",
        refused.refusal
    );
    assert_eq!(decoded(&members[2]), APPEND);
}

#[test]
fn an_append_naming_its_base_by_revision_joins_it_and_an_unknown_base_is_refused_alone() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    fixture(root.path());
    let db = rusqlite::Connection::open(root.path().join("source.db")).unwrap();
    db.execute_batch(
        "ALTER TABLE raw_sessions ADD COLUMN source_revision TEXT;
         ALTER TABLE raw_sessions ADD COLUMN predecessor_source_revision TEXT;
         UPDATE raw_sessions SET source_revision = 'rev-full' WHERE raw_id = 'full';
         UPDATE raw_sessions SET predecessor_raw_id = NULL, predecessor_source_revision = 'rev-full' WHERE raw_id = 'next';",
    )
    .unwrap();
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    assert!(members.iter().all(|member| member.origin.is_some()));
    assert_eq!(decoded(&members[3]), [FULL, APPEND].concat());

    db.execute(
        "UPDATE raw_sessions SET predecessor_source_revision = 'rev-gone' WHERE raw_id = 'next'",
        [],
    )
    .unwrap();
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    let refused: Vec<_> = members
        .iter()
        .filter(|member| member.origin.is_none())
        .collect();
    assert_eq!(refused.len(), 1);
    assert_eq!(
        refused[0].refusal.as_deref(),
        Some("it continues revision rev-gone, which no one raw session of the catalog is")
    );
    // The full revision still reads on its own.
    assert_eq!(members[1].origin.unwrap().name, "claude-code");
    assert_eq!(decoded(&members[1]), FULL);
}
#[test]
fn a_later_missing_or_changed_blob_stops_the_import_after_committed_members() {
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
        let error = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap_err();
        assert!(!error.to_string().is_empty());
        // The earlier full revision has already committed and parsed. Failure
        // in a later member never discards that progress.
        assert_eq!(w.texts("legacy"), ["one"]);
        std::fs::write(old.join("blob").join(&hash[..2]).join(&hash[2..]), APPEND).unwrap();
        let report = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap();
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(w.texts("legacy"), ["one", "two"]);
        let repeat = engine
            .import_with_options(&[old], &archive_format())
            .unwrap();
        assert!(repeat.imported.is_empty(), "{repeat:?}");
    }
}

#[test]
fn predecessor_import_parses_native_sessions_and_rebuilds_after_the_old_archive_is_gone() {
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("predecessor");
    fixture(&old);
    {
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        let report = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap();
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

fn blob(root: &Path, bytes: &[u8]) -> String {
    let name = format!("{:x}", Sha256::digest(bytes));
    std::fs::create_dir_all(root.join("blob").join(&name[..2])).unwrap();
    std::fs::write(root.join("blob").join(&name[..2]).join(&name[2..]), bytes).unwrap();
    name
}

#[test]
fn every_catalog_row_and_every_uncited_blob_is_kept() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    fixture(root.path());
    let db = rusqlite::Connection::open(root.path().join("source.db")).unwrap();
    db.execute_batch(
        "CREATE TABLE blob_refs(blob_hash BLOB, ref_id TEXT, ref_type TEXT);
         INSERT INTO blob_refs VALUES (x'00', 'r1', 'attachment');",
    )
    .unwrap();
    let attachment = blob(root.path(), b"an attachment no raw session cites");
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    assert_eq!(members.len(), 5);
    let rows: Vec<serde_json::Value> = decoded(&members[0])
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["table"], "blob_refs");
    assert_eq!(rows[0]["row"]["ref_type"], "attachment");
    assert!(members[0].path.ends_with("source.db/rows.jsonl"));
    let kept = members.last().unwrap();
    assert!(
        kept.path
            .to_string_lossy()
            .ends_with(&format!("!/blob/{attachment}"))
    );
    assert_eq!(kept.origin.unwrap().name, KEPT);
    assert_eq!(decoded(kept), b"an attachment no raw session cites");

    // An uncited blob whose bytes do not hash to its name refuses the import.
    let name = blob(root.path(), b"other bytes");
    std::fs::write(
        root.path().join("blob").join(&name[..2]).join(&name[2..]),
        b"changed",
    )
    .unwrap();
    let error = stage(root.path(), &Archive::at(scratch.path()))
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("differs from its name"), "{error}");
}

#[test]
fn a_raw_session_of_an_unnamed_origin_is_read_by_its_structure_or_kept() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    fixture(root.path());
    let db = rusqlite::Connection::open(root.path().join("source.db")).unwrap();
    let conversation = br#"{"mapping":{},"current_node":null,"conversation_id":"c1","title":"t"}"#;
    let opaque = b"bytes of no declared structure";
    for (id, bytes) in [("chat", &conversation[..]), ("opaque", &opaque[..])] {
        let name = blob(root.path(), bytes);
        let hash = Sha256::digest(bytes);
        assert_eq!(format!("{hash:x}"), name);
        db.execute(
            "INSERT INTO raw_sessions VALUES (?1,'unknown-export','/home/op/inbox/x.json',?2,?3,NULL,'full',NULL,NULL,NULL)",
            rusqlite::params![id, hash.as_slice(), bytes.len() as i64],
        )
        .unwrap();
    }
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    let origin = |id: &str| {
        members
            .iter()
            .find(|member| {
                member
                    .path
                    .to_string_lossy()
                    .contains(&format!("raw_sessions/{id}/"))
            })
            .unwrap()
            .origin
            .unwrap()
            .name
    };
    assert_eq!(origin("chat"), "chatgpt");
    assert_eq!(origin("opaque"), KEPT);
}
