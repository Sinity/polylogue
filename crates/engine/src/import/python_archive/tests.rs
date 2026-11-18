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
const FULL:&[u8]=b"{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"python\",\"message\":{\"role\":\"user\",\"content\":\"one\"}}\n";
const APPEND:&[u8]=b"{\"type\":\"assistant\",\"uuid\":\"a1\",\"sessionId\":\"python\",\"message\":{\"role\":\"assistant\",\"content\":\"two\"}}\n";
fn fixture(root: &Path) {
    fixture_with_parts(root, "/home/op/gone/python.jsonl", FULL, APPEND);
}

fn fixture_with_parts(root: &Path, source_path: &str, base: &[u8], tail: &[u8]) {
    std::fs::create_dir_all(root.join("blob")).unwrap();
    let db = rusqlite::Connection::open(root.join("source.db")).unwrap();
    db.execute_batch("CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY,origin TEXT,source_path TEXT,blob_hash BLOB,blob_size INTEGER,capture_mode TEXT,revision_kind TEXT,predecessor_raw_id TEXT,append_start_offset INTEGER,append_end_offset INTEGER)").unwrap();
    for (id, bytes, previous, start) in [
        ("full", base, None, None),
        ("next", tail, Some("full"), Some(base.len() as i64)),
    ] {
        let hash = Sha256::digest(bytes);
        let name = format!("{hash:x}");
        std::fs::create_dir_all(root.join("blob").join(&name[..2])).unwrap();
        std::fs::write(root.join("blob").join(&name[..2]).join(&name[2..]), bytes).unwrap();
        db.execute(
            "INSERT INTO raw_sessions VALUES (?1,'claude-code-session',?2,?3,?4,NULL,?5,?6,?7,?8)",
            rusqlite::params![
                id,
                source_path,
                hash.as_slice(),
                bytes.len() as i64,
                if previous.is_some() { "append" } else { "full" },
                previous,
                start,
                start.map(|n| n + bytes.len() as i64)
            ],
        )
        .unwrap();
    }
}
fn archive_format() -> ImportOptions {
    ImportOptions {
        format: Some(crate::import::python_polylogue::ARCHIVE.into()),
        ..Default::default()
    }
}

#[test]
fn python_archive_is_read_only_when_named() {
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("python-archive");
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
fn python_blobs_join_only_at_recorded_append_offsets() {
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
fn browser_capture_header_can_cross_an_append_boundary() {
    let root = tempfile::tempdir().unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let base = br#"{"capture_"#;
    let tail = br#"id":"synthetic","turns":[]}"#;
    fixture_with_parts(root.path(), "capture.json", base, tail);
    let members = stage(root.path(), &Archive::at(scratch.path()))
        .unwrap()
        .unwrap();
    assert_eq!(members[1].origin.unwrap().name, "claude-code");
    assert_eq!(decoded(&members[1]), base);
    assert_eq!(decoded(&members[2]), tail);
    assert_eq!(members[3].origin.unwrap().name, "browser-capture");
    assert_eq!(
        decoded(&members[3]),
        br#"{"capture_id":"synthetic","turns":[]}"#
    );
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
        let old = w.dir.path().join("python-archive");
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
        assert_eq!(w.texts("python"), ["one"]);
        std::fs::write(old.join("blob").join(&hash[..2]).join(&hash[2..]), APPEND).unwrap();
        let report = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap();
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(w.texts("python"), ["one", "two"]);
        let repeat = engine
            .import_with_options(&[old], &archive_format())
            .unwrap();
        assert!(repeat.imported.is_empty(), "{repeat:?}");
    }
}

#[test]
fn older_archive_import_parses_original_sessions() {
    let w = polylogue_testkit::ClaudeHome::new();
    let old = w.dir.path().join("python-archive");
    fixture(&old);
    {
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        let report = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(report.packages[0].sessions, 1);
        assert_eq!(w.texts("python"), ["one", "two"]);

        let expected: BTreeSet<_> = stage(&old, &engine.archive)
            .unwrap()
            .unwrap()
            .iter()
            .map(|member| {
                super::super::import_identity(
                    &engine.config.machine,
                    &old,
                    None,
                    std::iter::once((member.path.as_path(), member.bytes.hash())),
                )
            })
            .collect();
        let stored: BTreeSet<_> = w
            .read(|reader| reader.query("SELECT DISTINCT import_id FROM import_member", &[]))
            .iter()
            .map(|row| row["import_id"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(stored, expected);

        let hash = format!("{:x}", Sha256::digest(APPEND));
        std::fs::write(
            old.join("blob").join(&hash[..2]).join(&hash[2..]),
            vec![b'x'; APPEND.len()],
        )
        .unwrap();
        let error = engine
            .import_with_options(std::slice::from_ref(&old), &archive_format())
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("differs from its catalog hash or size"),
            "{error}"
        );
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
    assert_eq!(w.texts("python"), ["one", "two"]);
}

#[test]
fn wrapped_messages_keep_wire_sessions_ranges_and_receipts_through_cold_replay() {
    use polylogue_testkit::{ClaudeHome, claude_user, delete_index};
    let w = ClaudeHome::new();
    let root = w.dir.path().join("old");
    let name = "wrapper.jsonl.txt.json";
    let first = claude_user("u1", "wire-one", "first wrapped record");
    let second = claude_user("u2", "wire-two", "second wrapped record");
    let body = format!("{first}{second}");
    fixture_with_parts(&root, name, body.as_bytes(), APPEND);
    rusqlite::Connection::open(root.join("source.db"))
        .unwrap()
        .execute("DELETE FROM raw_sessions WHERE raw_id='next'", [])
        .unwrap();
    let member = format!("{}!/raw_sessions/full/{name}", root.display());
    let input = polylogue_model::BlobHash::of(body.as_bytes()).to_string();
    let assert_reading = || {
        assert_eq!(w.texts("wire-one"), ["first wrapped record"]);
        assert_eq!(w.texts("wire-two"), ["second wrapped record"]);
        assert_eq!(w.count("SELECT count(*) AS n FROM prefix_copy"), 0);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM session WHERE id='claude-code:wrapper'"),
            0
        );
        let rows = w.read(|r| {
            r.query(
                "SELECT m.id,s.source_start,s.source_end FROM message_source s
             JOIN message m ON m.pk=s.message_pk JOIN input_ref i ON i.pk=s.input_pk
             WHERE i.id=?1 ORDER BY s.source_start",
                &[polylogue_store::SqlValue::Text(input.clone())],
            )
        });
        assert_eq!(
            rows,
            [
                serde_json::json!({"id":"claude-code:u1","source_start":0,"source_end":first.len()-1}),
                serde_json::json!({"id":"claude-code:u2","source_start":first.len(),"source_end":body.len()-1})
            ]
        );
    };
    {
        let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
        let report = engine
            .import_with_options(std::slice::from_ref(&root), &archive_format())
            .unwrap();
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert!(report.imported.contains(&PathBuf::from(&member)));
        assert_reading();
        let sibling = w.dir.path().join("wire-one.jsonl");
        std::fs::write(&sibling, &first).unwrap();
        let duplicate = engine.import(&[sibling]).unwrap();
        assert!(duplicate.pass.errors.is_empty(), "{duplicate:?}");
        assert_reading();
        let reader = engine.reads.reader().unwrap();
        let state = reader.input_progress(&input).unwrap().unwrap();
        assert_eq!(
            reader
                .stored_range(&input, state.chain, 0, state.length, &engine.blobs)
                .unwrap(),
            body.as_bytes()
        );
        assert!(
            reader
                .import_receipts(&input, &member, "claude-code", &member)
                .unwrap()
                .iter()
                .any(|(_, r)| r.member == member)
        );
        drop(reader);
        let repeat = engine
            .import_with_options(std::slice::from_ref(&root), &archive_format())
            .unwrap();
        assert!(repeat.pass.errors.is_empty(), "{repeat:?}");
        assert_eq!(repeat.pass.bytes_stored, 0);
        assert!(repeat.already_stored.contains(&PathBuf::from(&member)));
    }
    std::fs::remove_dir_all(&root).unwrap();
    std::fs::remove_file(w.dir.path().join("wire-one.jsonl")).unwrap();
    delete_index(&w.root);
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_reading();
}

#[test]
fn wrapped_names_and_non_message_members_cannot_authorize_a_reading() {
    use polylogue_testkit::ClaudeHome;
    let w = ClaudeHome::new();
    let root = w.dir.path().join("old");
    let name = "wrapper.jsonl.txt.json";
    fixture_with_parts(
        &root,
        name,
        FULL,
        b"{\"type\":\"summary\",\"leafUuid\":\"leaf\",\"summary\":\"summary\"}\n",
    );
    let (mut engine, _lock) = crate::import::tests::engine(&w, 64);
    // The full member is valid. The joined reading ends with an unidentified
    // summary, so none of its records may become a message source.
    let _ = engine.import_with_options(std::slice::from_ref(&root), &archive_format());
    let valid = polylogue_model::BlobHash::of(FULL).to_string();
    let fake = w.dir.path().join("mimic!/raw_sessions/full").join(name);
    std::fs::create_dir_all(fake.parent().unwrap()).unwrap();
    std::fs::write(&fake, FULL).unwrap();
    let bare = engine
        .import_with_options(
            std::slice::from_ref(&fake),
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(bare.refused.len(), 1);
    assert!(bare.imported.is_empty());
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let reader = engine.reads.reader().unwrap();
    let stored = reader.input_progress(&valid).unwrap().unwrap();
    let selected = engine
        .stored_import_origin(
            Some(&reader),
            &fake,
            origin,
            &valid,
            stored.chain,
            stored.length,
        )
        .unwrap();
    assert!((selected.open)(&fake).is_none());
    assert_eq!(w.count("SELECT count(*) AS n FROM message_source"), 1);
}
#[test]
fn a_repeat_confirms_members_without_staging_and_a_new_author_needs_admission() {
    use polylogue_store::fault::{self, Action};
    let home = polylogue_testkit::ClaudeHome::new();
    let old = home.dir.path().join("catalog");
    fixture(&old);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let root = std::slice::from_ref(&old);
    let first = engine.import_with_options(root, &archive_format()).unwrap();
    assert_eq!(first.imported.len(), 4);
    let commits = || home.count("SELECT max(commit_seq) AS n FROM a.observation");
    let before = commits();
    fault::arm(
        "python_archive.prepare_raw",
        &format!("{}!/raw_sessions/", old.display()),
        Action::Fail,
        0,
        1,
    );
    let repeat = engine.import_with_options(root, &archive_format()).unwrap();
    assert_eq!(repeat.already_stored, first.imported);
    assert_eq!(repeat.pass.bytes_stored, 0);
    assert_eq!(commits(), before);
    let options = ImportOptions {
        authorship: Some(polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Script("second-importer".into()),
            via: polylogue_model::authorship::Channel::Import,
        }),
        ..archive_format()
    };
    let error = engine.import_with_options(root, &options).unwrap_err();
    assert!(
        error.to_string().contains("python_archive.prepare_raw"),
        "{error}"
    );
    let second = engine.import_with_options(root, &options).unwrap();
    assert_eq!(second.already_stored, first.imported);
    assert!(commits() > before);
    assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 4);
    assert_eq!(home.count("SELECT count(*) AS n FROM import_member"), 8);
    let moved = home.dir.path().join("second-path");
    std::fs::rename(&old, &moved).unwrap();
    let relocated = engine
        .import_with_options(&[moved], &archive_format())
        .unwrap();
    assert_eq!(relocated.already_stored.len(), 4);
    assert_eq!(home.count("SELECT count(*) AS n FROM import_member"), 12);
}

#[test]
fn a_changed_append_parent_is_read_again_and_a_conflicting_format_is_refused() {
    let home = polylogue_testkit::ClaudeHome::new();
    let old = home.dir.path().join("catalog");
    fixture(&old);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let run = |engine: &mut Engine| {
        engine.import_with_options(std::slice::from_ref(&old), &archive_format())
    };
    run(&mut engine).unwrap();
    let other = String::from_utf8(FULL.to_vec())
        .unwrap()
        .replace("u1", "u2")
        .replace("one", "new");
    blob(&old, other.as_bytes());
    let db = rusqlite::Connection::open(old.join("source.db")).unwrap();
    db.execute(
        "INSERT INTO raw_sessions SELECT 'other',origin,source_path,?1,blob_size,capture_mode,revision_kind,NULL,NULL,NULL FROM raw_sessions WHERE raw_id='full'",
        [Sha256::digest(other.as_bytes()).as_slice()],
    ).unwrap();
    db.execute(
        "UPDATE raw_sessions SET predecessor_raw_id='other' WHERE raw_id='next'",
        [],
    )
    .unwrap();
    let changed = run(&mut engine).unwrap();
    assert!(
        changed
            .imported
            .iter()
            .any(|path| path.ends_with("raw_sessions/next/python.jsonl")),
        "{changed:?}"
    );
    {
        let reader = engine.reads.reader().unwrap();
        for bytes in [
            [FULL, APPEND].concat(),
            [other.as_bytes(), APPEND].concat(),
            APPEND.to_vec(),
        ] {
            let input = polylogue_model::BlobHash::of(&bytes).to_string();
            let progress = reader.input_progress(&input).unwrap().unwrap();
            assert_eq!(
                reader
                    .stored_range(&input, progress.chain, 0, progress.length, &engine.blobs)
                    .unwrap(),
                bytes
            );
        }
    }
    db.execute(
        "UPDATE raw_sessions SET append_start_offset=0 WHERE raw_id='next'",
        [],
    )
    .unwrap();
    let refused = run(&mut engine).unwrap();
    assert_eq!(refused.refused.len(), 1);
    assert!(refused.refused[0].reason.contains("offsets"));
    assert!(
        refused
            .already_stored
            .iter()
            .any(|path| path.ends_with("raw_sessions/next/fragment"))
    );
    db.execute(
        "UPDATE raw_sessions SET origin='codex-session' WHERE raw_id='full'",
        [],
    )
    .unwrap();
    let error = run(&mut engine).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("already parsed as another origin"),
        "{error}"
    );
}

#[test]
fn a_writer_miss_checks_the_source_again_before_admitting_it() {
    let home = polylogue_testkit::ClaudeHome::new();
    let root = home.dir.path().join("catalog");
    fixture(&root);
    let name = format!("{:x}", Sha256::digest(FULL));
    let raw = RawMember {
        path: PathBuf::from(format!(
            "{}!/raw_sessions/full/python.jsonl",
            root.display()
        )),
        parts: vec![Blob {
            hash: Sha256::digest(FULL).into(),
            size: FULL.len() as u64,
        }],
        origin: polylogue_sources::origin::by_name("claude-code"),
    };
    let checked = raw.check(&root).unwrap();
    std::fs::write(
        root.join("blob").join(&name[..2]).join(&name[2..]),
        vec![b'x'; FULL.len()],
    )
    .unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let mut pending = vec![Pending::Checked(checked)];
    let error = engine
        .flush_python_members(
            &root,
            &mut pending,
            &archive_format(),
            &mut ImportReport::default(),
            &mut ImportPackage::default(),
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("differs from its catalog hash or size"),
        "{error}"
    );
    assert_eq!(home.count("SELECT count(*) AS n FROM a.revision"), 0);
    assert_eq!(home.count("SELECT count(*) AS n FROM a.observation"), 0);
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
    assert!(members[0].path.ends_with("source.db"));
    assert_eq!(
        decoded(&members[0]),
        std::fs::read(root.path().join("source.db")).unwrap()
    );
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
fn a_native_catalog_reopens_with_its_wal_and_repeats_without_new_revisions() {
    let home = polylogue_testkit::ClaudeHome::new();
    let root = home.dir.path().join("python-archive");
    fixture(&root);
    let main = root.join("source.db");
    let journal = root.join("source.db-wal");
    let held = rusqlite::Connection::open(&main).unwrap();
    held.pragma_update(None, "journal_mode", "wal").unwrap();
    held.pragma_update(None, "wal_autocheckpoint", 0).unwrap();
    let wal_only = polylogue_testkit::claude_user("wal-message", "wal-session", "catalog WAL");
    let hash = Sha256::digest(wal_only.as_bytes());
    blob(&root, wal_only.as_bytes());
    held.execute(
        "INSERT INTO raw_sessions VALUES ('wal-only','claude-code-session','/work/wal-session.jsonl',?1,?2,NULL,'full',NULL,NULL,NULL)",
        rusqlite::params![hash.as_slice(), wal_only.len() as i64],
    )
    .unwrap();
    let database = std::fs::read(&main).unwrap();
    let wal = std::fs::read(&journal).unwrap();
    assert!(!wal.is_empty());

    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    let legacy = b"{\"table\":\"raw_sessions\",\"row\":{\"raw_id\":\"legacy\"}}\n";
    let legacy_path = PathBuf::from(format!("{}!/source.db/rows.jsonl", root.display()));
    let captured = Captured::stream(&engine.archive, legacy.as_slice()).unwrap();
    engine
        .import_captured(
            &legacy_path,
            polylogue_sources::origin::by_name(KEPT).unwrap(),
            captured,
            None,
            &mut ImportReport::default(),
        )
        .unwrap();

    let run = |engine: &mut Engine| {
        let report = engine
            .import_with_options(std::slice::from_ref(&root), &archive_format())
            .unwrap();
        assert!(
            report.pass.errors.is_empty() && report.refused.is_empty(),
            "{report:?}"
        );
        report
    };
    let first = run(&mut engine);
    let database_path = first
        .imported
        .iter()
        .find(|path| path.ends_with("source.db"))
        .unwrap();
    let wal_path = first
        .imported
        .iter()
        .find(|path| path.ends_with("source.db-wal"))
        .unwrap();
    assert_eq!(database_path.parent(), wal_path.parent());
    assert_eq!(home.texts("wal-session"), ["catalog WAL"]);
    let first_revisions = home.count("SELECT count(*) AS n FROM a.revision");
    let repeat = run(&mut engine);
    assert!(repeat.imported.is_empty(), "{repeat:?}");
    assert!(repeat.already_stored.contains(database_path));
    assert!(repeat.already_stored.contains(wal_path));
    assert_eq!(repeat.pass.bytes_stored, 0);
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.revision"),
        first_revisions
    );
    assert_eq!(std::fs::read(&main).unwrap(), database);
    assert_eq!(std::fs::read(&journal).unwrap(), wal);

    let reader = engine.reads.reader().unwrap();
    let restore = |expected: &[u8]| {
        let hash = blake3::hash(expected).to_hex();
        let progress = reader.input_progress(hash.as_ref()).unwrap().unwrap();
        let bytes = reader
            .stored_range(
                hash.as_ref(),
                progress.chain,
                0,
                progress.length,
                &engine.blobs,
            )
            .unwrap();
        assert_eq!(bytes, expected);
        bytes
    };
    restore(legacy);
    let restored = tempfile::tempdir().unwrap();
    let restored_main = restored.path().join("source.db");
    std::fs::write(&restored_main, restore(&database)).unwrap();
    let count = || {
        rusqlite::Connection::open_with_flags(
            &restored_main,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .unwrap()
        .query_row(
            "SELECT count(*) FROM raw_sessions WHERE raw_id='wal-only'",
            [],
            |row| row.get::<_, u64>(0),
        )
        .unwrap()
    };
    assert_eq!(count(), 0);
    std::fs::write(restored.path().join("source.db-wal"), restore(&wal)).unwrap();
    assert_eq!(count(), 1);
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
