use super::*;
use crate::{ReadHandle, Writer};
use std::os::unix::fs::symlink;

fn config(root: PathBuf) -> Config {
    Config {
        root,
        machine: "synthetic".into(),
        host: "synthetic".into(),
        acquisition_quantum_bytes: 65536,
    }
}
fn write(root: &Path, path: &str, bytes: &[u8]) {
    let file = root.join(path);
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(file, bytes).unwrap();
}
fn collect(cursor: &mut Cursor, c: &Config, budget: usize) -> (Vec<Location>, Scan) {
    let mut locations = vec![];
    loop {
        let t = cursor
            .next_turn(c, budget, &AtomicBool::new(false))
            .unwrap();
        locations.extend(t.locations);
        if t.scan.cycle_complete {
            return (locations, t.scan);
        }
    }
}
#[test]
fn donor_layout_excludes_nested_copies_and_separates_roles() {
    let cases = [
        ("-repo/main.jsonl", Some(vec!["claude-main", "main"])),
        ("-repo/agent-a.jsonl", Some(vec!["claude-main", "agent-a"])),
        (
            "-repo/owner/subagents/agent-a.ndjson",
            Some(vec!["claude-subagent", "a", "owner"]),
        ),
        (
            "-repo/owner/subagents/workflows/run/agent-a.jsonl",
            Some(vec!["claude-subagent", "a", "owner"]),
        ),
        (".claude/worktrees/-repo/main.jsonl", None),
        ("-repo/.claude/worktrees/main.jsonl", None),
        ("-repo/owner/subagents/nested/agent-a.jsonl", None),
        ("-repo/owner/subagents/workflows/run/journal.jsonl", None),
        ("-repo/memory/MEMORY.md", None),
        ("-repo/sessions-index.json", None),
        ("main.jsonl", None),
        ("-repo/analysis/result.jsonl", None),
    ];
    for (path, expected) in cases {
        let got = match kind(&components(Path::new(path)).unwrap()) {
            Kind::Transcript(c) => Some(c),
            _ => None,
        };
        assert_eq!(
            got,
            expected.map(|v| v.into_iter().map(str::to_owned).collect()),
            "{path}"
        );
    }
}
#[test]
fn five_thousand_files_cross_scan_turns_then_late_project_and_root_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("projects");
    fs::create_dir(&root).unwrap();
    for i in 0..5000 {
        write(
            &root,
            &format!("-repo-{:02}/main-{i:05}.jsonl", i % 7),
            b"{}\n",
        );
    }
    write(&root, "-repo-00/memory/stray.jsonl", b"{}\n");
    write(&root, ".git/nested.jsonl", b"{}\n");
    let c = config(root.clone());
    let mut cursor = Cursor::default();
    let (all, scan) = collect(&mut cursor, &c, 17);
    assert_eq!(all.len(), 5000);
    assert_eq!(scan.observed_transcripts, 5000);
    assert_eq!(scan.state, "accessible");
    write(&root, "-late/later.jsonl", b"{}\n");
    let (later, _) = collect(&mut cursor, &c, 17);
    assert_eq!(later.len(), 5001);
    let first = cursor.next_turn(&c, 1, &AtomicBool::new(false)).unwrap();
    assert!(!first.scan.cycle_complete);
    fs::rename(&root, temp.path().join("old-root")).unwrap();
    fs::create_dir(&root).unwrap();
    write(&root, "-fresh/new.jsonl", b"{}\n");
    let (fresh, scan) = collect(&mut cursor, &c, 1);
    assert_eq!(fresh.len(), 1);
    assert_eq!(fresh[0].context(), ["claude-main", "new"]);
    assert_eq!(scan.observed_transcripts, 1);
}
#[test]
fn descriptor_relative_capture_refuses_symlink_swaps_and_archive_roots() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("projects");
    write(&root, "-repo/main.jsonl", b"inside\n");
    let (locations, _) = collect(&mut Cursor::default(), &config(root.clone()), 8);
    let location = &locations[0];
    let outside = temp.path().join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join("main.jsonl"), b"outside\n").unwrap();
    fs::rename(root.join("-repo"), root.join("held-original")).unwrap();
    symlink(&outside, root.join("-repo")).unwrap();
    assert!(location.open().is_err());
    let archive = temp.path().join("archive");
    let _writer = Writer::open(&archive).unwrap();
    let nested = archive.join("projects");
    fs::create_dir(&nested).unwrap();
    assert!(
        Cursor::default()
            .next_turn(&config(nested), 8, &AtomicBool::new(false))
            .is_err()
    );
}
fn retain(writer: &mut Writer, location: Location, id: &str) -> crate::RetainedSource {
    let prepared = writer
        .blob_capture()
        .unwrap()
        .prepare_claude_location_in_turns(
            location,
            None,
            &AtomicBool::new(false),
            crate::history_pages::turns::Budget::default(),
            |_| Ok(()),
            |_| Ok(()),
            || Ok(vec![]),
        )
        .unwrap();
    writer
        .retain_claude_main(prepared.main, "synthetic", "synthetic", id, 1)
        .unwrap()
}
#[test]
fn declared_agent_context_survives_restart_and_conflicting_main_never_reinterprets_it() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("projects");
    write(&root, "-repo/owner/subagents/agent-a.jsonl", b"{}\n");
    let (mut locations, _) = collect(&mut Cursor::default(), &config(root), 8);
    let location = locations.pop().unwrap();
    let archive = temp.path().join("archive");
    let mut writer = Writer::open(&archive).unwrap();
    let source = retain(&mut writer, location.clone(), "agent-input");
    assert_eq!(source.context, ["claude-subagent", "a", "owner"]);
    let before = writer.status().unwrap().write_counter;
    let refused = writer
        .blob_capture()
        .unwrap()
        .prepare_claude_location_in_turns(
            Location::explicit_main(location.path().to_path_buf()).unwrap(),
            Some(source.clone()),
            &AtomicBool::new(false),
            crate::history_pages::turns::Budget::default(),
            |_| Ok(()),
            |_| Ok(()),
            || Ok(vec![]),
        );
    assert!(matches!(refused, Err(Error::ObservedSourceRefused(_))));
    assert_eq!(writer.status().unwrap().write_counter, before);
    drop(writer);
    let reader = ReadHandle::open(&archive).unwrap();
    assert_eq!(
        reader.source(&source.input.id).unwrap().unwrap().context,
        source.context
    );
    drop(reader);
    let writer = Writer::open(&archive).unwrap();
    writer
        .archive
        .execute(
            "DELETE FROM observed_context WHERE input=?1",
            [source.input.id.clone()],
        )
        .unwrap();
    assert!(
        writer
            .read_handle()
            .unwrap()
            .source(&source.input.id)
            .is_err()
    );
}
#[test]
fn preserving_v3_upgrade_materializes_exact_old_context_and_fences_unaware_writers() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    let main = temp.path().join("legacy-main.jsonl");
    fs::write(&main, b"retained legacy bytes\n").unwrap();
    let mut writer = Writer::open(&archive).unwrap();
    let source = writer
        .acquire_claude_main(
            &main,
            "synthetic",
            "synthetic",
            "legacy",
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
    writer
        .archive
        .execute_batch("DROP INDEX input_claude_tool_result_main; DROP TABLE observed_context; PRAGMA user_version=3")
        .unwrap();
    let before = writer.status().unwrap().write_counter;
    let lineage =
        crate::copy_preparation::lineage(&writer.archive, &archive.join("archive.db")).unwrap();
    drop(writer);
    let preserved = fs::read(archive.join("archive.db")).unwrap();
    let mut writer = Writer::open(&archive).unwrap();
    let after = writer.retained_source(&source.input.id).unwrap().unwrap();
    assert_eq!(after.context, ["claude-main", "legacy-main"]);
    assert_eq!(after.input.id, source.input.id);
    assert_eq!(after.binding, source.binding);
    assert_eq!(writer.status().unwrap().write_counter, before + 1);
    assert_eq!(
        crate::copy_preparation::lineage(&writer.archive, &archive.join("archive.db")).unwrap(),
        lineage
    );
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .read_source_range(&source.input.id, 1, source.binding, 0, source.input.length)
            .unwrap(),
        b"retained legacy bytes\n"
    );
    let backups = fs::read_dir(archive.join("migrations"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(backups.len(), 1);
    assert_eq!(fs::read(backups[0].join("archive.db")).unwrap(), preserved);
    assert!(matches!(
        crate::validate_version(&archive, "archive.db", 3, true, false),
        Err(Error::NewerSchema {
            found: 4,
            supported: 3,
            ..
        })
    ));
}

#[test]
fn pending_v3_preparation_refuses_upgrade_without_changing_authority_or_custody() {
    fn snapshot(root: &Path) -> std::collections::BTreeMap<PathBuf, polylogue_model::BlobHash> {
        fn walk(
            root: &Path,
            dir: &Path,
            out: &mut std::collections::BTreeMap<PathBuf, polylogue_model::BlobHash>,
        ) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path == root.join("migrations")
                    || path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .ends_with("-shm")
                {
                    continue;
                }
                if path.is_dir() {
                    walk(root, &path, out);
                } else {
                    out.insert(
                        path.strip_prefix(root).unwrap().to_path_buf(),
                        polylogue_model::BlobHash::of(&fs::read(path).unwrap()),
                    );
                }
            }
        }
        let mut out = std::collections::BTreeMap::new();
        walk(root, root, &mut out);
        out
    }
    for existing_wal in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let archive = temp.path().join("archive");
        let mut writer = Writer::open(&archive).unwrap();
        writer.put_file_blob(b"preserved pending custody").unwrap();
        drop(writer);
        let stop = AtomicBool::new(false);
        drop(
            crate::copy_preparation::Preparation::begin(
                &archive,
                "restore",
                "local",
                "019a1100-0000-7000-8000-000000000001",
                true,
                &stop,
            )
            .unwrap(),
        );
        let conn = crate::open(&archive.join("archive.db"), false, true).unwrap();
        conn.execute_batch(
            "DROP INDEX input_claude_tool_result_main; DROP TABLE observed_context; PRAGMA user_version=3; PRAGMA wal_checkpoint(TRUNCATE)",
        )
        .unwrap();
        let held = if existing_wal {
            // A committed WAL remains authoritative even if main already has
            // the old format header. The refusal must not normalize/drop it.
            conn.execute(
                "UPDATE meta SET value=value||' ' WHERE key='copy_preparation'",
                [],
            )
            .unwrap();
            assert!(archive.join("archive.db-wal").metadata().unwrap().len() > 0);
            Some(conn)
        } else {
            drop(conn);
            None
        };
        let before = snapshot(&archive);
        assert!(matches!(
            Writer::open(&archive),
            Err(Error::CopyPreparationRefused(_))
        ));
        let after = snapshot(&archive);
        for (path, hash) in &before {
            assert_eq!(
                after.get(path),
                Some(hash),
                "pre-existing authoritative file: {path:?}"
            );
        }
        for path in after.keys().filter(|p| !before.contains_key(*p)) {
            assert_eq!(path, Path::new("archive.db-wal"));
            let meta = fs::symlink_metadata(archive.join(path)).unwrap();
            assert!(
                meta.is_file() && meta.len() == 0,
                "only a new empty coordination WAL is allowed"
            );
        }
        drop(held);
    }
}
