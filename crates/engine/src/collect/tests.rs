use super::*;

#[test]
fn read_access_does_not_wake_collection_but_changes_and_unknown_events_do() {
    use notify::EventKind;
    use notify::event::{AccessKind, AccessMode, CreateKind, ModifyKind, RemoveKind};
    for kind in [
        AccessKind::Read,
        AccessKind::Open(AccessMode::Any),
        AccessKind::Open(AccessMode::Read),
        AccessKind::Open(AccessMode::Write),
        AccessKind::Close(AccessMode::Read),
    ] {
        assert!(!requires_scan(&notify::Event::new(EventKind::Access(kind))));
    }
    for kind in [
        EventKind::Create(CreateKind::File),
        EventKind::Modify(ModifyKind::Any),
        EventKind::Remove(RemoveKind::File),
        EventKind::Access(AccessKind::Close(AccessMode::Write)),
        EventKind::Access(AccessKind::Close(AccessMode::Any)),
        EventKind::Any,
        EventKind::Other,
    ] {
        assert!(requires_scan(&notify::Event::new(kind)));
    }
}

#[test]
fn a_path_ends_in_a_layout() {
    let grok = Path::new("/x/.grok/sessions/%2Fhome/01a0/chat_history.jsonl");
    assert!(ends_in(grok, ".grok/sessions/*/*/chat_history.jsonl"));
    assert!(!ends_in(grok, ".claude/projects/*/*.jsonl"));
    let deep = Path::new("/c/.codex/sessions/2026/09/01/r.jsonl");
    assert!(ends_in(deep, "sessions/**/*.jsonl"));
    assert!(!ends_in(Path::new("/x/r.jsonl"), "sessions/**/*.jsonl"));
}

#[test]
fn wildcards_match_within_a_segment() {
    assert!(wildcard("*.jsonl", "a.jsonl"));
    assert!(wildcard("agent-*.jsonl", "agent-x1.jsonl"));
    assert!(!wildcard("agent-*.jsonl", "x.jsonl"));
    assert!(wildcard("*", "anything"));
    assert!(wildcard("a*b*c", "aXbYc"));
    assert!(!wildcard("a*b*c", "aXcYb"));
}

#[test]
fn changed_directories_find_files_written_before_the_watch_was_added() {
    let home = tempfile::tempdir().unwrap();
    let day = home.path().join(".codex/sessions/2026/10/27");
    fs::create_dir_all(&day).unwrap();
    let rollout = day.join("rollout-new.jsonl");
    fs::write(&rollout, "{}\n").unwrap();
    let sibling = home.path().join(".codex/sessions/2026/10/26");
    fs::create_dir_all(&sibling).unwrap();
    fs::write(sibling.join("rollout-older.jsonl"), "{}\n").unwrap();
    for changed in [day, rollout.clone()] {
        let (found, unreadable) =
            scan_paths(&config(home.path()), &[changed].into_iter().collect());
        assert!(unreadable.is_empty());
        assert_eq!(
            found.iter().map(|f| &f.path).collect::<Vec<_>>(),
            [&rollout]
        );
    }
}

#[test]
fn database_sidecars_name_the_database_for_its_capture_scheduler() {
    let home = tempfile::tempdir().unwrap();
    let config = config(home.path());
    let db = home.path().join(".codex/state_5.sqlite");
    for suffix in ["", "-wal", "-shm", "-journal"] {
        assert_eq!(
            database_path(
                &config,
                &db.with_file_name(format!("state_5.sqlite{suffix}"))
            ),
            Some(db.clone())
        );
    }
    assert_eq!(
        database_path(&config, &home.path().join(".codex/sessions/rollout.jsonl")),
        None
    );
}

fn config(home: &Path) -> Config {
    Config {
        home: home.to_owned(),
        tool_dirs: Default::default(),
        workers: 1,
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "m".into(),
        host: "h".into(),
        chunk: 1 << 20,
        rescan_every: std::time::Duration::from_secs(60),
        live_every: std::time::Duration::from_secs(2),
    }
}

#[test]
fn scans_exactly_the_claude_code_layout() {
    let home = tempfile::tempdir().unwrap();
    let p = home.path().join(".claude/projects/-proj");
    fs::create_dir_all(p.join("s1/subagents/workflows/wf_1")).unwrap();
    fs::write(p.join("s1.jsonl"), "{}\n").unwrap();
    fs::write(p.join("s1/s1.jsonl"), "{}\n").unwrap();
    fs::write(p.join("s1/subagents/agent-a.jsonl"), "{}\n").unwrap();
    fs::write(p.join("s1/subagents/workflows/wf_1/agent-b.jsonl"), "{}\n").unwrap();
    fs::write(p.join("s1/subagents/workflows/wf_1/journal.jsonl"), "{}\n").unwrap();
    fs::write(p.join("notes.txt"), "x").unwrap();
    fs::write(p.join("s1/subagents/other.jsonl"), "{}\n").unwrap();
    fs::write(p.join("s1/other.jsonl"), "{}\n").unwrap();
    let (found, unreadable) = scan(&config(home.path()));
    let names: Vec<_> = found
        .iter()
        .map(|f| f.path.strip_prefix(&p).unwrap().to_owned())
        .collect();
    let recognized: Vec<_> = found
        .iter()
        .filter(|f| recognize(f))
        .map(|f| f.path.strip_prefix(&p).unwrap().to_owned())
        .collect();
    assert_eq!(
        recognized,
        [
            PathBuf::from("s1/s1.jsonl"),
            PathBuf::from("s1/subagents/agent-a.jsonl"),
            PathBuf::from("s1/subagents/workflows/wf_1/agent-b.jsonl"),
            PathBuf::from("s1/subagents/workflows/wf_1/journal.jsonl"),
            PathBuf::from("s1.jsonl"),
        ]
    );
    assert!(names.contains(&PathBuf::from("s1/other.jsonl")));
    assert!(unreadable.is_empty());
    assert!(roots(&config(home.path())).contains(&WatchRoot {
        path: home.path().join(".claude/projects"),
        recursive: true
    }));
    assert!(roots(&config(home.path())).contains(&WatchRoot {
        path: home.path().to_path_buf(),
        recursive: false
    }));
}

#[test]
fn a_tool_directory_variable_moves_the_root() {
    let home = tempfile::tempdir().unwrap();
    let own = home.path().join(".config/claude");
    fs::create_dir_all(own.join("projects/-proj")).unwrap();
    fs::write(own.join("projects/-proj/s1.jsonl"), "{}\n").unwrap();
    let mut c = config(home.path());
    c.tool_dirs.insert("CLAUDE_CONFIG_DIR".into(), own.clone());
    let (found, _) = scan(&c);
    assert_eq!(found.len(), 1);
    assert!(found[0].path.starts_with(&own));
    assert!(roots(&c).contains(&WatchRoot {
        path: own.join("projects"),
        recursive: true
    }));
}
