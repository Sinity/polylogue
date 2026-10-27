use super::*;

#[test]
fn changed_paths_visit_only_the_named_file_or_new_subtree() {
    let home = tempfile::tempdir().unwrap();
    let config = config(home.path());
    let project = home.path().join(".claude/projects/p");
    fs::create_dir_all(&project).unwrap();
    let a = project.join("a.jsonl");
    let b = project.join("b.jsonl");
    fs::write(&a, "{}\n").unwrap();
    fs::write(&b, "{}\n").unwrap();
    let paths = |changed: &[PathBuf]| {
        let (found, unreadable) = scan_changed(&config, changed);
        assert!(unreadable.is_empty());
        found.into_iter().map(|f| f.path).collect::<Vec<_>>()
    };
    assert_eq!(paths(std::slice::from_ref(&a)), [a.clone()]);
    assert_eq!(paths(std::slice::from_ref(&project)), [a.clone(), b]);
    let unrelated = home.path().join("notes.jsonl");
    fs::write(&unrelated, "{}\n").unwrap();
    assert!(paths(&[unrelated]).is_empty());
    fs::remove_file(&a).unwrap();
    assert!(paths(&[a]).is_empty());
}

#[test]
fn sqlite_wal_changes_find_the_database_and_deep_directory_moves_find_sessions() {
    let home = tempfile::tempdir().unwrap();
    let config = config(home.path());
    let codex = home.path().join(".codex");
    fs::create_dir_all(codex.join("sessions/2026/10/27")).unwrap();
    let db = codex.join("state_5.sqlite");
    fs::write(&db, b"SQLite format 3\0").unwrap();
    let (found, _) = scan_changed(&config, &[codex.join("state_5.sqlite-wal")]);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, db);
    let session = codex.join("sessions/2026/10/27/rollout-x.jsonl");
    fs::write(&session, "{}\n").unwrap();
    let (found, _) = scan_changed(&config, &[codex.join("sessions/2026")]);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].path, session);
}

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
