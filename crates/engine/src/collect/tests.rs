use super::*;

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
        bytes_in_flight: 1 << 20,
        machine: "m".into(),
        host: "h".into(),
        chunk: 1 << 20,
        blob_roots: Vec::new(),
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
    assert_eq!(
        roots(&config(home.path())),
        [home.path().join(".claude/projects")]
    );
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
    assert_eq!(roots(&c), [own.join("projects")]);
}
