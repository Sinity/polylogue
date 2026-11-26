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
fn a_trailing_double_star_location_finds_every_file_beneath_it() {
    let home = tempfile::tempdir().unwrap();
    let memory = home.path().join(".claude/memory");
    let nested = memory.join("project");
    fs::create_dir_all(&nested).unwrap();
    let note = nested.join("note.md");
    fs::write(&note, "x").unwrap();
    let top = memory.join("top.md");
    fs::write(&top, "x").unwrap();
    fs::write(memory.join("untouched.md"), "x").unwrap();
    let scanned = |changed: &Path| {
        let (found, unreadable) = scan_paths(
            &config(home.path()),
            &[changed.to_path_buf()].into_iter().collect(),
        );
        assert!(unreadable.is_empty());
        let mut paths: Vec<_> = found.into_iter().map(|f| f.path).collect();
        paths.sort();
        paths
    };
    assert_eq!(scanned(&note), std::slice::from_ref(&note));
    assert_eq!(
        scanned(&memory),
        [note.clone(), top, memory.join("untouched.md")]
    );
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
        clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
        home: home.to_owned(),
        tool_dirs: Default::default(),
        workers: 1,
        memory_budget: Default::default(),
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 1 << 20,
        machine: "m".into(),
        host: "h".into(),
        chunk: 1 << 20,
        live_every: std::time::Duration::from_secs(2),
        reads_sources: true,
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
    let mut names = names;
    names.sort();
    assert_eq!(
        names,
        [
            PathBuf::from("s1/other.jsonl"),
            PathBuf::from("s1/s1.jsonl"),
            PathBuf::from("s1/subagents/agent-a.jsonl"),
            PathBuf::from("s1/subagents/workflows/wf_1/agent-b.jsonl"),
            PathBuf::from("s1/subagents/workflows/wf_1/journal.jsonl"),
            PathBuf::from("s1.jsonl"),
        ]
    );
    assert!(unreadable.is_empty());
    let claude_roots: Vec<_> = roots(&config(home.path()))
        .into_iter()
        .filter(|root| root.path.starts_with(home.path().join(".claude")))
        .collect();
    assert_eq!(
        claude_roots,
        [
            WatchRoot {
                path: home.path().join(".claude"),
                recursive: true
            },
            WatchRoot {
                path: home.path().join(".claude/projects"),
                recursive: true
            },
        ]
    );
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
    let own_roots: Vec<_> = roots(&c)
        .into_iter()
        .filter(|root| root.path.starts_with(&own))
        .collect();
    assert_eq!(
        own_roots,
        [
            WatchRoot {
                path: own.clone(),
                recursive: true
            },
            WatchRoot {
                path: own.join("projects"),
                recursive: true
            },
        ]
    );
}

#[test]
fn classes_partition_files_and_rename_moves_class() {
    let home = tempfile::tempdir().unwrap();
    let root = home.path().join(".codex");
    fs::create_dir_all(root.join("memories")).unwrap();
    fs::write(root.join("memories/guide.md"), "memory").unwrap();
    fs::write(root.join("auth.json"), "secret").unwrap();
    fs::write(root.join("novel.xyz"), "unknown").unwrap();
    let c = config(home.path());
    let first = scan_with_coverage(&c);
    let totals = |scan: &Scan, class| {
        scan.coverage
            .iter()
            .filter(|row| row.root == root && row.class == class)
            .map(|row| row.files)
            .sum::<u64>()
    };
    assert_eq!(
        (
            totals(&first, Class::Collected),
            totals(&first, Class::Ignored),
            totals(&first, Class::Unrecognized)
        ),
        (1, 1, 1)
    );
    assert_eq!(first.coverage.iter().map(|row| row.bytes).sum::<u64>(), 19);
    assert!(
        first
            .coverage
            .iter()
            .all(|row| row.newest_mtime_ms.is_some())
    );
    fs::rename(
        root.join("memories/guide.md"),
        root.join("memories/guide.xyz"),
    )
    .unwrap();
    let second = scan_with_coverage(&c);
    assert_eq!(
        (
            totals(&second, Class::Collected),
            totals(&second, Class::Ignored),
            totals(&second, Class::Unrecognized)
        ),
        (0, 1, 2)
    );
}

#[test]
fn root_follows_tool_variable_and_hermes_profiles_are_full_homes() {
    let home = tempfile::tempdir().unwrap();
    let own = home.path().join("custom-hermes");
    fs::create_dir_all(own.join("profiles/research/memories")).unwrap();
    fs::write(own.join("SOUL.md"), "context").unwrap();
    fs::write(own.join("profiles/research/memories/MEMORY.md"), "memory").unwrap();
    let mut c = config(home.path());
    c.tool_dirs.insert("HERMES_HOME".into(), own.clone());
    let scan = scan_with_coverage(&c);
    assert_eq!(scan.found.len(), 2);
    assert!(scan.found.iter().all(|file| file.path.starts_with(&own)));
    assert_eq!(scan.coverage.iter().map(|row| row.files).sum::<u64>(), 2);
}

#[test]
fn a_symlinked_hermes_profile_is_not_followed_by_the_walk_or_its_coverage() {
    let home = tempfile::tempdir().unwrap();
    let own = home.path().join(".hermes");
    let outside = home.path().join("outside");
    fs::create_dir_all(own.join("profiles")).unwrap();
    fs::create_dir_all(outside.join("memories")).unwrap();
    fs::write(own.join("SOUL.md"), "context").unwrap();
    fs::write(outside.join("memories/MEMORY.md"), "memory").unwrap();
    std::os::unix::fs::symlink(&outside, own.join("profiles/linked")).unwrap();
    let scan = scan_with_coverage(&config(home.path()));
    let found: Vec<_> = scan.found.iter().map(|file| file.path.clone()).collect();
    assert_eq!(found, [own.join("SOUL.md")]);
    let links: Vec<_> = scan
        .coverage
        .iter()
        .filter(|row| row.reason.as_deref() == Some("symbolic link is not followed"))
        .map(|row| (row.root.clone(), row.files))
        .collect();
    assert_eq!(links, [(own, 1)]);
    assert_eq!(scan.coverage.iter().map(|row| row.files).sum::<u64>(), 2);
}

#[test]
fn known_locations_fully_recognized() {
    let home = tempfile::tempdir().unwrap();
    let c = config(home.path());
    let mut written = std::collections::BTreeSet::new();
    for tool in polylogue_sources::locations::TOOLS {
        for pattern in tool.locations {
            let relative = pattern.replace("**", "nested").replace('*', "one");
            let path = c.root(tool.root).join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "{}").unwrap();
            written.insert(path);
        }
    }
    let scan = scan_with_coverage(&c);
    let total = |class| {
        scan.coverage
            .iter()
            .filter(|row| row.class == class)
            .map(|row| row.files)
            .sum::<u64>()
    };
    assert_eq!(total(Class::Collected), written.len() as u64);
    assert_eq!(total(Class::Ignored), 0);
    let unknown: Vec<_> = scan
        .coverage
        .iter()
        .filter(|row| row.class == Class::Unrecognized)
        .collect();
    assert!(unknown.is_empty(), "{unknown:?}");
}

/// A checkout under the Claude Code home that no location names is the
/// user's own project, ignored with its reason by both walks.
#[test]
fn a_user_project_inside_claude_home_is_ignored_with_its_reason() {
    let home = tempfile::tempdir().unwrap();
    let project = home.path().join(".claude/work");
    fs::create_dir_all(project.join(".git")).unwrap();
    fs::write(project.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(project.join("src/main.rs"), "fn main() {}\n").unwrap();
    fs::write(project.join("code.jsonl"), "{}\n").unwrap();
    let c = config(home.path());
    let scan = scan_with_coverage(&c);
    assert!(scan.found.is_empty());
    let mut rows: Vec<_> = scan
        .coverage
        .iter()
        .map(|row| {
            (
                row.class,
                row.pattern.as_str(),
                row.reason.as_deref(),
                row.files,
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        [
            (
                Class::Ignored,
                "**",
                Some("user project under tool home"),
                2
            ),
            (Class::Ignored, ".git/**", Some("git metadata"), 1),
        ]
    );
    let changed = std::collections::BTreeSet::from([project]);
    assert!(scan_paths(&c, &changed).0.is_empty());
}

/// The projects folder being a git repository (a backup of the transcripts)
/// does not make the transcripts a user project: every file a location names
/// is collected, and only the git metadata is ignored.
#[test]
fn transcripts_inside_a_git_repository_are_collected() {
    let home = tempfile::tempdir().unwrap();
    let claude = home.path().join(".claude");
    let projects = claude.join("projects");
    fs::create_dir_all(projects.join(".git")).unwrap();
    fs::write(projects.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
    let session = "0b5e8c1a-4d2f-4e6a-9c3b-1f2e3d4c5b6a";
    let transcripts = [
        projects.join(format!("-home-op-work/{session}.jsonl")),
        projects.join(format!("-home-op-work/{session}/subagents/agent-a1.jsonl")),
        projects.join("-home-op-notes/1c2d3e4f-5a6b-4c7d-8e9f-0a1b2c3d4e5f.jsonl"),
        projects.join("-home-op-work/sessions-index.json"),
    ];
    for path in &transcripts {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "{}\n").unwrap();
    }
    let c = config(home.path());
    let scan = scan_with_coverage(&c);
    let found: Vec<_> = scan.found.iter().map(|f| f.path.clone()).collect();
    let mut expected = transcripts.to_vec();
    expected.sort();
    assert_eq!(found, expected);
    let ignored: Vec<_> = scan
        .coverage
        .iter()
        .filter(|row| row.class != Class::Collected)
        .map(|row| {
            (
                row.class,
                row.pattern.as_str(),
                row.reason.as_deref(),
                row.files,
            )
        })
        .collect();
    assert_eq!(
        ignored,
        [(Class::Ignored, ".git/**", Some("git metadata"), 1)]
    );
    let changed = std::collections::BTreeSet::from([projects.join("-home-op-work")]);
    let (found, _) = scan_paths(&c, &changed);
    let found: Vec<_> = found.into_iter().map(|f| f.path).collect();
    assert_eq!(found, expected[1..].to_vec());
}

#[test]
fn hidden_directories_need_a_named_location() {
    assert!(!matches_location(
        Path::new("brain/one/.private/entry.jsonl"),
        "brain/**"
    ));
    assert!(matches_location(
        Path::new("brain/one/.system_generated/logs/entry.jsonl"),
        "brain/*/.system_generated/logs/*.jsonl"
    ));
    assert!(matches_path(Path::new("cache/.nested/file"), "cache/**"));
}

#[test]
fn completed_walk_distinguishes_accessible_empty_absent_and_unavailable_roots() {
    use polylogue_model::capture::Accessibility;
    let home = polylogue_testkit::Home::new();
    let mut config = crate::test_config();
    config.home = home.home.clone();
    config.tool_dirs.clear();
    std::fs::create_dir_all(home.home.join(".codex")).unwrap();
    // A self-referential symlink is a real metadata failure even as root.
    std::os::unix::fs::symlink(".gemini", home.home.join(".gemini")).unwrap();
    let scan = scan_with_coverage(&config);
    let root = |name: &str| {
        scan.roots
            .iter()
            .find(|root| root.path_bytes == home.home.join(name).as_os_str().as_encoded_bytes())
            .unwrap()
    };
    assert_eq!(root(".codex").status, Accessibility::Accessible);
    assert!(root(".codex").entries.is_empty());
    assert_eq!(root(".gemini").status, Accessibility::Unavailable);
    assert_eq!(
        root(".gemini").unavailable_paths,
        [home
            .home
            .join(".gemini")
            .as_os_str()
            .as_encoded_bytes()
            .to_vec()]
    );
    assert_eq!(root(".grok").status, Accessibility::Absent);
}

#[test]
fn unknown_directories_share_a_bounded_bucket_and_links_are_explicitly_ignored() {
    let home = polylogue_testkit::Home::new();
    let mut config = crate::test_config();
    config.home = home.home.clone();
    config.tool_dirs.clear();
    let root = home.home.join(".codex");
    std::fs::create_dir_all(root.join("sessions")).unwrap();
    for i in 0..20 {
        let path = root.join(format!("unknown-{i}/file.dat"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"unknown").unwrap();
    }
    let external = home.dir.path().join("external");
    std::fs::create_dir(&external).unwrap();
    std::fs::write(external.join("rollout-outside.jsonl"), b"{}\n").unwrap();
    let directory_link = root.join("sessions/linked");
    let file_link = root.join("sessions/rollout-linked.jsonl");
    std::os::unix::fs::symlink(&external, &directory_link).unwrap();
    std::os::unix::fs::symlink(external.join("rollout-outside.jsonl"), &file_link).unwrap();
    let scan = scan_with_coverage(&config);
    let unknown = scan
        .coverage
        .iter()
        .filter(|row| row.root == root && row.class == Class::Unrecognized)
        .collect::<Vec<_>>();
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].files, 20);
    assert_eq!(unknown[0].bytes, 140);
    let ignored = scan
        .coverage
        .iter()
        .filter(|row| {
            row.root == root
                && row.class == Class::Ignored
                && row.reason.as_deref() == Some("symbolic link is not followed")
        })
        .collect::<Vec<_>>();
    assert_eq!(ignored.len(), 1);
    assert_eq!(ignored[0].files, 2);
    assert!(
        scan.found
            .iter()
            .all(|f| !f.path.starts_with(&directory_link) && f.path != file_link)
    );
    let (found, _) = scan_paths(
        &config,
        &std::collections::BTreeSet::from([directory_link, file_link]),
    );
    assert!(found.is_empty());
}
