use super::*;

#[test]
fn missing_or_bad_proc_readings_stay_unknown_and_measured_zero_stays_zero() {
    let io = Some("read_bytes: 0\nwrite_bytes: 4096\nwchar: broken\n");
    assert_eq!(io_count(io, "read_bytes:"), Some(0));
    assert_eq!(io_count(io, "write_bytes:"), Some(4096));
    assert_eq!(io_count(io, "wchar:"), None);
    assert_eq!(io_count(io, "missing:"), None);
    assert_eq!(io_count(None, "read_bytes:"), None);
    assert_eq!(peak_memory_mb(Some("VmHWM: 2048 kB\n")), Some(2));
    assert_eq!(peak_memory_mb(Some("VmHWM: 2048 MB\n")), None);
    assert_eq!(peak_memory_mb(Some("VmHWM: broken kB\n")), None);
    assert_eq!(peak_memory_mb(None), None);
}

#[test]
fn io_deltas_exclude_earlier_work_without_guessing_missing_or_reset_counters() {
    assert_eq!(counter_delta(Some(8192), Some(12288)), Some(4096));
    assert_eq!(counter_delta(Some(8192), Some(8192)), Some(0));
    assert_eq!(counter_delta(Some(8192), Some(4096)), None);
    assert_eq!(counter_delta(None, Some(4096)), None);
    assert_eq!(counter_delta(Some(4096), None), None);
}

#[test]
fn process_cpu_counts_survive_comm_parentheses_and_missing_fields_stay_unknown() {
    assert_eq!(
        cpu_ticks(Some("1 (worker ) name) S 1 2 3 4 5 6 7 8 9 10 100 20")),
        Some(120)
    );
    assert_eq!(cpu_ticks(Some("1 (worker) S 1")), None);
    assert_eq!(cpu_ticks(None), None);
}

#[test]
fn completed_build_measurement_pays_its_wal_and_refuses_a_live_reader() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    // A pass ends its first build by installing the built file as `index.db`.
    polylogue_store::fixture::install_empty_rebuild(
        &archive,
        &polylogue_store::Writer::start(&archive).unwrap().sender(),
    )
    .unwrap();
    let writer = polylogue_store::archive::open_write(
        &archive.index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    writer
        .set_db_config(
            rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
            true,
        )
        .unwrap();
    writer
        .execute_batch("CREATE TABLE measured(value INTEGER); INSERT INTO measured VALUES(1)")
        .unwrap();
    let reader = rusqlite::Connection::open(archive.index_db()).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    writer.execute("UPDATE measured SET value=2", []).unwrap();
    assert!(checkpoint_build(&archive).is_err());
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    reader.execute_batch("ROLLBACK").unwrap();
    let databases = checkpoint_build(&archive).unwrap();
    let index = databases
        .iter()
        .find(|database| database.name == "index")
        .unwrap();
    assert!(index.wal_before > 0);
    assert!(databases.iter().all(|database| database.wal_after == 0));
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}

/// The benchmark's sample is drawn from `--list-inputs`. Over a tree with a
/// compressed Codex rollout, a hook spool file, a Claude Code session with
/// its subagent and tool result in a relocated config directory, and an
/// export ZIP, the list is exactly what discovery finds; laid out by its
/// default layouts, the same files are found again in a home with no root
/// variables, which is how a measured build reads the sample.
#[test]
fn sample_candidates_equal_engine_discovery() {
    let directory = tempfile::tempdir().unwrap();
    let home = directory.path().join("home");
    let claude = directory.path().join("elsewhere/claude");
    let codex = directory.path().join("elsewhere/codex");
    let files = [
        (claude.join("projects/-w/s1.jsonl"), "{\"type\":\"user\"}\n"),
        (
            claude.join("projects/-w/s1/subagents/agent-a1.jsonl"),
            "{\"type\":\"user\"}\n",
        ),
        (
            claude.join("projects/-w/s1/tool-results/t1.txt"),
            "output\n",
        ),
        (
            codex.join("sessions/2026/01/02/rollout-2026-01-02T00-00-00-t1.jsonl.zst"),
            "compressed",
        ),
        (
            codex.join("sessions/2026/01/02/rollout-2026-01-02T00-00-00-t2.jsonl"),
            "{}\n",
        ),
        (
            home.join(".local/state/polylogue/hooks/events.jsonl"),
            "{}\n",
        ),
        (home.join("Downloads/chatgpt-export.zip"), "PK\u{3}\u{4}"),
    ];
    for (path, text) in &files {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    let mut config = measurement_config(home.clone());
    config.tool_dirs = [
        ("CLAUDE_CONFIG_DIR".to_owned(), claude.clone()),
        ("CODEX_HOME".to_owned(), codex.clone()),
    ]
    .into();
    let listed = inputs(&config);
    let discovered: Vec<(PathBuf, &str)> = polylogue_engine::collect::scan(&config)
        .0
        .into_iter()
        .map(|found| (found.path, found.origin.name))
        .collect();
    assert_eq!(
        listed
            .iter()
            .map(|input| (input.path.clone(), input.origin))
            .collect::<Vec<_>>(),
        discovered
    );
    let layouts: Vec<&Path> = listed.iter().map(|input| input.layout.as_path()).collect();
    for expected in [
        ".claude/projects/-w/s1.jsonl",
        ".claude/projects/-w/s1/subagents/agent-a1.jsonl",
        ".claude/projects/-w/s1/tool-results/t1.txt",
        ".codex/sessions/2026/01/02/rollout-2026-01-02T00-00-00-t1.jsonl.zst",
        ".local/state/polylogue/hooks/events.jsonl",
    ] {
        assert!(
            layouts.contains(&Path::new(expected)),
            "{expected}: {layouts:?}"
        );
    }
    // An export enters through `polylogue import`, so neither discovery nor
    // the sample holds it.
    assert!(
        !listed
            .iter()
            .any(|input| input.path.ends_with("chatgpt-export.zip"))
    );
    let sample = directory.path().join("sample");
    for input in &listed {
        let to = sample.join(&input.layout);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(&input.path, &to).unwrap();
        assert_eq!(input.bytes, fs::metadata(&to).unwrap().len());
    }
    let mut again: Vec<(PathBuf, &str)> =
        polylogue_engine::collect::scan(&measurement_config(sample.clone()))
            .0
            .into_iter()
            .map(|found| {
                (
                    found.path.strip_prefix(&sample).unwrap().to_owned(),
                    found.origin.name,
                )
            })
            .collect();
    let mut expected: Vec<(PathBuf, &str)> = listed
        .iter()
        .map(|input| (input.layout.clone(), input.origin))
        .collect();
    again.sort();
    expected.sort();
    assert_eq!(again, expected);
}
