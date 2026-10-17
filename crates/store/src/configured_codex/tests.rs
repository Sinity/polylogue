use super::*;
fn config(home: &Path) -> Config {
    Config {
        paginated_history: None,
        paginated_interval_seconds: 300,
        home: home.into(),
        machine: "synthetic".into(),
        host: "synthetic".into(),
        rollouts: rollouts(),
        history: history(),
        catalog: catalog(),
    }
}
#[test]
fn unavailable_catalog_does_not_gate_rollout_and_cancel_does_not_mutate() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    fs::create_dir(&home).unwrap();
    fs::create_dir(home.join("sessions")).unwrap();
    fs::write(home.join("sessions/rollout-one.jsonl"), b"{}\n").unwrap();
    fs::write(home.join("state_5.sqlite"), b"broken catalog").unwrap();
    fs::write(home.join("history.jsonl"), b"").unwrap();
    let mut writer = Writer::open(&root.path().join("archive")).unwrap();
    let before = writer.status().unwrap().write_counter;
    assert!(matches!(
        writer.acquire_codex_home(&config(&home), "cancelled", 1, &AtomicBool::new(true)),
        Err(Error::Cancelled)
    ));
    assert_eq!(writer.status().unwrap().write_counter, before);
    let report = writer
        .acquire_codex_home(&config(&home), "first", 1, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(report.catalog.state, "unavailable");
    assert_eq!(report.catalog.entries, None);
    assert_eq!(report.rollouts.entries, Some(1));
    assert_eq!(report.retained.len(), 2);
    assert_eq!(report.history.state, "accessible_empty");
    assert_eq!(
        writer
            .read_handle()
            .unwrap()
            .codex_prompt_sources()
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn discovery_and_fingerprints_do_not_mutate_and_respect_remaining_bytes() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    fs::create_dir(&home).unwrap();
    fs::create_dir(home.join("sessions")).unwrap();
    let path = home.join("sessions/rollout-later.jsonl");
    fs::write(&path, b"{}\n").unwrap();
    let writer = Writer::open(&root.path().join("archive")).unwrap();
    let before = writer.status().unwrap().write_counter;
    let plan = discover(&config(&home), &AtomicBool::new(false)).unwrap();
    assert_eq!(plan.catalog.state, "unavailable");
    assert_eq!(plan.rollouts.entries, Some(1));
    let candidate = &plan.candidates[0];
    assert!(candidate.fingerprint(&AtomicBool::new(false), 2).is_err());
    assert!(matches!(
        candidate.fingerprint(&AtomicBool::new(true), 3),
        Err(Error::Cancelled)
    ));
    let (hash, length) = candidate.fingerprint(&AtomicBool::new(false), 3).unwrap();
    assert_eq!(length, 3);
    assert_eq!(hash.bytes(), blake3::hash(b"{}\n").as_bytes());
    assert_eq!(writer.status().unwrap().write_counter, before);
    assert_eq!(fs::read(&path).unwrap(), b"{}\n");
    fs::write(&path, b"{ }\n").unwrap();
    assert!(
        writer
            .blob_capture()
            .unwrap()
            .capture_codex_candidate(candidate, 3, &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(writer.status().unwrap().write_counter, before);
}
#[test]
fn reusable_cursor_advances_past_prefix_and_revisits_changed_paths() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path();
    fs::create_dir(home.join("sessions")).unwrap();
    for index in 0..5 {
        fs::write(
            home.join(format!("sessions/rollout-{index}.jsonl")),
            b"{}\n",
        )
        .unwrap();
    }
    let mut cursor = RolloutCursor::default();
    let stop = AtomicBool::new(false);
    let first = cursor.next_turn(&config(home), 2, &stop).unwrap();
    assert_eq!(first.visited_entries, 2);
    assert!(!first.cycle_complete);
    assert_eq!(first.discovery.rollouts.state, "partial");
    assert_eq!(first.discovery.rollouts.entries, None);
    let revisit = first.discovery.candidates[0].path.clone();
    let old = first.discovery.candidates[0]
        .fingerprint(&stop, 3)
        .unwrap()
        .0;
    fs::write(&revisit, b"[]\n").unwrap();
    let late = home.join("sessions/rollout-later.jsonl");
    fs::write(&late, b"{ }\n").unwrap();
    let mut late_seen = false;
    let mut changed_seen = false;
    let mut completed = false;
    for _ in 0..20 {
        let turn = cursor.next_turn(&config(home), 2, &stop).unwrap();
        assert!(turn.visited_entries <= 2);
        if turn.cycle_complete {
            completed = true;
            assert_ne!(turn.discovery.rollouts.state, "accessible_empty");
        }
        for candidate in turn.discovery.candidates {
            if candidate.path == late {
                late_seen = true;
            }
            if candidate.path == revisit {
                assert_ne!(candidate.fingerprint(&stop, 3).unwrap().0, old);
                changed_seen = true;
            }
        }
        if completed && late_seen && changed_seen {
            break;
        }
    }
    assert!(completed && late_seen && changed_seen);
}
#[test]
fn cursor_keeps_catalog_candidates_beside_a_full_filesystem_turn() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path();
    fs::create_dir(home.join("sessions")).unwrap();
    for index in 0..512 {
        fs::write(
            home.join(format!("sessions/rollout-{index:04}.jsonl")),
            b"{}\n",
        )
        .unwrap();
    }
    let catalog = Connection::open(home.join("state_5.sqlite")).unwrap();
    catalog
        .execute_batch("CREATE TABLE threads(id TEXT PRIMARY KEY,rollout_path TEXT NOT NULL)")
        .unwrap();
    catalog
        .execute(
            "INSERT INTO threads VALUES('catalog-only','outside/rollout-catalog.jsonl')",
            [],
        )
        .unwrap();
    drop(catalog);
    let turn = RolloutCursor::default()
        .next_turn(&config(home), 512, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(turn.discovery.candidates.len(), 513);
    assert_eq!(turn.discovery.catalog.entries, Some(1));
    assert!(
        turn.discovery
            .candidates
            .iter()
            .any(|candidate| candidate.path() == home.join("outside/rollout-catalog.jsonl"))
    );
    let startup = discover(&config(home), &AtomicBool::new(false)).unwrap();
    assert_eq!(startup.candidates.len(), 512);
    assert_eq!(startup.deferred_rollout_paths, 1);
}
#[test]
fn unchanged_content_uses_existing_origin_without_new_source_effects() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    fs::create_dir(&home).unwrap();
    fs::create_dir(home.join("sessions")).unwrap();
    let path = home.join("sessions/rollout-owned.jsonl");
    fs::write(&path, b"{}\n").unwrap();
    fs::write(home.join("history.jsonl"), b"").unwrap();
    let config = config(&home);
    let stop = AtomicBool::new(false);
    let mut writer = Writer::open(&root.path().join("archive")).unwrap();
    writer
        .acquire_codex_home(&config, "initial", 1, &stop)
        .unwrap();
    let before = writer.status().unwrap().write_counter;
    let candidate = discover(&config, &stop).unwrap().candidates.pop().unwrap();
    let hash = candidate.fingerprint(&stop, 3).unwrap().0;
    assert!(
        writer
            .known_configured_snapshot(
                &candidate,
                hash,
                "synthetic",
                "synthetic",
                path.to_str().unwrap()
            )
            .unwrap()
            .is_some()
    );
    assert!(
        writer
            .known_configured_snapshot(
                &candidate,
                hash,
                "synthetic",
                "other-host",
                path.to_str().unwrap()
            )
            .unwrap()
            .is_none()
    );
    assert!(
        writer
            .known_configured_snapshot(
                &candidate,
                hash,
                "synthetic",
                "synthetic",
                "/different/path"
            )
            .unwrap()
            .is_none()
    );
    assert_eq!(writer.status().unwrap().write_counter, before);
}
#[test]
fn relative_resolution_refuses_parent_escape_and_symlinks() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path();
    fs::create_dir(home.join("sessions")).unwrap();
    assert_eq!(
        resolve(home, Path::new("sessions/rollout-one.jsonl")).unwrap(),
        home.join("sessions/rollout-one.jsonl")
    );
    assert!(resolve(home, Path::new("../escape")).is_err());
    std::os::unix::fs::symlink(home.join("sessions"), home.join("linked")).unwrap();
    assert!(resolve(home, Path::new("linked/rollout-one.jsonl")).is_err());
}
