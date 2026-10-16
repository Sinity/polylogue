use super::*;

#[test]
fn configured_hash_change_refuses_before_retained_source_effects() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(home.join("sessions")).unwrap();
    let path = home.join("sessions/rollout-owned.jsonl");
    std::fs::write(&path, b"{}\n").unwrap();
    let config = polylogue_store::configured_codex::Config {
        home,
        machine: "synthetic-machine".into(),
        host: "synthetic-host".into(),
        rollouts: "sessions".into(),
        catalog: "state_5.sqlite".into(),
        history: "history.jsonl".into(),
    };
    let plan =
        polylogue_store::configured_codex::discover(&config, &AtomicBool::new(false)).unwrap();
    let candidate = plan.candidates.into_iter().next().unwrap();
    let (hash, _) = candidate.fingerprint(&AtomicBool::new(false), 3).unwrap();
    let archive = temp.path().join("archive");
    let writer = Writer::open(&archive).unwrap();
    let before = writer.status().unwrap().write_counter;
    let reader = writer.read_handle().unwrap();
    let owner = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    std::fs::write(&path, b"[]\n").unwrap();
    let result = owner
        .handle()
        .try_configured_import(
            request(&path, "01900000-0000-7000-8000-000000000071"),
            candidate,
            hash,
            3,
        )
        .unwrap()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert!(matches!(result, Err(Error::InvalidEvidence(_))));
    assert_eq!(reader.status().unwrap().write_counter, before);
    owner.shutdown().unwrap();
}
#[test]
fn configured_known_content_does_not_bypass_request_conflict() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(home.join("sessions")).unwrap();
    let path = home.join("sessions/rollout-owned.jsonl");
    std::fs::write(&path, b"{}\n").unwrap();
    std::fs::write(home.join("history.jsonl"), b"").unwrap();
    let config = polylogue_store::configured_codex::Config {
        home: home.clone(),
        machine: "synthetic-machine".into(),
        host: "synthetic-host".into(),
        rollouts: "sessions".into(),
        catalog: "state_5.sqlite".into(),
        history: "history.jsonl".into(),
    };
    let stop = AtomicBool::new(false);
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    writer
        .acquire_codex_home(&config, "startup", 1, &stop)
        .unwrap();
    let reader = writer.read_handle().unwrap();
    let owner = Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap();
    let other = home.join("sessions/rollout-other.jsonl");
    std::fs::write(&other, b"[]\n").unwrap();
    let id = "01900000-0000-7000-8000-000000000072";
    import(&owner.handle(), request(&other, id));
    let before = reader.status().unwrap().write_counter;
    let candidate = polylogue_store::configured_codex::discover(&config, &stop)
        .unwrap()
        .candidates
        .into_iter()
        .find(|c| c.path() == path)
        .unwrap();
    let hash = candidate.fingerprint(&stop, 3).unwrap().0;
    let result = owner
        .handle()
        .try_configured_import(request(&path, id), candidate, hash, 3)
        .unwrap()
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    assert!(matches!(
        result,
        Err(Error::Store(StoreError::ClientRequestConflict { .. }))
    ));
    assert_eq!(reader.status().unwrap().write_counter, before);
    owner.shutdown().unwrap();
}
