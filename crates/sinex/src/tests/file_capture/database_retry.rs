use super::*;

#[test]
fn an_unchanged_database_read_after_restart_keeps_the_original_pending_capture() {
    let home = ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    std::fs::copy(
        polylogue_testkit::morrow_dir().join("codex/state_5.sqlite"),
        &db,
    )
    .unwrap();
    let (mut engine, archive, _lock, sdk) = open(&home);
    let first = engine.run_pass().unwrap();
    assert!(first.capture_pending, "{first:?}");
    let original = sdk.pending_captures().unwrap();
    assert!(!original.is_empty());
    for pending in &original {
        let Capture::Input(input) = &pending.captured else {
            panic!("a database capture")
        };
        assert_eq!(input.rows.origin, "codex-database");
        assert!(matches!(
            input.generation,
            Some(ObservedState::Database { .. })
        ));
    }
    drop(engine);
    for _ in 0..2 {
        // A new reader cannot recall this read from the confirmed index yet.
        // It takes a new real database transaction and a new custody nonce.
        let mut engine = Engine::on_sinex(
            home.config(CHUNK),
            archive.clone(),
            sdk.clone(),
            polylogue_testkit::stopped_attempt(),
        )
        .unwrap();
        let repeated = engine.run_pass().unwrap();
        assert!(repeated.capture_pending, "{repeated:?}");
        assert!(
            repeated.inputs_seen > 0,
            "the database must be read: {repeated:?}"
        );
        assert!(
            repeated
                .errors
                .iter()
                .all(|error| error.starts_with("confirmed stream:")
                    && error.contains("Connection refused")),
            "{repeated:?}"
        );
        // This compares original generation, nonce, observation/revision IDs,
        // original body hash, group identity and complete source coordinates.
        assert_eq!(sdk.pending_captures().unwrap(), original);
        drop(engine);
    }
}
