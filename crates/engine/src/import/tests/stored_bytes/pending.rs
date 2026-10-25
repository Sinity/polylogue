use super::*;
use polylogue_store::{
    StoreError,
    fault::{self, Action},
    staging::PendingState,
};

#[test]
fn a_stop_before_or_after_installing_keeps_the_exact_bytes_across_restart() {
    for point in ["import.install", "import.installed"] {
        let world = ClaudeHome::new();
        let path = world.dir.path().join("manual.jsonl");
        let original = claude_user(
            "pending-user",
            "pending-session",
            &format!("synthetic capture at {point}"),
        );
        std::fs::write(&path, &original).unwrap();
        let hash = hashed_blobs::Hash::of(original.as_bytes());
        let id = hash.hex();
        let physically_installed = point != "import.install";
        {
            let (mut engine, _lock) = engine(&world, 64);
            fault::arm(point, &id, Action::Fail, 0, 1);
            let report = engine.import(std::slice::from_ref(&path)).unwrap();
            assert!(!report.pass.errors.is_empty(), "{point}");
            assert_eq!(engine.blobs.contains(&hash), physically_installed);
            let reader = engine.reads.reader().unwrap();
            let stored = reader.input_progress(&id).unwrap().unwrap();
            assert!(matches!(
                reader.stored_chain(&id, stored.chain, stored.length),
                Err(StoreError::InputPending { .. })
            ));
            let status = reader.status().unwrap();
            assert_eq!(status.pending_inputs.len(), 1);
            assert_eq!(status.pending_inputs[0].state, PendingState::Writing);
            assert_eq!(status.stored_bytes, 0);
        }
        // The provider file now holds other bytes; only the stored hash may
        // finish the import, never this replacement.
        std::fs::write(
            &path,
            claude_user(
                "replacement-user",
                "replacement-session",
                "synthetic replacement",
            ),
        )
        .unwrap();
        let (mut engine, _lock) = engine(&world, 64);
        let pass = engine.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        if physically_installed {
            assert!(pass.pending_inputs.is_empty());
            assert_eq!(
                world.texts("pending-session"),
                [format!("synthetic capture at {point}")]
            );
            assert_eq!(
                world.count("SELECT count(*) AS n FROM a.blob WHERE state='present'"),
                1
            );
        } else {
            assert_eq!(pass.pending_inputs.len(), 1);
            assert_eq!(pass.pending_inputs[0].state, PendingState::Missing);
            assert_eq!(world.count("SELECT count(*) AS n FROM message"), 0);
        }
        let replacement = engine.import(std::slice::from_ref(&path)).unwrap();
        assert!(
            replacement.pass.errors.is_empty(),
            "{:?}",
            replacement.pass.errors
        );
        assert_eq!(
            world.texts("replacement-session"),
            ["synthetic replacement"]
        );
        assert_eq!(engine.blobs.contains(&hash), physically_installed);
    }
}

#[test]
fn whole_input_excision_removes_bytes_never_installed_and_refuses_their_retry() {
    let world = ClaudeHome::new();
    let path = world.dir.path().join("manual.jsonl");
    let original = claude_user(
        "missing-user",
        "missing-session",
        "synthetic bytes never installed",
    );
    std::fs::write(&path, &original).unwrap();
    let id = blake3::hash(original.as_bytes()).to_hex().to_string();
    let (mut engine, _lock) = engine(&world, 64);
    fault::arm("import.install", &id, Action::Fail, 0, 1);
    assert!(
        !engine
            .import(std::slice::from_ref(&path))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    engine
        .excise(ExciseRequest {
            id: "remove-bytes-never-installed".into(),
            scope: ExciseScope::Input(id.clone()),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    let retry = engine.import(&[path]).unwrap();
    assert!(
        retry
            .pass
            .errors
            .iter()
            .any(|error| error.contains("excluded"))
    );
    assert!(
        !engine
            .blobs
            .contains(&hashed_blobs::Hash::of(original.as_bytes()))
    );
    assert_eq!(world.count("SELECT count(*) AS n FROM a.revision"), 0);
}
