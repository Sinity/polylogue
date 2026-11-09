#[test]
fn the_command_tree_is_well_formed() {
    super::cli::command().debug_assert();
}

#[test]
fn deployment_flags_preserve_daemon_signal_policy_and_native_host_selection() {
    let daemon = super::cli::command()
        .try_get_matches_from([
            "polylogue",
            "--archive",
            "/synthetic/archive",
            "--config",
            "/synthetic/config",
            "daemon",
        ])
        .unwrap();
    assert!(super::daemon_command(&daemon));
    assert!(!super::native_host(&daemon));
    assert!(super::cli::needs_engine(
        &daemon,
        &super::daemon::service::ArchiveMode::Standalone
    ));
    let tour = super::cli::command()
        .try_get_matches_from(["polylogue", "demo", "tour", "/synthetic/tour"])
        .unwrap();
    assert!(super::cli::needs_engine(
        &tour,
        &super::daemon::service::ArchiveMode::Standalone
    ));
    #[cfg(feature = "sinex")]
    {
        let blocked = super::daemon::service::ArchiveMode::Blocked {
            configured: Some("primary"),
            path: "/synthetic/sinex.toml".into(),
            error: "invalid configuration".into(),
        };
        assert!(!super::cli::needs_engine(&daemon, &blocked));
        assert!(super::cli::needs_engine(&tour, &blocked));
    }
    let origin = "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/";
    for args in [
        vec!["polylogue", origin],
        vec![
            "polylogue",
            "--archive",
            "/synthetic/archive",
            "--config",
            "/synthetic/config",
            origin,
        ],
    ] {
        let native = super::cli::command().try_get_matches_from(args).unwrap();
        assert!(super::native_host(&native));
        assert!(!super::daemon_command(&native));
        assert!(!super::cli::needs_engine(
            &native,
            &super::daemon::service::ArchiveMode::Standalone
        ));
        if let Some(archive) = super::config::Overrides::from_matches(&native).archive {
            assert_eq!(archive, std::path::Path::new("/synthetic/archive"));
            assert_eq!(
                super::config::Overrides::from_matches(&native)
                    .config_file
                    .unwrap(),
                std::path::Path::new("/synthetic/config")
            );
        }
    }
    for args in [
        vec!["polylogue", "find", origin],
        vec!["polylogue", "read", origin],
        vec!["polylogue", "--archive", origin, "status"],
        vec!["polylogue", origin, "another-word"],
    ] {
        let query = super::cli::command().try_get_matches_from(args).unwrap();
        assert!(!super::native_host(&query));
        assert!(!super::cli::needs_engine(
            &query,
            &super::daemon::service::ArchiveMode::Standalone
        ));
    }
}

#[test]
fn stopped_attempts_hash_executable_bytes_instead_of_paths_or_build_labels() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("first-image");
    let second = dir.path().join("second-image");
    std::fs::write(&first, b"synthetic executable one").unwrap();
    std::fs::copy(&first, &second).unwrap();
    let before = super::attempt_from_executable(std::fs::File::open(&first).unwrap()).unwrap();
    assert_eq!(
        before,
        super::attempt_from_executable(std::fs::File::open(&second).unwrap()).unwrap()
    );
    assert_eq!(
        before.executable,
        polylogue_model::BlobHash::of(b"synthetic executable one")
    );
    assert_eq!(before.parser_settings, None);
    std::fs::write(&first, b"synthetic executable two").unwrap();
    assert_ne!(
        before,
        super::attempt_from_executable(std::fs::File::open(&first).unwrap()).unwrap()
    );
}
