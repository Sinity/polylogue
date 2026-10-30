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
    }
}
