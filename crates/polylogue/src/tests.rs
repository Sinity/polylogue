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

/// Each catalog key, set to a value other than its default, changes what its
/// consumer produces; a key nothing reads does not belong in the catalog.
#[test]
fn every_catalog_key_changes_an_output() {
    use polylogue_model::settings::{CATALOG, Current, Owner, Snapshot, Value, current};
    let snapshot = |overrides: &[(&str, &str)]| {
        let mut keys: Vec<&str> = CATALOG
            .iter()
            .map(|declaration| declaration.key)
            .filter(|key| !key.ends_with('*'))
            .collect();
        for (key, _) in overrides {
            if !keys.contains(key) {
                keys.push(key);
            }
        }
        Snapshot::new(keys.into_iter().map(|key| {
            let declaration = polylogue_model::settings::declaration(key).unwrap();
            let stored = overrides
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| *value);
            Current {
                key: key.into(),
                kind: declaration.kind,
                default: declaration.default.map(|text| {
                    polylogue_model::settings::parse(key, text, Owner::Standalone).unwrap()
                }),
                scope: declaration.scope,
                owner: "polylogue".into(),
                value: current(key, stored, Owner::Standalone).unwrap(),
                authored: None,
                conflicts: vec![],
            }
        }))
    };
    // The output each key's consumer produces from a snapshot, and a value
    // that differs from the default. Local embeddings are on where a case
    // reads the provider's other settings.
    type Output = fn(&Snapshot) -> String;
    let cases: &[(&str, &str, &[(&str, &str)], Output)] = &[
        ("browser.actions", "on", &[], |s| {
            format!("{:?}", s.browser_actions())
        }),
        ("drive.ai-studio", "on", &[], |s| {
            // The ai-studio collector runs only while its declared setting is on.
            let collector = polylogue_sources::ORIGINS
                .iter()
                .filter_map(|origin| origin.collector.as_ref())
                .find(|collector| collector.setting == Some("drive.ai-studio"))
                .expect("a collector reads drive.ai-studio");
            let on = matches!(
                s.get(collector.setting.unwrap()),
                Ok(Some(Value::Boolean(true)))
            );
            format!("runs: {on}")
        }),
        ("reading.variants", r#"["translation:pl"]"#, &[], |s| {
            format!("{:?}", s.reading_variants())
        }),
        (
            "signal.quiet_hours",
            r#"{"from":1320,"to":420}"#,
            &[],
            |s| format!("{:?}", s.standalone_attention()),
        ),
        ("signal.interrupts_per_hour", "9", &[], |s| {
            format!("{:?}", s.standalone_attention())
        }),
        ("embed.provider", "local", &[], provider_output),
        (
            "embed.endpoint",
            "http://127.0.0.1:9000/v1",
            &[("embed.provider", "local")],
            provider_output,
        ),
        (
            "embed.model",
            "synthetic-model",
            &[("embed.provider", "local")],
            provider_output,
        ),
        (
            "embed.price_per_mtok",
            "0.5",
            &[("embed.provider", "local")],
            provider_output,
        ),
        (
            "plan.claude-code",
            r#"{"name":"max-20x","origin":"claude-code","cycle_anchor_day":1,"price_nusd":200000000000,"budget":null}"#,
            &[],
            |s| format!("{:?}", s.get("plan.claude-code")),
        ),
    ];
    fn provider_output(snapshot: &Snapshot) -> String {
        format!(
            "{:?}",
            crate::embeddings::Provider::from_settings(
                snapshot,
                &crate::embeddings::Credentials::new(None)
            )
        )
    }
    for declaration in CATALOG {
        let covered = cases.iter().any(|(key, ..)| {
            *key == declaration.key
                || declaration
                    .key
                    .strip_suffix('*')
                    .is_some_and(|prefix| key.starts_with(prefix))
        });
        assert!(covered, "{} has no case", declaration.key);
    }
    for &(key, value, context, output) in cases {
        let before = output(&snapshot(context));
        let mut changed = context.to_vec();
        changed.push((key, value));
        let after = output(&snapshot(&changed));
        assert_ne!(before, after, "{key} changes nothing");
    }
}
