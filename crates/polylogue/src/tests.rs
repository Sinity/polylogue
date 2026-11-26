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
    assert!(super::keeps_sigpipe_ignored(&daemon));
    assert!(super::cli::needs_engine(
        &daemon,
        &super::daemon::service::ArchiveMode::Standalone
    ));
    let tour = super::cli::command()
        .try_get_matches_from(["polylogue", "demo", "/synthetic/tour"])
        .unwrap();
    assert!(super::cli::needs_engine(
        &tour,
        &super::daemon::service::ArchiveMode::Standalone
    ));
    assert!(!super::keeps_sigpipe_ignored(&tour));
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
    assert!(super::cli::run_line::native_host(&[origin.into()]));
    assert!(!super::cli::run_line::native_host(&[
        origin.into(),
        "extra".into()
    ]));
    let (overrides, words) = super::cli::run_line::deployment_arguments(&[
        "--archive".into(),
        "/synthetic/archive".into(),
        "--config".into(),
        "/synthetic/config".into(),
        origin.into(),
    ])
    .unwrap();
    assert!(super::cli::run_line::native_host(&words));
    assert_eq!(
        overrides.archive.unwrap(),
        std::path::Path::new("/synthetic/archive")
    );
    assert_eq!(
        overrides.config_file.unwrap(),
        std::path::Path::new("/synthetic/config")
    );
}

/// Each catalog key, set to a value other than its default, changes what its
/// consumer produces; a key nothing reads does not belong in the catalog.
#[test]
fn every_catalog_key_changes_an_output() {
    use polylogue_model::settings::{CATALOG, Current, SettingMode, Snapshot, Value, current};
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
                    polylogue_model::settings::parse(key, text, SettingMode::Standalone).unwrap()
                }),
                scope: declaration.scope,
                supplier: "polylogue".into(),
                value: current(key, stored, SettingMode::Standalone).unwrap(),
                authored: None,
                conflicts: vec![],
            }
        }))
    };
    // The output each key's consumer produces from a snapshot, and a value
    // that differs from the default. Local embeddings are on where a case
    // reads the provider's other settings.
    type Output = fn(&Snapshot) -> String;
    /// A key, a non-default value, other settings the consumer needs, and the
    /// consumer's output.
    type Case = (
        &'static str,
        &'static str,
        &'static [(&'static str, &'static str)],
        Output,
    );
    let cases: &[Case] = &[
        ("browser.actions", "on", &[], |s| {
            format!("{:?}", s.browser_actions())
        }),
        ("drive.client_id", "synthetic-client", &[], |settings| {
            let id = match settings.get("drive.client_id") {
                Ok(Some(Value::Text(value))) => value.to_string(),
                _ => String::new(),
            };
            crate::auth::consent_url(
                "https://accounts.example/authorize",
                &id,
                "http://127.0.0.1:1234/",
                "synthetic-state",
                "synthetic-verifier",
            )
            .unwrap()
            .to_string()
        }),
        ("reading.alternates", r#"["translation:pl"]"#, &[], |s| {
            format!("{:?}", s.reading_alternates())
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
            "embed.calls",
            "on",
            &[("embed.provider", "local")],
            provider_output,
        ),
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

#[test]
fn completion_keeps_valid_deployment_flags_beside_an_invalid_one() {
    let words = |words: &[&str]| -> Vec<std::ffi::OsString> {
        words.iter().map(std::ffi::OsString::from).collect()
    };
    let overrides = super::completion_overrides(&words(&[
        "polylogue",
        "--listen",
        "not-an-address",
        "--archive",
        "/synthetic/archive",
        "rows",
        "--config",
    ]));
    assert_eq!(
        overrides.archive,
        Some(std::path::PathBuf::from("/synthetic/archive"))
    );
    assert_eq!(overrides.listen, None);
    assert_eq!(overrides.config_file, None);
}
