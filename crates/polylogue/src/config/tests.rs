use super::*;

fn process(root: &Path) -> ProcessInputs {
    ProcessInputs {
        environment: EnvSnapshot::from_values([
            ("HOME".into(), root.join("home").into_os_string()),
            (
                "XDG_CONFIG_HOME".into(),
                root.join("config").into_os_string(),
            ),
        ]),
        output: OutputConfig::new(false, false, false),
        clock: WallClock::fixed("2026-03-07T18:00:00Z".parse().unwrap()),
        identity: Some(Identity {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
        }),
        workers: Some(2),
        cwd: Some(root.into()),
    }
}
fn deployment(root: &Path, text: &str) {
    let config = root.join("config/polylogue");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(config.join("polylogue.toml"), text).unwrap();
}
#[test]
fn standalone_day_uses_the_named_zone_across_dst_and_refuses_invalid_zones() {
    let root = tempfile::tempdir().unwrap();
    deployment(
        root.path(),
        "[day]\nzone='America/Los_Angeles'\nstart='06:00:00'\n",
    );
    let startup = Startup::load(&process(root.path()), Overrides::default()).unwrap();
    let day = startup.config.standalone_day().unwrap();
    assert_eq!(day.zone_name(), "America/Los_Angeles");
    let range = day.day("2026-03-07".parse().unwrap()).unwrap();
    assert_eq!(
        range.start,
        "2026-03-07T14:00:00Z".parse::<jiff::Timestamp>().unwrap()
    );
    assert_eq!(
        range.end,
        "2026-03-08T13:00:00Z".parse::<jiff::Timestamp>().unwrap()
    );
    assert_eq!(
        day.logical_day("2026-03-08T12:59:59Z".parse().unwrap())
            .unwrap(),
        "2026-03-07".parse::<jiff::civil::Date>().unwrap()
    );
    deployment(
        root.path(),
        "[day]\nzone='Invalid/Zone'\nstart='06:00:00'\n",
    );
    assert!(Startup::load(&process(root.path()), Overrides::default()).is_err());
}
#[test]
fn configuration_precedence_is_file_then_flags_then_meaningful_environment() {
    let root = tempfile::tempdir().unwrap();
    deployment(
        root.path(),
        "archive='file-archive'\n[http]\nlisten='127.0.0.1:8101'\n",
    );
    let mut process = process(root.path());
    process
        .environment
        .values
        .insert("POLYLOGUE_LISTEN".into(), "127.0.0.1:8103".into());
    // There is no archive environment alias.
    process
        .environment
        .values
        .insert("POLYLOGUE_ARCHIVE".into(), "/unrelated".into());
    let startup = Startup::load(
        &process,
        Overrides {
            archive: Some("flag/../archive".into()),
            listen: Some("127.0.0.1:8102".parse().unwrap()),
            ..Overrides::default()
        },
    )
    .unwrap();
    assert_eq!(startup.config.archive, root.path().join("archive"));
    assert_eq!(
        startup.config.http.listen,
        "127.0.0.1:8103".parse().unwrap()
    );
    assert_eq!(startup.engine.machine, "synthetic-machine");
    assert_eq!(startup.config.requests.now(), process.clock.now().into());
}
#[test]
fn remote_listen_requires_a_loaded_token_and_never_reads_it_again() {
    let root = tempfile::tempdir().unwrap();
    let process = process(root.path());
    let overrides = || Overrides {
        listen: Some("0.0.0.0:8765".parse().unwrap()),
        ..Overrides::default()
    };
    assert!(Startup::load(&process, overrides()).is_err());
    let token = root.path().join("token");
    std::fs::write(&token, "synthetic-token\n").unwrap();
    let mut options = overrides();
    options.auth_token_file = Some(token.clone());
    let startup = Startup::load(&process, options).unwrap();
    std::fs::write(token, "changed-token").unwrap();
    assert_eq!(
        startup.config.http.token.as_ref().unwrap().expose(),
        "synthetic-token"
    );
    assert_eq!(
        format!("{:?}", startup.config.http.token.as_ref().unwrap()),
        "BearerToken([redacted])"
    );
}
#[test]
fn output_and_grammar_inputs_need_no_deployment_or_machine_identity() {
    let output = OutputConfig::new(true, true, true);
    assert!(!output.stdout_color());
    assert!(!output.stderr_color());
    assert!(OutputConfig::new(true, false, false).stdout_color());
    let environment = EnvSnapshot::from_values([
        ("CODEX_THREAD_ID".into(), "synthetic-thread".into()),
        ("POLYLOGUE_AUTH_TOKEN_FILE".into(), "/secret-path".into()),
        ("VOYAGE_API_KEY".into(), "synthetic-secret".into()),
    ]);
    assert_eq!(
        environment
            .caller_environment()
            .get("CODEX_THREAD_ID")
            .map(String::as_str),
        Some("synthetic-thread")
    );
    assert!(
        !environment
            .caller_environment()
            .contains_key("POLYLOGUE_AUTH_TOKEN_FILE")
    );
    assert!(!format!("{environment:?}").contains("synthetic-secret"));
    let matches = augment_command(Command::new("polylogue"))
        .try_get_matches_from(["polylogue", "--archive", "/synthetic/archive"])
        .unwrap();
    assert_eq!(
        Overrides::from_matches(&matches).archive,
        Some("/synthetic/archive".into())
    );
}
#[test]
fn outlets_use_typed_arguments_and_stable_resolved_identity() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("signals.toml");
    std::fs::write(&path, "[notifications]\nroute='command'\ncommand={program='./bin/../notify',arguments=['a b','literal;$HOME']}\n[value_targets.answer]\npath='./state/../answer'\n[action_commands.refresh]\nprogram='bin/refresh'\narguments=['--mode','once']\n").unwrap();
    let config = SignalDeployment::load(&path).unwrap();
    assert_eq!(
        config.notifications.command.as_ref().unwrap().arguments,
        ["a b", "literal;$HOME"]
    );
    assert_eq!(
        config.notifications.command.as_ref().unwrap().program,
        root.path().join("notify")
    );
    assert_eq!(
        config.value_targets["answer"].path,
        root.path().join("answer")
    );
    assert!(
        config
            .rule_configuration()
            .action_commands
            .contains("refresh")
    );
    assert_eq!(
        config.identity().unwrap(),
        SignalDeployment::load(&path).unwrap().identity().unwrap()
    );
    std::fs::write(&path, "[notifications]\nroute='command'\n").unwrap();
    assert!(SignalDeployment::load(&path).is_err());
    std::fs::write(&path, "[value_targets.answer]\npath=''\n").unwrap();
    assert!(SignalDeployment::load(&path).is_err());
}
#[cfg(feature = "sinex")]
#[test]
fn primary_mode_keeps_unavailable_credentials_and_never_supplies_a_local_day() {
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "");
    let path = root.path().join("config/polylogue/sinex.toml");
    std::fs::write(&path, "mode='primary'\nnats='nats://127.0.0.1:1'\nstore='store'\npassword_file='missing-password'\n").unwrap();
    let startup = Startup::load(&process(root.path()), Overrides::default()).unwrap();
    assert!(matches!(
        startup.mode,
        crate::daemon::service::ArchiveMode::Primary { .. }
    ));
    assert!(matches!(
        startup.config.primary_auth,
        polylogue_sinex::Authentication::Unavailable { .. }
    ));
    assert!(matches!(
        startup.config.standalone_day(),
        Err(Error::SinexDay)
    ));
    deployment(root.path(), "[day]\nzone='UTC'\nstart='04:00:00'\n");
    assert!(matches!(
        Startup::load(&process(root.path()), Overrides::default()),
        Err(Error::SinexDay)
    ));
    std::fs::write(path, "broken = [").unwrap();
    let startup = Startup::load(&process(root.path()), Overrides::default()).unwrap();
    assert!(matches!(
        startup.mode,
        crate::daemon::service::ArchiveMode::Blocked { .. }
    ));
    assert!(startup.config.standalone_day().is_err());
}

#[cfg(feature = "sinex")]
#[test]
fn unreadable_primary_deployment_stays_blocked_without_local_day() {
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "");
    // A directory gives a deterministic read failure even under a privileged test runner.
    std::fs::create_dir(root.path().join("config/polylogue/sinex.toml")).unwrap();
    let startup = Startup::load(&process(root.path()), Overrides::default()).unwrap();
    assert!(matches!(
        startup.mode,
        crate::daemon::service::ArchiveMode::Blocked { .. }
    ));
    assert!(startup.config.standalone_day().is_err());
}

#[test]
fn fixture_configuration_uses_explicit_engine_inputs_and_has_no_external_outlets() {
    let root = tempfile::tempdir().unwrap();
    let engine = polylogue_engine::Config::new(
        root.path().join("home"),
        BTreeMap::new(),
        "fixture-machine".into(),
        "fixture-host".into(),
        2,
        WallClock::fixed("2026-03-08T12:34:56Z".parse().unwrap()),
    )
    .unwrap();
    let startup = Startup::for_test(engine.clone(), &root.path().join("archive"));
    assert_eq!(startup.config.identity.machine, "fixture-machine");
    assert_eq!(startup.config.requests.now(), engine.clock.now().into());
    assert_eq!(startup.clock.now(), engine.clock.now());
    assert_eq!(startup.config.standalone_day().unwrap().zone_name(), "UTC");
    assert!(startup.config.notification_bus_address.is_none());
    assert!(startup.config.signals.action_commands.is_empty());
    assert!(startup.config.signals.value_targets.is_empty());
}

#[test]
fn completion_and_startup_share_declared_archive_selection_without_clock_dependencies() {
    let root = tempfile::tempdir().unwrap();
    deployment(
        root.path(),
        "archive='file-archive'\n[day]\nzone='Invalid/Zone'\nstart='04:00:00'\n",
    );
    let mut inputs = process(root.path());
    inputs.identity = None;
    inputs.workers = None;
    let overrides = Overrides::default();
    assert_eq!(
        archive_location(&inputs, &overrides).unwrap(),
        root.path().join("config/polylogue/file-archive")
    );
    assert_eq!(
        archive_location(
            &process,
            &Overrides {
                archive: Some("chosen".into()),
                ..Overrides::default()
            }
        )
        .unwrap(),
        root.path().join("chosen")
    );
    deployment(root.path(), "archive='file-archive'\n");
    let inputs = process(root.path());
    assert_eq!(
        archive_location(&inputs, &overrides).unwrap(),
        Startup::load(&inputs, overrides).unwrap().config.archive
    );
}

#[test]
fn synthetic_archive_gets_an_explicit_day_without_operator_outlets() {
    let root = tempfile::tempdir().unwrap();
    let startup = Startup::load(&process(root.path()), Overrides::default()).unwrap();
    let day = DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::constant(6, 0, 1, 0),
    )
    .unwrap();
    let separate = startup
        .config
        .standalone_for_archive(root.path().join("sample"), day.clone())
        .unwrap();
    assert_eq!(separate.archive, root.path().join("sample"));
    assert_eq!(separate.standalone_day().unwrap(), &day);
    assert!(separate.notification_bus_address.is_none());
    assert!(separate.signals.action_commands.is_empty());
    assert!(
        startup
            .config
            .standalone_for_archive("relative".into(), day)
            .is_err()
    );
    assert_ne!(startup.config.archive, separate.archive);
}
