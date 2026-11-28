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
        output: OutputConfig::new(false, false, false, None),
        clock: WallClock::fixed("2026-03-07T18:00:00Z".parse().unwrap()),
        identity: Some(Identity {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
        }),
        memory_bytes: Some(4 << 30),
        workers: Some(2),
        cwd: Some(root.into()),
    }
}
/// The daemon's load: the client's configuration extended with the rest.
fn loaded(process: &ProcessInputs, overrides: Overrides) -> Result<Startup, Error> {
    Client::load(process, overrides).and_then(|client| Startup::extend(client, process))
}
fn deployment(root: &Path, text: &str) {
    let config = root.join("config/polylogue");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(config.join("polylogue.toml"), text).unwrap();
}
#[test]
fn explicit_configuration_selects_its_auxiliary_files_without_changing_xdg_paths() {
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "invalid = [");
    let defaults = root.path().join("config/polylogue");
    std::fs::write(defaults.join("sinex.toml"), "invalid = [").unwrap();
    std::fs::write(defaults.join("signals.toml"), "invalid = [").unwrap();
    let selected = root.path().join("deployment");
    std::fs::create_dir(&selected).unwrap();
    std::fs::write(selected.join("polylogue.toml"), "archive='archive'\n").unwrap();
    std::fs::write(
        selected.join("signals.toml"),
        "[action_commands.refresh]\nprogram='bin/refresh'\narguments=['two words']\n",
    )
    .unwrap();
    let process = process(root.path());
    let options = || Overrides {
        config_file: Some("deployment/polylogue.toml".into()),
        ..Overrides::default()
    };
    let startup = loaded(&process, options()).unwrap();
    assert_eq!(startup.config.archive, selected.join("archive"));
    assert_eq!(startup.config.paths.config, root.path().join("config"));
    assert_eq!(startup.config.paths.home, root.path().join("home"));
    assert!(matches!(
        startup.mode,
        crate::daemon::service::ArchiveMode::Standalone
    ));
    let command = &startup.config.signals.action_commands["refresh"];
    assert_eq!(command.program, selected.join("bin/refresh"));
    assert_eq!(command.arguments, ["two words"]);
    let signals = selected.join("signals.toml");
    std::fs::write(&signals, "[notifications]\ndeadline_ms==100").unwrap();
    match loaded(&process, options()) {
        Err(Error::Invalid(message)) => assert_eq!(
            message,
            format!(
                "{}: TOML parse error at line 2, column 13\n  |\n2 | deadline_ms==100\n  |             ^\nextra `=`, expected nothing\n",
                signals.display()
            )
        ),
        Err(error) => panic!("expected invalid signals configuration, got {error:?}"),
        Ok(_) => panic!("accepted an extra equals sign in signals configuration"),
    }
}

/// The day's zone and start when the deployment names no zone, loaded in a
/// process whose host zone is `POLYLOGUE_TEST_HOST_ZONE`.
#[test]
fn standalone_day_host_zone_child() {
    let Some(zone) = std::env::var_os("POLYLOGUE_TEST_HOST_ZONE") else {
        return;
    };
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "");
    let startup = loaded(&process(root.path()), Overrides::default()).unwrap();
    let day = startup.config.standalone_day().unwrap();
    assert_eq!(day.zone_name(), zone.to_str().unwrap());
    assert_eq!(day.start(), jiff::civil::Time::constant(4, 0, 0, 0));
    deployment(root.path(), "[day]\nstart='06:00:00'\n");
    let startup = loaded(&process(root.path()), Overrides::default()).unwrap();
    let day = startup.config.standalone_day().unwrap();
    assert_eq!(day.zone_name(), zone.to_str().unwrap());
    assert_eq!(day.start(), jiff::civil::Time::constant(6, 0, 0, 0));
}

#[test]
fn standalone_day_without_a_zone_uses_the_hosts_named_zone() {
    // The host zone is a process setting, so each zone runs in a child
    // process started with it.
    for zone in ["Asia/Tokyo", "Pacific/Auckland"] {
        let test = "config::tests::standalone_day_host_zone_child";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env("TZ", zone)
            .env("POLYLOGUE_TEST_HOST_ZONE", zone)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{zone}: {stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(stdout.contains("1 passed"), "{zone}: {stdout}");
    }
}

#[test]
fn standalone_day_uses_the_named_zone_across_dst_and_refuses_invalid_zones() {
    let root = tempfile::tempdir().unwrap();
    deployment(
        root.path(),
        "[day]\nzone='America/Los_Angeles'\nstart='06:00:00'\n",
    );
    let startup = loaded(&process(root.path()), Overrides::default()).unwrap();
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
    assert!(matches!(
        loaded(&process(root.path()), Overrides::default()),
        Err(Error::Invalid(_))
    ));
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
    let startup = loaded(
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
    match loaded(&process, overrides()) {
        Err(Error::Invalid(message)) => assert_eq!(
            message,
            "a non-loopback listen address requires an authentication token file"
        ),
        Err(error) => panic!("expected an invalid listen address, got {error:?}"),
        Ok(_) => panic!("accepted a non-loopback listen address without a token"),
    }
    let token = root.path().join("token");
    std::fs::write(&token, "synthetic-token\n").unwrap();
    let mut options = overrides();
    options.auth_token_file = Some(token.clone());
    let startup = loaded(&process, options).unwrap();
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
    let output = OutputConfig::new(true, true, true, None);
    assert!(!output.stdout_color());
    assert!(!output.stderr_color());
    assert!(OutputConfig::new(true, false, false, None).stdout_color());
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
}
#[test]
fn outlets_use_typed_arguments_and_stable_resolved_identity() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("signals.toml");
    std::fs::write(&path, "[notifications]\naway_command={program='./bin/../notify',arguments=['a b','literal;$HOME']}\n[value_targets.answer]\npath='./state/../answer'\n[action_commands.refresh]\nprogram='bin/refresh'\narguments=['--mode','once']\n").unwrap();
    let config = SignalDeployment::read(&path).unwrap();
    assert_eq!(
        config
            .notifications
            .away_command
            .as_ref()
            .unwrap()
            .arguments,
        ["a b", "literal;$HOME"]
    );
    assert_eq!(
        config.notifications.away_command.as_ref().unwrap().program,
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
        SignalDeployment::read(&path).unwrap().identity().unwrap()
    );
    let before = config.identity().unwrap();
    std::fs::write(
        &path,
        "[action_commands.refresh]\nprogram='bin/refresh'\narguments=['--mode','twice']\n",
    )
    .unwrap();
    assert_ne!(
        SignalDeployment::read(&path).unwrap().identity().unwrap(),
        before
    );
    std::fs::write(&path, "[notifications]\nroute='command'\n").unwrap();
    match SignalDeployment::read(&path).unwrap_err() {
        Error::Invalid(message) => assert_eq!(
            message,
            format!(
                "{}: TOML parse error at line 2, column 1\n  |\n2 | route='command'\n  | ^^^^^\nunknown field `route`, expected `away_command` or `deadline_ms`\n",
                path.display()
            )
        ),
        error => panic!("expected an unknown notification field, got {error:?}"),
    }
    for (text, expected) in [
        (
            "[notifications]\naway_command={program=' '}\n",
            "configured command program cannot be blank",
        ),
        (
            "[value_targets.answer]\npath=''\n",
            "value target path cannot be blank",
        ),
    ] {
        std::fs::write(&path, text).unwrap();
        match SignalDeployment::read(&path).unwrap_err() {
            Error::Invalid(message) => assert_eq!(message, expected),
            error => panic!("expected invalid signals configuration, got {error:?}"),
        }
    }
}
#[test]
fn unreadable_primary_deployment_stays_blocked_without_local_day() {
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "");
    // A directory gives a deterministic read failure even under a privileged test runner.
    std::fs::create_dir_all(root.path().join("home/.local/share/polylogue/sinex.toml")).unwrap();
    let startup = loaded(&process(root.path()), Overrides::default()).unwrap();
    assert!(matches!(
        startup.mode,
        crate::daemon::service::ArchiveMode::Blocked { .. }
    ));
    assert!(matches!(
        startup.config.standalone_day(),
        Err(Error::SinexDay)
    ));
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
    assert_eq!(startup.client.clock.now(), engine.clock.now());
    assert_eq!(startup.config.standalone_day().unwrap().zone_name(), "UTC");
    assert!(startup.config.notification_bus_address.is_none());
    assert!(startup.config.signals.action_commands.is_empty());
    assert!(startup.config.signals.value_targets.is_empty());
}

#[test]
fn the_client_loads_without_identity_clock_or_daemon_files() {
    let root = tempfile::tempdir().unwrap();
    deployment(
        root.path(),
        "archive='file-archive'\n[http]\nauth_token_file='missing-token'\n[day]\nzone='Invalid/Zone'\nstart='04:00:00'\n",
    );
    let defaults = root.path().join("config/polylogue");
    std::fs::write(defaults.join("signals.toml"), "invalid = [").unwrap();
    std::fs::create_dir_all(root.path().join("config/polylogue/file-archive")).unwrap();
    std::fs::write(
        root.path().join("config/polylogue/file-archive/sinex.toml"),
        "invalid = [",
    )
    .unwrap();
    let mut inputs = process(root.path());
    inputs.identity = None;
    inputs.workers = None;
    let client = Client::load(&inputs, Overrides::default()).unwrap();
    assert_eq!(
        client.archive,
        root.path().join("config/polylogue/file-archive")
    );
    assert_eq!(client.zone(), jiff::tz::TimeZone::UTC);
    assert_eq!(
        Client::load(
            &inputs,
            Overrides {
                archive: Some("chosen".into()),
                ..Overrides::default()
            }
        )
        .unwrap()
        .archive,
        root.path().join("chosen")
    );
    // The daemon's load reads what the client did not, and refuses it.
    assert!(matches!(
        Startup::extend(client, &process(root.path())),
        Err(Error::File { path, .. }) if path == defaults.join("missing-token")
    ));
}

/// The client's load: the file, then the flag; no environment variable
/// selects the archive.
#[test]
fn client_configuration_precedence_is_file_then_flags() {
    let root = tempfile::tempdir().unwrap();
    let mut inputs = process(root.path());
    inputs
        .environment
        .values
        .insert("POLYLOGUE_ARCHIVE".into(), "/unrelated".into());
    let archive = |overrides: Overrides| Client::load(&inputs, overrides).unwrap().archive;
    assert_eq!(
        archive(Overrides::default()),
        root.path().join("home/.local/share/polylogue")
    );
    deployment(root.path(), "archive='file-archive'\n");
    assert_eq!(
        archive(Overrides::default()),
        root.path().join("config/polylogue/file-archive")
    );
    assert_eq!(
        archive(Overrides {
            archive: Some("flag/../archive".into()),
            ..Overrides::default()
        }),
        root.path().join("archive")
    );
}

#[test]
fn startup_without_working_directory_is_an_io_failure() {
    let root = tempfile::tempdir().unwrap();
    let mut inputs = process(root.path());
    inputs.cwd = None;
    let error = Client::load(&inputs, Overrides::default()).unwrap_err();
    assert!(matches!(error, Error::WorkingDirectory));
    let failure = error.failure();
    assert_eq!(failure.kind.exit_code(), 4);
    assert_eq!(failure.message, "the current directory is unavailable");
}

#[test]
fn synthetic_archive_gets_an_explicit_day_without_operator_outlets() {
    let root = tempfile::tempdir().unwrap();
    let startup = loaded(&process(root.path()), Overrides::default()).unwrap();
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
    match startup
        .config
        .standalone_for_archive("relative".into(), day)
    {
        Err(Error::Invalid(message)) => {
            assert_eq!(message, "the standalone archive path must be absolute")
        }
        Err(error) => panic!("expected an invalid standalone archive path, got {error:?}"),
        Ok(_) => panic!("accepted a relative standalone archive path"),
    }
    assert_ne!(startup.config.archive, separate.archive);
}

#[test]
fn captured_terminal_hints_disable_color_on_both_sinks() {
    let environment = EnvSnapshot::from_values([("TERM".into(), "dumb".into())]);
    let output = OutputConfig::new(
        true,
        true,
        environment.get_os("NO_COLOR").is_some(),
        environment.text("TERM").unwrap(),
    );
    assert!(!output.stdout_color());
    assert!(!output.stderr_color());
    let environment = EnvSnapshot::from_values([
        ("TERM".into(), "xterm".into()),
        ("NO_COLOR".into(), "".into()),
    ]);
    let output = OutputConfig::new(
        true,
        true,
        environment.get_os("NO_COLOR").is_some(),
        environment.text("TERM").unwrap(),
    );
    assert!(!output.stdout_color());
    assert!(!output.stderr_color());
}

#[test]
fn archive_override_selects_its_own_mode_configuration() {
    let root = tempfile::tempdir().unwrap();
    deployment(root.path(), "");
    let first = root.path().join("first");
    let second = root.path().join("second");
    std::fs::create_dir(&first).unwrap();
    std::fs::create_dir(&second).unwrap();
    std::fs::write(first.join("sinex.toml"), "mode='unknown'\n").unwrap();
    std::fs::write(second.join("sinex.toml"), "mode='standalone'\n").unwrap();
    let load = |archive| {
        loaded(
            &process(root.path()),
            Overrides {
                archive: Some(archive),
                ..Default::default()
            },
        )
        .unwrap()
    };
    let broken = load(first.clone());
    let crate::daemon::service::ArchiveMode::Blocked { path, error, .. } = broken.mode else {
        panic!("invalid archive configuration was accepted")
    };
    assert_eq!(path, first.join("sinex.toml"));
    assert!(!error.is_empty());
    assert!(matches!(
        broken.config.standalone_day(),
        Err(Error::SinexDay)
    ));
    assert!(matches!(
        load(second).mode,
        crate::daemon::service::ArchiveMode::Standalone
    ));
}

#[test]
fn startup_sizes_the_pipeline_within_the_process_allowance() {
    let root = tempfile::tempdir().unwrap();
    // Workers follow the cores while each has 128 MiB of a third of the
    // budget (three quarters of the allowance); no share has a fixed
    // ceiling.
    for (cpus, bytes, workers, readers, queued) in [
        (24, 4u64 << 30, 8, 128 << 20, 192 << 20),
        (16, 1 << 30, 2, 48 << 20, 48 << 20),
        (2, 4 << 30, 2, 128 << 20, 192 << 20),
        (16, 32 << 30, 16, 128 << 20, 512 << 20),
    ] {
        let mut inputs = process(root.path());
        inputs.workers = Some(cpus);
        inputs.memory_bytes = Some(bytes);
        let startup = loaded(&inputs, Overrides::default()).unwrap();
        assert_eq!(startup.engine.workers, workers);
        assert_eq!(startup.engine.reader_memory_bytes, readers);
        assert_eq!(startup.engine.bytes_in_flight, queued);
        assert_eq!(startup.engine.chunk, 16 << 20);
    }
}

#[test]
fn zsh_startup_directory_is_resolved_from_the_captured_environment() {
    let root = tempfile::tempdir().unwrap();
    let mut inputs = process(root.path());
    inputs.environment = EnvSnapshot::from_values([
        ("HOME".into(), root.path().join("home").into_os_string()),
        ("ZDOTDIR".into(), "shell-startup".into()),
    ]);
    let startup = loaded(&inputs, Overrides::default()).unwrap();
    assert_eq!(
        startup.client.zsh_directory,
        root.path().join("shell-startup")
    );
}
#[test]
fn an_empty_xdg_variable_is_unset() {
    let environment = EnvSnapshot::from_values([
        ("HOME".into(), "/home/op".into()),
        ("XDG_CONFIG_HOME".into(), "".into()),
        ("XDG_DATA_HOME".into(), "/data".into()),
    ]);
    let paths = Paths::resolve(&environment).unwrap();
    assert_eq!(paths.config, PathBuf::from("/home/op/.config"));
    assert_eq!(paths.data, PathBuf::from("/data"));
}
