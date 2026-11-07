use super::*;

#[test]
fn a_missing_daemon_is_daemon_required() {
    let dir = tempfile::tempdir().unwrap();
    let err = crate::client::send(&dir.path().join("daemon.sock"), &Request::Status).unwrap_err();
    assert!(is_down(&err));
    assert!(!Archive::at(dir.path()).lock_held());
    let err = send_with_timeout(&Archive::at(dir.path()), &Request::Status, None).unwrap_err();
    assert!(is_down(&err));
}

#[test]
fn a_starting_daemon_is_waited_for() {
    let dir = tempfile::tempdir().unwrap();
    let archive = Archive::at(dir.path());
    // The daemon's order: take the lock, open the archive, then bind.
    let (_, lock) = Archive::open_or_create(dir.path()).unwrap();
    assert!(archive.lock_held());
    let socket = archive.socket();
    let server = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(300));
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        let (stream, _) = listener.accept().unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(&stream), &mut line).unwrap();
        let reply = serde_json::to_string(&command_surface::Answer::ok(
            serde_json::json!({"ready": true}),
        ))
        .unwrap();
        std::io::Write::write_all(&mut &stream, format!("{reply}\n").as_bytes()).unwrap();
        drop(lock);
    });
    let response = send_with_timeout(&archive, &Request::Status, None).unwrap();
    assert_eq!(
        crate::test_answers::ok(response),
        serde_json::json!({"ready": true})
    );
    server.join().unwrap();
}

#[test]
fn manual_is_an_offline_topic() {
    let manual = COMMANDS.iter().find(|c| c.name == "manual").unwrap();
    assert!(matches!(manual.action, Action::Local(_)));
    let matches = command()
        .try_get_matches_from(["polylogue", "manual"])
        .unwrap();
    assert_eq!(
        run_local(
            &matches,
            crate::config::OutputConfig::new(false, false, false, None)
        ),
        Some(std::process::ExitCode::SUCCESS)
    );
}

#[test]
fn generated_ready_cli_call_reaches_the_socket_with_its_identity_and_deadline() {
    use std::io::{BufRead, Write};
    let daemon = crate::test_daemon::start("canonical command witness");
    let request = command_surface::Request {
        operation: "query".into(),
        arguments: serde_json::json!({"query":"sessions"}),
        request_id: "quoted ' identity".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let ready = daemon.daemon.ready_call(&request).unwrap();
    let stages = polylogue_model::command::effective_stages(&ready.cli);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].args[0], "find");
    let root = tempfile::tempdir().unwrap();
    let archive = Archive::at(root.path());
    let listener = std::os::unix::net::UnixListener::bind(archive.socket()).unwrap();
    listener.set_nonblocking(true).unwrap();
    let expected = request.clone();
    let server = std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && start.elapsed() < std::time::Duration::from_secs(3) =>
                {
                    std::thread::yield_now()
                }
                Err(error) => panic!("CLI did not reach the canonical socket: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        let mut text = String::new();
        std::io::BufReader::new(&stream)
            .read_line(&mut text)
            .unwrap();
        let (actual, _, _) = crate::protocol::wire::incoming(&text).unwrap().unwrap();
        assert_eq!(actual, expected);
        let answer = command_surface::Answer::ok(serde_json::json!({"count":1}));
        writeln!(stream, "{}", serde_json::to_string(&answer).unwrap()).unwrap();
    });
    let mut arguments = vec!["polylogue".to_owned()];
    arguments.extend(stages[0].args.clone());
    arguments.push("--json".into());
    let matches = command().try_get_matches_from(arguments).unwrap();
    let startup = crate::config::Startup::for_test(daemon.config.clone(), root.path());
    assert_eq!(run(&matches, &startup, root.path()), ExitCode::SUCCESS);
    server.join().unwrap();
    let mixed = command()
        .try_get_matches_from([
            "polylogue",
            "find",
            "--request",
            r#"{"query":"sessions"}"#,
            "--limit",
            "1",
        ])
        .unwrap();
    assert_eq!(run(&mixed, &startup, root.path()), ExitCode::from(2));
}

#[test]
fn query_options_work_after_words_and_the_separator_keeps_literal_options() {
    for argv in [
        vec!["polylogue", "--json", "sessions where parser"],
        vec!["polylogue", "sessions where parser", "--json"],
        vec!["polylogue", "find", "sessions where parser", "--json"],
    ] {
        let matches = command().try_get_matches_from(argv).unwrap();
        let (selected, arguments) = chosen(&matches).unwrap();
        assert_eq!(selected.name, "query");
        assert!(matches.get_flag("json"));
        assert_eq!(
            values(selected, arguments).text("query").as_deref(),
            Some("sessions where parser")
        );
    }
    let (_, literal) = parse(&["polylogue", "find", "--", "--json"]).unwrap();
    assert_eq!(literal.text("query").as_deref(), Some("--json"));
}

#[test]
fn raw_cli_arguments_replace_required_positionals() {
    let (selected, values) =
        parse(&["polylogue", "find", "--request", "{}", "--deadline", "none"]).unwrap();
    assert_eq!(selected.name, "query");
    assert!(values.text("query").is_none());
    assert!(
        command()
            .try_get_matches_from(["polylogue", "read", "--request", r#"{"target":"sessions"}"#])
            .is_ok()
    );
}

#[test]
fn clients_send_deadlines_only_for_reads_and_explicit_stream_limits() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    use std::time::Duration;
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    for (request, timeout, expected) in [
        (Request::Status, None, true),
        (
            Request::Read {
                target: "sessions".into(),
                options: polylogue_model::read::ReadOptions::default(),
                delivery: polylogue_model::read::ReadDelivery::Stream,
                format: "json".into(),
                budget: None,
                with_blobs: false,
                cursor: None,
            },
            None,
            false,
        ),
        (
            Request::Read {
                target: "sessions".into(),
                options: polylogue_model::read::ReadOptions::default(),
                delivery: polylogue_model::read::ReadDelivery::Stream,
                format: "json".into(),
                budget: None,
                with_blobs: false,
                cursor: None,
            },
            Some(Duration::from_secs(120)),
            true,
        ),
        (
            Request::Embed {
                what: "build".into(),
                query: None,
                limit: None,
                yes: false,
            },
            Some(Duration::from_secs(120)),
            false,
        ),
    ] {
        let listener = UnixListener::bind(archive.socket()).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            BufReader::new(&stream).read_line(&mut line).unwrap();
            let outgoing: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(outgoing.get("deadline_ms").is_some(), expected);
            if let Some(timeout) = timeout.filter(|_| expected) {
                let left =
                    outgoing["deadline_ms"].as_i64().unwrap() - crate::daemon::service::now_ms();
                assert!(left > timeout.as_millis() as i64 - 5000);
            }
            stream
                .write_all(b"{\"outcome\":\"ok\",\"value\":{},\"notes\":[],\"next\":[]}\n")
                .unwrap();
        });
        send_with_timeout(&archive, &request, timeout).unwrap();
        server.join().unwrap();
        std::fs::remove_file(archive.socket()).unwrap();
    }
}

#[test]
fn shell_completion_escapes_daemon_help_and_omits_control_values() {
    let data = serde_json::json!([
        {"value": "tag:ordinary", "help": "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}"},
        {"value": "tag:bad\x1b", "help": ""},
        {"value": "query:bad\u{202e}", "help": ""},
        {"value": "tag:bad\n", "help": ""},
        {"value": "tag:bad\t", "help": ""}
    ]);
    let found = data
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|item| {
            completion::candidate(
                item["value"].as_str().unwrap().into(),
                item["help"].as_str().unwrap().into(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].get_value(), "tag:ordinary");
    assert_eq!(
        found[0].get_help().unwrap().to_string(),
        polylogue_render::terminal::escape(data[0]["help"].as_str().unwrap())
    );
}

#[test]
fn a_json_value_without_its_final_line_boundary_is_not_a_completed_answer() {
    use std::io::{BufRead, Write};
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let listener = std::os::unix::net::UnixListener::bind(archive.socket()).unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        let mut request = String::new();
        std::io::BufReader::new(&stream)
            .read_line(&mut request)
            .unwrap();
        write!(
            stream,
            "{}",
            serde_json::to_string(&command_surface::Answer::ok(
                serde_json::json!({"stale":"complete JSON but incomplete frame"})
            ))
            .unwrap()
        )
        .unwrap();
    });
    let call = command_surface::Request {
        operation: "status".into(),
        arguments: serde_json::json!({}),
        request_id: "incomplete-answer-witness".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let error = crate::client::call_ready(&archive, &call).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    server.join().unwrap();
}

#[test]
fn literal_read_envelope_is_not_expanded_with_cli_defaults() {
    use std::io::{BufRead, Write};
    let daemon = crate::test_daemon::start("read envelope witness");
    for form in ["generated", "literal", "flags", "export", "export-hidden"] {
        let root = tempfile::tempdir().unwrap();
        let archive = Archive::at(root.path());
        let destination = root.path().join("reading");
        let listener = std::os::unix::net::UnixListener::bind(archive.socket()).unwrap();
        listener.set_nonblocking(true).unwrap();
        let mut expected = command_surface::Request {
            operation: "read".into(),
            arguments: serde_json::json!({"target":"sessions"}),
            request_id: "read ' identity".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let mut argv = vec!["polylogue".to_string()];
        match form {
            "generated" => {
                let ready = daemon.daemon.ready_call(&expected).unwrap();
                argv.extend(
                    polylogue_model::command::effective_stages(&ready.cli)[0]
                        .args
                        .clone(),
                );
            }
            "literal" => argv.extend([
                "read".into(),
                "--request".into(),
                expected.arguments.to_string(),
                "--request-id".into(),
                expected.request_id.clone(),
                "--deadline".into(),
                "none".into(),
            ]),
            _ => {
                expected.arguments = serde_json::json!({
                    "target":"sessions", "view":"transcript", "format":"markdown",
                    "page_tokens":41, "output_lines":100, "show":["tools:line"],
                    "for_others":true, "variant":["concise"], "with_blobs":true,
                });
                expected.deadline = command_surface::Deadline::Milliseconds(60_000);
                expected.cursor = Some("sealed:cursor".into());
                argv.extend(
                    [
                        "read",
                        "sessions",
                        "--view",
                        "transcript",
                        "--format",
                        "markdown",
                        "--page-tokens",
                        "41",
                        "--output-lines",
                        "100",
                        "--show",
                        "tools:line",
                        "--for-others",
                        "--variant",
                        "concise",
                        "--with-blobs",
                        "--request-id",
                        "read ' identity",
                        "--deadline",
                        "60000",
                        "--cursor",
                        "sealed:cursor",
                    ]
                    .map(str::to_owned),
                );
                if form.starts_with("export") {
                    argv.extend(["--to".into(), destination.to_str().unwrap().into()]);
                    expected.arguments["delivery"] = serde_json::json!("stream");
                    if form == "export-hidden" {
                        argv.extend(["--hide".into(), "inherited".into()]);
                        expected.arguments["hide"] = serde_json::json!(["inherited"]);
                    } else {
                        expected.arguments["show"] = serde_json::json!(["tools:line", "inherited"]);
                    }
                }
            }
        }
        argv.push("--json".into());
        let peer = std::thread::spawn(move || {
            let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error)
                        if error.kind() == std::io::ErrorKind::WouldBlock
                            && std::time::Instant::now() < until =>
                    {
                        std::thread::yield_now()
                    }
                    Err(error) => {
                        panic!("{form} read did not reach the configured socket: {error}")
                    }
                }
            };
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut line = String::new();
            std::io::BufReader::new(&stream)
                .read_line(&mut line)
                .unwrap();
            let (actual, _, _) = crate::protocol::wire::incoming(&line).unwrap().unwrap();
            assert_eq!(actual, expected, "{form} changed the accepted envelope");
            let refusal =
                command_surface::Answer::<serde_json::Value>::error(command_surface::Failure::new(
                    command_surface::FailureKind::Refused,
                    "synthetic refusal",
                ));
            writeln!(stream, "{}", serde_json::to_string(&refusal).unwrap()).unwrap();
        });
        let matches = command().try_get_matches_from(argv).unwrap();
        let startup = crate::config::Startup::for_test(daemon.config.clone(), root.path());
        assert_eq!(
            run(&matches, &startup, root.path()),
            ExitCode::from(3),
            "{form}"
        );
        peer.join().unwrap();
        assert!(
            !destination.exists(),
            "a refused export must not publish its destination"
        );
    }
}

#[test]
fn captured_output_policy_reaches_query_and_error_renderers_per_sink() {
    use crate::config::OutputConfig;
    let data = rows_document(
        serde_json::json!([{"name":"text","kind":"text"}]),
        serde_json::json!([{"text":"stored \x1b[31m λ", "snippet":{"text":"match stored \x1b[31m λ","marks":[[0,5]]}}]),
        1,
    );
    let original = data.clone();
    for (stdout, stderr, no_color, term) in [
        (false, false, false, Some("xterm")),
        (true, false, false, Some("xterm")),
        (false, true, false, Some("xterm")),
        (true, true, false, Some("xterm")),
        (true, true, true, Some("xterm")),
        (true, true, false, Some("dumb")),
    ] {
        let output = OutputConfig::new(stdout, stderr, no_color, term);
        let command = crate::commands::command("query").unwrap();
        let text = (command.print)(&data, output.stdout_color());
        assert_eq!(text.contains("\x1b[1m"), output.stdout_color());
        assert!(
            !text.contains("\x1b[31m"),
            "archived ANSI must remain inert"
        );
        assert!(text.contains("\\x1b[31m"));
        let error = print::diagnostic(
            "invalid_query",
            "stored \x1b[31m λ",
            Some("query λ"),
            output.stderr_color(),
        );
        assert_eq!(error.contains("\x1b[1m"), output.stderr_color());
        assert!(!error.contains("\x1b[31m"));
        assert!(error.contains("\\x1b[31m"));
        assert!(error.ends_with("polylogue: try: query λ\n"));
        let json = print::json(&data, output.stdout_color());
        assert!(!json.contains('\x1b'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap(),
            original
        );
    }
    assert_eq!(data, original);
}

#[test]
fn help_and_parse_errors_use_captured_stdout_and_stderr_policies() {
    let help = command()
        .try_get_matches_from(["polylogue", "--help"])
        .unwrap_err();
    let error = command()
        .try_get_matches_from(["polylogue", "--unknown-option"])
        .unwrap_err();
    assert!(!help.use_stderr());
    assert!(error.use_stderr());
    // The parser produces plain text before the displaying client's styling.
    assert!(!help.render().ansi().to_string().contains('\x1b'));
    assert!(!error.render().ansi().to_string().contains('\x1b'));
    for (stdout, stderr, no_color, term) in [
        (true, false, false, Some("xterm")),
        (false, true, false, Some("xterm")),
        (false, false, false, Some("xterm")),
        (true, true, true, Some("xterm")),
        (true, true, false, Some("dumb")),
    ] {
        let output = crate::config::OutputConfig::new(stdout, stderr, no_color, term);
        assert_eq!(
            parse_text(&help, output).contains("\x1b[1m"),
            output.stdout_color()
        );
        assert_eq!(
            parse_text(&error, output).contains("\x1b[1m"),
            output.stderr_color()
        );
    }
}

#[test]
fn help_and_build_revision_exit_before_loading_an_explicit_deployment() {
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("polylogue.toml");
    let archive = root.path().join("unopened-archive");
    let malformed = "invalid = [";
    std::fs::write(&config, malformed).unwrap();
    for (flag, kind) in [
        ("--version", clap::error::ErrorKind::DisplayVersion),
        ("--help", clap::error::ErrorKind::DisplayHelp),
    ] {
        let error = command()
            .try_get_matches_from([
                std::ffi::OsStr::new("polylogue"),
                std::ffi::OsStr::new("--config"),
                config.as_os_str(),
                std::ffi::OsStr::new("--archive"),
                archive.as_os_str(),
                std::ffi::OsStr::new(flag),
            ])
            .unwrap_err();
        assert_eq!(error.kind(), kind);
        assert!(!error.use_stderr());
        if flag == "--version" {
            let revision = option_env!("POLYLOGUE_REVISION").unwrap_or("unavailable");
            assert_eq!(
                error.to_string(),
                format!(
                    "polylogue {} (revision {revision})\n",
                    env!("CARGO_PKG_VERSION")
                )
            );
        }
    }
    assert!(!archive.exists());
    assert_eq!(std::fs::read_to_string(config).unwrap(), malformed);
}

#[test]
fn an_answer_from_another_protocol_is_refused_before_it_is_printed() {
    // A daemon from an earlier build still put the outcome in the document.
    let older = Answer::ok(serde_json::json!({
        "outcome": "rows", "unit": "sessions", "rows": [], "count": 0
    }));
    for name in ["query", "list", "measure sessions"] {
        let command = crate::commands::command(name).unwrap_or_else(|| panic!("{name}"));
        let refused = super::decoded(command, &older).unwrap_err();
        assert_eq!(
            refused.code.as_deref(),
            Some("daemon_protocol_mismatch"),
            "{name}"
        );
        assert_eq!(
            refused.kind,
            command_surface::FailureKind::NotRunning,
            "{name}"
        );
    }
    let failed: Answer<serde_json::Value> =
        Answer::error(crate::protocol::failure("not_found", "nothing"));
    assert!(super::decoded(crate::commands::command("query").unwrap(), &failed).is_ok());
}

/// A query answer document of the declared type, with these columns and rows.
fn rows_document(
    columns: serde_json::Value,
    rows: serde_json::Value,
    count: u64,
) -> serde_json::Value {
    let context = crate::operation::Context::standalone(
        crate::operation::Invocation::new(
            polylogue_model::Timestamp::parse_rfc3339("2026-03-01T12:00:00Z").unwrap(),
            "synthetic-print".into(),
            polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
        ),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::UTC,
            jiff::civil::Time::constant(4, 0, 0, 0),
        )
        .unwrap(),
    );
    let rows = crate::protocol::query::Rows {
        read_at: crate::protocol::query::Position {
            currency: None,
            generation: "synthetic".into(),
            index: 1,
            archive: 1,
            cursor: "synthetic:1:1".into(),
        },
        unit: "messages".into(),
        shape: polylogue_query::Shape::Rows,
        columns: serde_json::from_value(columns).unwrap(),
        rows: serde_json::from_value(rows).unwrap(),
        count,
        next: None,
        ranked: false,
        notes: vec![],
        incomplete: vec![],
        resolved: crate::protocol::query::Resolved {
            context: context.provenance(),
            times: vec![],
        },
        definitions: vec![],
        unknown: Default::default(),
        why: None,
    };
    serde_json::to_value(crate::protocol::query::Answer::Rows(Box::new(rows))).unwrap()
}
