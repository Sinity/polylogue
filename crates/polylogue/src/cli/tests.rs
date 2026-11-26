use super::*;

fn status() -> command_surface::RunnableRequest {
    crate::client::request("status", serde_json::json!({}), "status".into())
}

#[test]
fn a_missing_daemon_is_down_to_the_client() {
    let dir = tempfile::tempdir().unwrap();
    let err = crate::client::call(&dir.path().join("daemon.sock"), &status()).unwrap_err();
    assert!(is_down(&err));
    assert!(!Archive::at(dir.path()).lock_held());
    let err = crate::client::call_ready(&Archive::at(dir.path()), &status()).unwrap_err();
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
        let listener = crate::socket::bind(&socket).unwrap();
        loop {
            let (stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            if std::io::BufRead::read_line(&mut std::io::BufReader::new(&stream), &mut line)
                .unwrap()
                == 0
            {
                // The readiness connection checks listening without a request.
                continue;
            }
            let reply = serde_json::to_string(&command_surface::Answer::ok(
                serde_json::json!({"ready": true}),
            ))
            .unwrap();
            std::io::Write::write_all(
                &mut &stream,
                format!("event: answer\ndata: {reply}\n\n").as_bytes(),
            )
            .unwrap();
            break;
        }
        drop(lock);
    });
    // A deadline longer than the start; the status class's own is shorter.
    let mut request = status();
    request.deadline = command_surface::Deadline::Milliseconds(30_000);
    let response = crate::client::call_ready(&archive, &request).unwrap();
    assert_eq!(
        crate::test_answers::ok(response),
        serde_json::json!({"ready": true})
    );
    server.join().unwrap();
}

#[test]
fn clients_wait_with_deadlines_only_for_reads_and_explicit_stream_limits() {
    use std::time::Duration;
    let stream = || {
        crate::client::request(
            "rows",
            serde_json::json!({"selection":"sessions", "steps":[{"verb":"read","delivery":"stream","format":"json"}]}),
            "stream".into(),
        )
    };
    let mut limited = stream();
    limited.deadline = command_surface::Deadline::Milliseconds(120_000);
    let mut build =
        crate::client::request("embed", serde_json::json!({"yes":true}), "build".into());
    build.deadline = command_surface::Deadline::Milliseconds(120_000);
    for (request, expected) in [
        (status(), Some(Duration::from_millis(80 + 50))),
        (stream(), None),
        (limited, Some(Duration::from_millis(120_000 + 50))),
        (build, None),
    ] {
        assert_eq!(
            crate::client::connection_deadline(&request, None),
            expected,
            "{}",
            request.operation
        );
    }
}

#[test]
fn only_read_only_requests_end_the_client_wait_at_their_read_deadline() {
    // The writer can finish after the read deadline, so a writing line keeps
    // waiting for its committed outcome.
    let writing = crate::client::request(
        "rows",
        serde_json::json!({"selection":"messages","steps":[{"verb":"note","text":"a lesson","as":"lesson"}]}),
        "write-1".into(),
    );
    assert_eq!(crate::client::connection_deadline(&writing, None), None);
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
        r"\x1b]8;;https://example.test\x1b\link\x1b]8;;\x1b\\x7f\u{202e}"
    );
}

#[test]
fn an_answer_frame_without_its_end_is_the_daemon_gone() {
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
            "event: answer\ndata: {}",
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
        request_id: "incomplete-answer-example".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let answer = crate::client::call_ready(&archive, &call).unwrap();
    assert_eq!(
        answer.failure().and_then(|failure| failure.code.as_deref()),
        Some("not_running"),
        "{answer:?}"
    );
    server.join().unwrap();
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
    // Colour is decided for each sink, as specified in terminal.md.
    for (stdout, stderr, no_color, term, stdout_color, stderr_color) in [
        (false, false, false, Some("xterm"), false, false),
        (true, false, false, Some("xterm"), true, false),
        (false, true, false, Some("xterm"), false, true),
        (true, true, false, Some("xterm"), true, true),
        (true, true, true, Some("xterm"), false, false),
        (true, true, false, Some("dumb"), false, false),
    ] {
        let output = OutputConfig::new(stdout, stderr, no_color, term);
        let style = style(output);
        let text = crate::commands::declarations()
            .render_text(
                "rows",
                &command_surface::Answer::ok(
                    serde_json::json!({"kind":"listing","reply":data.clone()}),
                ),
                style,
            )
            .unwrap();
        assert_eq!(text.contains("\x1b[1m"), stdout_color, "stdout: {output:?}");
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
        assert_eq!(
            error.contains("\x1b[1m"),
            stderr_color,
            "stderr: {output:?}"
        );
        assert!(!error.contains("\x1b[31m"));
        assert!(error.contains("\\x1b[31m"));
        assert!(error.ends_with("polylogue: try: query λ\n"));
        let json = print::json(&data, style);
        assert!(!json.contains('\x1b'));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&json).unwrap(),
            original
        );
    }
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
    // Colour is decided for each sink, as specified in terminal.md.
    for (stdout, stderr, no_color, term, stdout_color, stderr_color) in [
        (true, false, false, Some("xterm"), true, false),
        (false, true, false, Some("xterm"), false, true),
        (false, false, false, Some("xterm"), false, false),
        (true, true, false, Some("xterm"), true, true),
        (true, true, true, Some("xterm"), false, false),
        (true, true, false, Some("dumb"), false, false),
    ] {
        let output = crate::config::OutputConfig::new(stdout, stderr, no_color, term);
        assert_eq!(
            parse_text(&help, output).contains("\x1b[1m"),
            stdout_color,
            "stdout: {output:?}"
        );
        assert_eq!(
            parse_text(&error, output).contains("\x1b[1m"),
            stderr_color,
            "stderr: {output:?}"
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
    for name in ["rows", "status"] {
        let command = crate::commands::command(name).unwrap_or_else(|| panic!("{name}"));
        let refused = crate::commands::declarations()
            .validate_answer(command.name, &older)
            .map_err(|failure| crate::protocol::protocol_mismatch(command.name, failure.message))
            .unwrap_err();
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
    let failed: Answer<serde_json::Value> = Answer::error(crate::protocol::failure(
        polylogue_model::Code::NotFound,
        "nothing",
    ));
    assert!(
        crate::commands::declarations()
            .validate_answer("rows", &failed)
            .is_ok()
    );
}

/// A query answer document of the declared type, with these columns and rows.
fn rows_document(
    columns: serde_json::Value,
    rows: serde_json::Value,
    count: u64,
) -> serde_json::Value {
    let context = polylogue_reading::operation::Context::standalone(
        polylogue_reading::operation::Invocation::new(
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
            freshness: None,
            generation: "synthetic".into(),
            index: 1,
            archive: 1,
            cursor: "synthetic:1:1".into(),
        },
        unit: "messages".into(),
        shape: polylogue_query::Shape::Rows,
        columns: serde_json::from_value(columns).unwrap(),
        rows: serde_json::from_value(rows).unwrap(),
        references: None,
        count: Some(count),
        next: None,
        ranked: false,
        notes: vec![],
        incomplete: vec![],
        resolved: crate::protocol::query::Resolved {
            context: context.run_context(),
            times: vec![],
        },
        definitions: vec![],
        unknown: Default::default(),
        why: None,
    };
    serde_json::to_value(crate::protocol::query::Answer::Rows(Box::new(rows))).unwrap()
}

#[test]
fn status_exit_code_is_degraded_when_answer_is_degraded() {
    let root = tempfile::tempdir().unwrap();
    let daemon = crate::test_daemon::start("status example");
    let startup = crate::config::Startup::for_test(daemon.config.clone(), root.path());
    // A whole status answer, read or degraded, with its running part unreadable.
    let mut data = crate::test_daemon::call(&daemon.daemon, "status", serde_json::json!({}))
        .value()
        .cloned()
        .expect("status answers with a value");
    data.as_object_mut().unwrap().insert(
        "running".into(),
        serde_json::json!({"error": "the live table is unavailable"}),
    );
    let degraded =
        command_surface::Answer::degraded(data.clone(), vec!["the index is behind".into()]);
    for line in [false, true] {
        let argv = if line {
            vec!["polylogue", "status", "--line"]
        } else {
            vec!["polylogue", "status"]
        };
        let (_, values) = parse(&argv).unwrap();
        assert_eq!(
            status_output(&values, &degraded, &startup.client, true),
            ExitCode::from(5),
            "line: {line}"
        );
        assert_eq!(
            status_output(
                &values,
                &command_surface::Answer::ok(data.clone()),
                &startup.client,
                true
            ),
            ExitCode::SUCCESS
        );
    }
}

fn resume_plan(harness: Option<crate::resume::Command>) -> crate::resume::Plan {
    let session = polylogue_model::SessionId::parse("claude-code:s1").unwrap();
    crate::resume::Plan {
        harness_command: harness,
        context_assisted: crate::resume::Command {
            program: "polylogue".into(),
            args: vec!["session:claude-code:s1".into(), "brief".into()],
            cwd: None,
        },
        session,
    }
}

/// docs/interfaces.md#exit-codes: a session that cannot be reopened here
/// is `refused` (3), whichever part is missing.
#[test]
fn resume_without_a_way_to_reopen_exits_3() {
    let output = crate::config::OutputConfig::new(false, false, true, None);
    let missing = tempfile::tempdir().unwrap().path().join("gone");
    for harness in [
        None,
        Some(crate::resume::Command {
            program: "claude".into(),
            args: vec![],
            cwd: None,
        }),
        Some(crate::resume::Command {
            program: "claude".into(),
            args: vec![],
            cwd: Some(missing.display().to_string()),
        }),
    ] {
        for replace in [false, true] {
            assert_eq!(run_plan(&resume_plan(harness.clone()), output, replace), 3);
        }
    }
}

/// A harness that cannot be started is an `io` failure (4), whether this
/// process waits for it or becomes it.
#[test]
fn resume_exec_failure_exits_4() {
    let output = crate::config::OutputConfig::new(false, false, true, None);
    let directory = tempfile::tempdir().unwrap();
    let harness = crate::resume::Command {
        program: directory
            .path()
            .join("no-such-harness")
            .display()
            .to_string(),
        args: vec!["--resume".into(), "s1".into()],
        cwd: Some(directory.path().display().to_string()),
    };
    for replace in [false, true] {
        assert_eq!(
            run_plan(&resume_plan(Some(harness.clone())), output, replace),
            4
        );
    }
}
