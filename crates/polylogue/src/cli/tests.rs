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
        let reply = serde_json::to_string(&Response::Ok {
            data: serde_json::json!({"ready": true}),
        })
        .unwrap();
        std::io::Write::write_all(&mut &stream, format!("{reply}\n").as_bytes()).unwrap();
        drop(lock);
    });
    let response = send_with_timeout(&archive, &Request::Status, None).unwrap();
    assert!(matches!(response, Response::Ok { .. }));
    server.join().unwrap();
}

#[test]
fn manual_is_an_offline_topic() {
    let manual = COMMANDS.iter().find(|c| c.name == "manual").unwrap();
    assert!(matches!(manual.action, Action::Local(_)));
    let matches = command()
        .try_get_matches_from(["polylogue", "manual"])
        .unwrap();
    assert_eq!(run_local(&matches), Some(std::process::ExitCode::SUCCESS));
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
        let (actual, _) = crate::protocol::wire::incoming(&text).unwrap().unwrap();
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
                cursor: None,
                with_blobs: false,
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
                cursor: None,
                with_blobs: false,
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
