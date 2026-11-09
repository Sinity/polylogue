use super::*;
use serde_json::json;

/// A value the command line accepts for a required argument.
fn sample(a: &Arg) -> &'static str {
    match a.kind {
        Kind::Choice(choices) => choices[0],
        Kind::Integer => "1",
        Kind::Json => "{}",
        _ => "claude-code:s1",
    }
}

/// A command line naming `c` with every required argument.
fn argv(c: &Command) -> Vec<&'static str> {
    let mut argv = vec!["polylogue"];
    if c.cli != Cli::NoSubcommand {
        argv.extend(c.name.split(' '));
    }
    for a in c.args.iter().filter(|a| a.required && a.default.is_none()) {
        if !a.positional {
            argv.push(Box::leak(
                format!("--{}", a.name.replace('_', "-")).into_boxed_str(),
            ));
        }
        argv.push(sample(a));
    }
    argv
}

/// The arguments a command line sends, prepared in a fixed directory.
fn cli(argv: &[&str]) -> (&'static Command, Values, serde_json::Value) {
    let (c, values, arguments) =
        crate::cli::request_arguments(argv, std::path::Path::new("/home/me/repo"))
            .unwrap_or_else(|e| panic!("{argv:?}: {e}"));
    (c, values, serde_json::Value::Object(arguments))
}

#[test]
fn every_declared_cli_command_is_reachable() {
    for c in COMMANDS.iter() {
        let argv = argv(c);
        let (found, _) = crate::cli::parse(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        assert_eq!(found.name, c.name, "{argv:?}");
    }
}

#[test]
fn the_command_line_has_nothing_outside_the_table() {
    let root = crate::cli::command();
    for sub in root.get_subcommands() {
        let name = sub.get_name();
        if sub.has_subcommands() {
            assert!(GROUPS.iter().any(|g| g.name == name), "group {name}");
            for leaf in sub.get_subcommands() {
                let full = format!("{name} {}", leaf.get_name());
                assert!(
                    COMMANDS
                        .iter()
                        .any(|command| command.cli_words().join(" ") == full),
                    "{full} is not in the table"
                );
            }
        } else {
            assert!(
                COMMANDS
                    .iter()
                    .any(|command| command.cli_words().join(" ") == name),
                "{name} is not in the table"
            );
        }
    }
    for g in GROUPS {
        assert!(
            COMMANDS
                .iter()
                .any(|c| c.name.starts_with(&format!("{} ", g.name))),
            "group {} has no commands",
            g.name
        );
    }
}

#[test]
fn mcp_offers_one_tool_per_family_over_the_commands_marked_for_it() {
    let tools: Vec<String> = crate::mcp::tools()
        .into_iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(tools, Tool::ALL.map(|tool| tool.name().to_owned()).to_vec());
    for tool in Tool::ALL {
        assert!(
            crate::mcp::operations(tool).next().is_some(),
            "{}",
            tool.name()
        );
    }
}

#[test]
fn only_daemon_rows_go_over_mcp_and_http() {
    for c in COMMANDS.iter().filter(|c| c.tool.is_some() || c.http) {
        assert!(
            !matches!(c.run, Run::Local(_) | Run::ConfiguredLocal(_)),
            "{} runs locally",
            c.name
        );
    }
    // Every daemon row is declared once, with its handler, to the shared
    // declarations; nothing else is.
    let declared: Vec<&str> = declarations()
        .descriptions()
        .iter()
        .map(|description| description.spec.name.as_str())
        .collect();
    let rows: Vec<&str> = COMMANDS
        .iter()
        .filter(|c| c.registration().is_some())
        .map(|c| c.name)
        .collect();
    assert_eq!(declared, rows);
}

#[test]
fn cli_help_lists_every_command() {
    let help = crate::cli::command()
        .try_get_matches_from(["polylogue", "--help"])
        .unwrap_err();
    assert_eq!(help.kind(), clap::error::ErrorKind::DisplayHelp);
    let help = help.to_string();
    let words: std::collections::BTreeSet<_> = COMMANDS
        .iter()
        .filter(|command| command.cli == Cli::Subcommand)
        .map(|command| command.name.split(' ').next().unwrap())
        .collect();
    for word in words {
        assert!(
            help.lines()
                .any(|line| line.split_whitespace().next() == Some(word)),
            "CLI help misses `{word}`"
        );
    }
}

#[test]
fn one_operation_makes_one_request_on_every_surface() {
    let (_, _, from_cli) = cli(&[
        "polylogue",
        "read",
        "claude-code:s1",
        "--view",
        "dialogue",
        "--page-tokens",
        "100",
        "--show",
        "inherited",
    ]);
    let from_mcp = json!({"target": "claude-code:s1", "view": "dialogue", "page_tokens": 100, "show": ["inherited"]});
    let from_http = declarations()
        .parse_http_query(
            "read",
            &[
                ("target".into(), "claude-code:s1".into()),
                ("view".into(), "dialogue".into()),
                ("page_tokens".into(), "100".into()),
                ("show".into(), "inherited".into()),
            ],
            "read-http",
        )
        .unwrap()
        .arguments;
    let resolved = |arguments: serde_json::Value| {
        let arguments: crate::protocol::read::Arguments =
            serde_json::from_value(arguments).unwrap();
        crate::daemon::service::read::ReadRequest::from_arguments(arguments, None).unwrap()
    };
    let a = resolved(from_cli);
    for other in [resolved(from_mcp), resolved(from_http)] {
        assert_eq!(format!("{other:?}"), format!("{a:?}"));
    }
    assert_eq!(a.options.view, polylogue_model::read::View::Dialogue);
    assert_eq!(a.budget, Some(100));
}

#[test]
fn query_commands_carry_every_option() {
    let (c, _, arguments) = cli(&[
        "polylogue",
        "--why",
        "measure",
        "survival-turns",
        "repo:x",
        "--fields",
        "id,cost",
    ]);
    let measure: crate::protocol::query_aliases::Measure =
        serde_json::from_value(arguments).unwrap();
    assert_eq!(
        crate::daemon::service::handlers::measure_query(c.name, &measure),
        "repo:x | measure \"survival-turns\""
    );
    assert!(measure.why);
    assert_eq!(measure.fields, ["id", "cost"]);

    let (c, _, arguments) = cli(&["polylogue", "repo:x", "since:7d"]);
    assert_eq!(c.name, "query");
    assert_eq!(arguments["query"], "repo:x since:7d");
}

#[test]
fn typed_read_validation_owns_defaults_and_conditional_requirements() {
    let decode = |arguments: serde_json::Value| {
        serde_json::from_value::<crate::protocol::read::Arguments>(arguments)
            .map_err(|e| e.to_string())
            .and_then(|arguments| {
                crate::daemon::service::read::ReadRequest::from_arguments(arguments, None)
            })
    };
    for invalid in [
        json!({"target":"claude-code:s1","pages":2}),
        json!({"target":"claude-code:s1","page_tokens":"many"}),
        json!({"target":"claude-code:s1","page_tokens":0}),
        json!({"target":"claude-code:s1","view":"sideways"}),
        json!({}),
        json!({"target":"claude-code:s1","to":"synthetic-output"}),
    ] {
        assert!(decode(invalid.clone()).is_err(), "{invalid}");
    }
    let request = decode(json!({"target":"claude-code:s1"})).unwrap();
    assert_eq!(
        request.options.view,
        polylogue_model::read::View::Transcript
    );
    assert_eq!(request.options.output_lines, 100);
    // A read without `page_tokens` is a page under the default budget, with
    // a continuation, never the whole session in one answer.
    assert_eq!(request.delivery, polylogue_model::read::ReadDelivery::Page);
    assert_eq!(
        request.budget, None,
        "the daemon applies its default page budget"
    );
}

#[test]
fn mcp_schemas_follow_the_arguments() {
    let tools = crate::mcp::tools();
    for command in COMMANDS.iter().filter(|command| command.tool.is_some()) {
        let tool = tools
            .iter()
            .find(|tool| tool["name"] == command.tool.unwrap().name())
            .unwrap();
        let schema = &tool["inputSchema"];
        assert_eq!(schema["additionalProperties"], false, "{}", command.name);
        for arg in command
            .args_for(Surface::Mcp)
            .filter(|arg| !arg.unlisted && !matches!(arg.name, "cursor" | "budget"))
        {
            assert!(
                schema["properties"]["arguments"]["properties"]
                    .get(arg.name)
                    .is_some(),
                "{} {}",
                command.name,
                arg.name
            );
        }
    }
    let input = |name: &str| {
        declarations()
            .descriptions()
            .iter()
            .find(|description| description.spec.name == name)
            .unwrap()
            .input_schema
            .clone()
    };
    // A continuation is the same arguments with a cursor; there is no
    // second argument form.
    let text = serde_json::to_string(&input("read")).unwrap();
    assert!(!text.contains("root_run") && !text.contains("ResumeArguments"));
    // A write's retry identity travels beside its arguments.
    for command in COMMANDS
        .iter()
        .filter(|c| c.writes && c.registration().is_some())
    {
        assert!(
            input(command.name)["properties"]
                .get("request_id")
                .is_none(),
            "{}",
            command.name
        );
    }
}

#[test]
fn list_without_a_query_names_no_query() {
    let (c, _, arguments) = cli(&["polylogue", "list", "--limit", "5"]);
    assert_eq!(c.name, "list");
    assert_eq!(arguments, json!({"limit": 5}));
}

#[test]
fn every_measurement_is_a_command_and_named_in_the_query_help() {
    let query = command("query").unwrap();
    for m in MEASURES {
        let c = command(&format!("measure {}", m.name)).unwrap();
        assert_eq!(c.help, m.help);
        assert_eq!(c.args.iter().any(|a| a.name == "query"), !m.whole_archive);
        assert!(query.help.contains(m.name), "{}", m.name);
    }
    for u in polylogue_query::ast::Unit::all() {
        assert!(query.help.contains(u.name()), "{}", u.name());
    }
}

#[test]
fn chat_send_reads_attached_files_on_the_command_line() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.md"), b"synthetic").unwrap();
    let (_, _, arguments) = crate::cli::request_arguments(
        &[
            "polylogue",
            "chat",
            "send",
            "chatgpt",
            "draft",
            "this",
            "--model",
            "X",
            "--attach",
            "notes.md",
        ],
        dir.path(),
    )
    .unwrap();
    let send: crate::protocol::browser::ChatSend =
        serde_json::from_value(serde_json::Value::Object(arguments)).unwrap();
    assert_eq!(
        (send.text.as_str(), send.model.as_str(), send.submit),
        ("draft this", "X", false)
    );
    assert_eq!(send.files.len(), 1);
    assert_eq!(
        (
            send.files[0].name.as_str(),
            send.files[0].media_type.as_str()
        ),
        ("notes.md", "text/plain")
    );
    assert_eq!(send.files[0].bytes, b"synthetic");
    // Over MCP there is no file to read.
    assert!(
        command("chat send")
            .unwrap()
            .args_for(Surface::Mcp)
            .all(|a| a.name != "attach")
    );
}

mod client_answers;

#[test]
fn request_id_travels_beside_the_arguments_on_every_surface() {
    let (_, _, arguments) = cli(&[
        "polylogue",
        "assert",
        r#"[{"kind":"note","value":"b","links":[{"role":"about","reference":"repo:x"}]}]"#,
        "--request-id",
        "retry-one",
    ]);
    let assert: crate::protocol::assertion::Assert = serde_json::from_value(arguments).unwrap();
    assert_eq!(assert.assertions.len(), 1);
    assert_eq!(assert.assertions[0].value, Some(json!("b")));
    let root = crate::cli::command();
    assert!(
        root.get_arguments()
            .any(|arg| arg.get_id() == "request_id" && arg.is_global_set())
    );
    for command in COMMANDS.iter().filter(|command| command.writes) {
        for surface in [Surface::Cli, Surface::Mcp, Surface::Http] {
            assert!(
                !command
                    .args_for(surface)
                    .any(|arg| arg.name == "request_id"),
                "{} {surface:?}",
                command.name
            );
        }
        if let Some(tool) = command.tool {
            let listed = crate::mcp::tools()
                .into_iter()
                .find(|listed| listed["name"] == tool.name())
                .unwrap();
            let schema = &listed["inputSchema"];
            assert!(schema["properties"]["request_id"].is_object());
            assert!(
                schema["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("request_id"))
            );
        }
    }
}

#[test]
fn measure_options_build_the_same_query_as_the_query_surface() {
    for (name, option, metric) in [
        ("facets", "--option", "deep"),
        ("outlook", "--option", "example:linear"),
        ("compare", "--option", "origin,words"),
        ("daily", "--metric", "tokens"),
    ] {
        let (command, _, arguments) = cli(&[
            "polylogue",
            "measure",
            name,
            option,
            metric,
            "sessions where origin:codex",
        ]);
        let measure: crate::protocol::query_aliases::Measure =
            serde_json::from_value(arguments).unwrap();
        assert_eq!(
            crate::daemon::service::handlers::measure_query(command.name, &measure),
            format!(r#"sessions where origin:codex | measure "{name}:{metric}""#)
        );
    }
    let (command, _, arguments) = cli(&["polylogue", "measure", "daily", "--metric", "tokens"]);
    let measure: crate::protocol::query_aliases::Measure =
        serde_json::from_value(arguments).unwrap();
    assert_eq!(
        crate::daemon::service::handlers::measure_query(command.name, &measure),
        r#"sessions | measure "daily:tokens""#
    );
}

mod assertions;
