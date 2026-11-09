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

fn cli_caller() -> Caller {
    Caller {
        surface: Surface::Cli,
        cwd: Some("/home/me/repo".into()),
        json: true,
    }
}

fn mcp_caller() -> Caller {
    Caller {
        surface: Surface::Mcp,
        cwd: Some("/home/me/repo".into()),
        json: true,
    }
}

/// The request as JSON without its request ID, which is fresh per call.
fn request_json(c: &Command, v: &Values, caller: &Caller) -> serde_json::Value {
    let (Action::Request(build) | Action::Stream { build, .. }) = c.action else {
        panic!("{} is local", c.name)
    };
    let mut r = serde_json::to_value(build(c, v, caller).unwrap()).unwrap();
    r.as_object_mut().unwrap().remove("request_id");
    r
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
fn only_daemon_requests_go_over_mcp_and_http() {
    for c in COMMANDS.iter().filter(|c| c.tool.is_some() || c.http) {
        assert!(
            matches!(c.action, Action::Request(_) | Action::Stream { .. }),
            "{} runs locally",
            c.name
        );
    }
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
    let read = command("read").unwrap();
    let (_, cli) = crate::cli::parse(&[
        "polylogue",
        "read",
        "claude-code:s1",
        "--view",
        "dialogue",
        "--page-tokens",
        "100",
        "--show",
        "inherited",
    ])
    .unwrap();
    let mcp = Values::from_json(
        read,
        Surface::Mcp,
        json!({"target": "claude-code:s1", "view": "dialogue", "page_tokens": 100, "show": ["inherited"]})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let http = Values::from_pairs(
        read,
        Surface::Http,
        vec![
            ("target".into(), "claude-code:s1".into()),
            ("view".into(), "dialogue".into()),
            ("page_tokens".into(), "100".into()),
            ("show".into(), "inherited".into()),
        ],
    )
    .unwrap();
    let a = request_json(read, &cli, &cli_caller());
    assert_eq!(a, request_json(read, &mcp, &mcp_caller()));
    assert_eq!(a, request_json(read, &http, &cli_caller()));
    assert_eq!(a["options"]["view"], "dialogue");
    assert_eq!(a["budget"], 100);
}

#[test]
fn query_commands_carry_every_option() {
    let (c, v) = crate::cli::parse(&[
        "polylogue",
        "--why",
        "measure",
        "survival-turns",
        "repo:x",
        "--fields",
        "id,cost",
    ])
    .unwrap();
    let r = request_json(c, &v, &cli_caller());
    assert_eq!(r["text"], "repo:x | measure \"survival-turns\"");
    assert_eq!(r["why"], true);
    assert_eq!(r["fields"], json!(["id", "cost"]));

    let (c, v) = crate::cli::parse(&["polylogue", "repo:x", "since:7d"]).unwrap();
    assert_eq!(c.name, "query");
    assert_eq!(
        request_json(c, &v, &cli_caller())["text"],
        "repo:x since:7d"
    );
}

#[test]
fn typed_read_validation_owns_defaults_and_conditional_requirements() {
    let read = command("read").unwrap();
    let build = read.build_typed.unwrap();
    let caller = mcp_caller();
    for invalid in [
        json!({"target":"claude-code:s1","pages":2}),
        json!({"target":"claude-code:s1","page_tokens":"many"}),
        json!({"target":"claude-code:s1","view":"sideways"}),
        json!({}),
        json!({"target":"claude-code:s1","to":"synthetic-output"}),
    ] {
        assert!(build(invalid, &caller, "read-invalid").is_err());
    }
    let request = build(json!({"target":"claude-code:s1"}), &caller, "read-default").unwrap();
    let Request::Read { options, .. } = request else {
        panic!("full read");
    };
    assert_eq!(options.view, polylogue_model::read::View::Transcript);
    assert_eq!(options.output_lines, 100);
}

#[test]
fn http_callers_name_their_directory() {
    let resume = command("continue").unwrap();
    let v = Values::default();
    let Action::Request(build) = resume.action else {
        unreachable!()
    };
    let http = Caller {
        surface: Surface::Http,
        cwd: None,
        json: true,
    };
    assert!(build(resume, &v, &http).unwrap_err().contains("`cwd`"));
    let r = request_json(resume, &v, &cli_caller());
    assert_eq!(r["cwd"], "/home/me/repo");
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
            .filter(|arg| !arg.unlisted && !matches!(arg.name, "request_id" | "cursor"))
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
    let read = command("read").unwrap();
    let s = read.typed_schemas.unwrap()().0;
    // A continuation is the same arguments with a cursor; there is no
    // second argument form.
    let text = serde_json::to_string(&s).unwrap();
    assert!(!text.contains("root_run") && !text.contains("ResumeArguments"));
    for name in ["assert", "assert-query", "revise", "withdraw"] {
        let s = schema(command(name).unwrap());
        assert!(s["properties"].get("request_id").is_none(), "{name}");
    }
}

#[test]
fn list_without_a_query_runs_the_sessions_unit() {
    let (c, v) = crate::cli::parse(&["polylogue", "list", "--limit", "5"]).unwrap();
    let r = request_json(c, &v, &cli_caller());
    assert_eq!(r["op"], "query");
    assert_eq!(r["text"], "sessions");
    assert_eq!(r["page_size"], 5);
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
    let (c, v) =
        crate::cli::parse(&["polylogue", "measure", "daily", "--metric", "tokens"]).unwrap();
    let r = request_json(c, &v, &cli_caller());
    assert_eq!(r["text"], r#"sessions | measure "daily:tokens""#);
}

#[test]
fn chat_send_reads_attached_files_on_the_command_line() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.md"), b"synthetic").unwrap();
    let (c, v) = crate::cli::parse(&[
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
    ])
    .unwrap();
    let caller = Caller {
        cwd: Some(dir.path().to_owned()),
        ..cli_caller()
    };
    let Request::ChatSend { action, files, .. } = (match c.action {
        Action::Request(build) => build(c, &v, &caller).unwrap(),
        _ => panic!("chat send is a daemon request"),
    }) else {
        panic!("chat send builds ChatSend")
    };
    assert_eq!(
        (action.text.as_str(), action.model.as_str(), action.submit),
        ("draft this", "X", false)
    );
    assert_eq!(files.len(), 1);
    assert_eq!(
        (files[0].name.as_str(), files[0].media_type.as_str()),
        ("notes.md", "text/plain")
    );
    assert_eq!(files[0].bytes, b"synthetic");
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
fn request_id_is_declared_for_all_write_surfaces() {
    let (assert, values) = crate::cli::parse(&[
        "polylogue",
        "assert",
        r#"[{"kind":"note","value":"b","links":[{"role":"about","reference":"repo:x"}]}]"#,
        "--request-id",
        "retry-one",
    ])
    .unwrap();
    assert_eq!(values.request_id(), "retry-one");
    let Action::Request(build) = assert.action else {
        panic!("assert must build a request")
    };
    let Request::Assert {
        request_id,
        assertions,
    } = build(assert, &values, &cli_caller()).unwrap()
    else {
        panic!("wrong assertion operation")
    };
    assert_eq!(request_id, "retry-one");
    assert_eq!(assertions.len(), 1);
    assert_eq!(assertions[0].value, Some(json!("b")));
    let root = crate::cli::command();
    assert!(
        root.get_arguments()
            .any(|arg| arg.get_id() == "request_id" && arg.is_global_set())
    );
    for command in COMMANDS.iter().filter(|command| command.writes) {
        // CLI identity is global; typed remote identity belongs to the envelope.
        assert!(
            !command
                .args_for(Surface::Cli)
                .any(|arg| arg.name == "request_id")
        );
        for surface in [Surface::Mcp, Surface::Http] {
            assert_eq!(
                command
                    .args_for(surface)
                    .filter(|arg| arg.name == "request_id")
                    .count(),
                usize::from(command.register.is_none()),
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
    for (name, metric) in [
        ("facets", "deep"),
        ("outlook", "example:linear"),
        ("compare", "origin,words"),
    ] {
        let (command, values) = crate::cli::parse(&[
            "polylogue",
            "measure",
            name,
            "--option",
            metric,
            "sessions where origin:codex",
        ])
        .unwrap();
        let request = request_json(command, &values, &cli_caller());
        assert_eq!(
            request["text"],
            format!(r#"sessions where origin:codex | measure "{name}:{metric}""#)
        );
    }
}

#[test]
fn daily_series_is_a_metric_option() {
    let (command, values) = crate::cli::parse(&[
        "polylogue",
        "measure",
        "daily",
        "--metric",
        "tokens",
        "sessions where origin:codex",
    ])
    .unwrap();
    assert_eq!(
        request_json(command, &values, &cli_caller())["text"],
        r#"sessions where origin:codex | measure "daily:tokens""#
    );
}

mod assertions;

#[test]
fn mcp_read_default_budget_pages() {
    // A read without `page_tokens` is a page under the default budget, with
    // a continuation, never the whole session in one answer.
    let read = command("read").unwrap();
    let build = read.build_typed.unwrap();
    let request = build(
        json!({"target":"claude-code:s1"}),
        &mcp_caller(),
        "read-page",
    )
    .unwrap();
    let Request::Read {
        delivery, budget, ..
    } = request
    else {
        panic!("read request");
    };
    assert_eq!(delivery, polylogue_model::read::ReadDelivery::Page);
    assert_eq!(budget, None, "the daemon applies its default page budget");
}
