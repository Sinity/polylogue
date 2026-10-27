use super::*;
use serde_json::json;

/// A value the command line accepts for a required argument.
fn sample(a: &Arg) -> &'static str {
    match a.kind {
        Kind::Choice(choices) => choices[0],
        Kind::Integer => "1",
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
    let Action::Request(build) = c.action else {
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
                assert!(command(&full).is_some(), "{full} is not in the table");
            }
        } else {
            assert!(command(name).is_some(), "{name} is not in the table");
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
fn mcp_offers_exactly_the_table_commands_marked_for_it() {
    let tools: Vec<String> = crate::mcp::tools()
        .into_iter()
        .map(|t| t.name.into_owned())
        .collect();
    let declared: Vec<String> = COMMANDS
        .iter()
        .filter(|c| c.mcp)
        .map(Command::mcp_name)
        .collect();
    assert_eq!(tools, declared);
}

#[test]
fn only_daemon_requests_go_over_mcp_and_http() {
    for c in COMMANDS.iter().filter(|c| c.mcp || c.http) {
        assert!(
            matches!(c.action, Action::Request(_)),
            "{} runs locally",
            c.name
        );
    }
}

#[test]
fn command_words_are_every_first_word() {
    let words = command_words();
    for w in [
        "status",
        "list",
        "note",
        "measure",
        "embed",
        "hooks",
        "keep",
        "work",
        "browser-host",
    ] {
        assert!(words.iter().any(|x| x == w), "{w} missing from {words:?}");
    }
    assert!(!words.iter().any(|w| w == "query"));
    assert_eq!(
        words.len(),
        words
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
}

#[test]
fn the_readme_lists_every_command() {
    let readme = include_str!("../../../../README.md");
    for w in command_words().iter().filter(|w| *w != "browser-host") {
        assert!(readme.contains(&format!("`{w}")), "README misses `{w}`");
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
        "--budget",
        "100",
        "--with-parents",
    ])
    .unwrap();
    let mcp = Values::from_json(
        read,
        Surface::Mcp,
        json!({"session": "claude-code:s1", "view": "dialogue", "budget": 100, "with_parents": true})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let http = Values::from_pairs(
        read,
        Surface::Http,
        vec![
            ("session".into(), "claude-code:s1".into()),
            ("view".into(), "dialogue".into()),
            ("budget".into(), "100".into()),
            ("with_parents".into(), "true".into()),
        ],
    )
    .unwrap();
    let a = request_json(read, &cli, &cli_caller());
    assert_eq!(a, request_json(read, &mcp, &mcp_caller()));
    assert_eq!(a, request_json(read, &http, &cli_caller()));
    assert_eq!(a["view"], "dialogue");
    assert_eq!(a["budget"], 100);
}

#[test]
fn note_lists_and_queries_carry_every_option() {
    let (c, v) = crate::cli::parse(&[
        "polylogue",
        "note",
        "list",
        "repo:x",
        "--kind",
        "lesson,handoff",
        "--agents",
        "--limit",
        "3",
    ])
    .unwrap();
    let r = request_json(c, &v, &cli_caller());
    assert_eq!(
        r["text"],
        r#"notes where (target:"repo:x") (kind:lesson OR kind:handoff) agent:true"#
    );
    assert_eq!(r["page_size"], 3);

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
    assert_eq!(r["text"], "repo:x | measure survival-turns");
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
fn an_agent_writes_notes_as_itself() {
    let add = command("note add").unwrap();
    let args = json!({"targets": ["repo:x"], "body": "b"});
    let v = Values::from_json(add, Surface::Mcp, args.as_object().unwrap().clone()).unwrap();
    assert_eq!(
        request_json(add, &v, &mcp_caller())["note"]["author"],
        "agent"
    );
    let (_, v) = crate::cli::parse(&["polylogue", "note", "add", "repo:x", "-m", "b"]).unwrap();
    assert_eq!(
        request_json(add, &v, &cli_caller())["note"]["author"],
        "operator"
    );
    let (_, v) =
        crate::cli::parse(&["polylogue", "note", "add", "repo:x", "-m", "b", "--agent"]).unwrap();
    assert_eq!(
        request_json(add, &v, &cli_caller())["note"]["author"],
        "agent"
    );
}

#[test]
fn values_are_checked_against_the_table() {
    let read = command("read").unwrap();
    let check = |v: serde_json::Value| {
        Values::from_json(read, Surface::Mcp, v.as_object().unwrap().clone())
    };
    let e = check(json!({"session": "claude-code:s1", "pages": 2})).unwrap_err();
    assert!(e.contains("no argument `pages`"), "{e}");
    let e = check(json!({"session": "claude-code:s1", "budget": "many"})).unwrap_err();
    assert!(e.contains("whole number"), "{e}");
    let e = check(json!({"session": "claude-code:s1", "view": "sideways"})).unwrap_err();
    assert!(e.contains("one of full"), "{e}");
    let e = check(json!({})).unwrap_err();
    assert!(e.contains("needs `session`"), "{e}");
    // Arguments that act on the calling machine are not offered.
    let export = command("export").unwrap();
    let e = Values::from_json(
        export,
        Surface::Http,
        json!({"sessions": ["claude-code:s1"], "out": "/tmp/x"})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap_err();
    assert!(e.contains("no argument `out`"), "{e}");
    // Defaults fill what is left out.
    let v = check(json!({"session": "claude-code:s1"})).unwrap();
    assert_eq!(v.text("view").as_deref(), Some("full"));
}

#[test]
fn a_write_may_carry_its_request_id() {
    let add = command("note add").unwrap();
    let args = json!({"targets": ["repo:x"], "body": "b", "request_id": "r1"});
    let v = Values::from_json(add, Surface::Http, args.as_object().unwrap().clone()).unwrap();
    assert_eq!(v.request_id(), "r1");
    let read = command("read").unwrap();
    let args = json!({"session": "claude-code:s1", "request_id": "r1"});
    assert!(Values::from_json(read, Surface::Http, args.as_object().unwrap().clone()).is_err());
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
    let s = schema(command("read").unwrap());
    assert_eq!(s["required"], json!(["session"]));
    assert_eq!(s["properties"]["budget"]["type"], "integer");
    assert_eq!(s["properties"]["view"]["default"], "full");
    assert_eq!(s["additionalProperties"], false);
    let s = schema(command("note add").unwrap());
    assert!(s["properties"]["request_id"].is_object());
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
    assert_eq!(r["text"], "sessions | measure daily:tokens");
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

#[test]
fn measure_options_build_the_same_query_as_the_query_surface() {
    for (name, metric) in [
        ("facets", "deep"),
        ("pace", "60"),
        ("outlook", "sample:linear"),
        ("correlation", "message_count:word_count"),
        ("distribution", "median_tool_call_ms"),
    ] {
        let (command, values) = crate::cli::parse(&[
            "polylogue",
            "measure",
            name,
            if matches!(name, "correlation" | "distribution") {
                "--metric"
            } else {
                "--option"
            },
            metric,
            "sessions where origin:codex",
        ])
        .unwrap();
        let request = request_json(command, &values, &cli_caller());
        assert_eq!(
            request["text"],
            format!("sessions where origin:codex | measure {name}:{metric}")
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
        "sessions where origin:codex | measure daily:tokens"
    );
}
