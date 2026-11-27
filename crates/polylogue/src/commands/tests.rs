use super::*;
use serde_json::json;

/// A command line naming `c` with every required argument.
fn argv(c: &Command) -> Vec<&'static str> {
    let mut argv = vec!["polylogue"];
    argv.extend(c.name.split(' '));
    for a in crate::cli::arguments_of(c)
        .into_iter()
        .filter(|argument| argument.required)
    {
        if !matches!(
            a.spec.form,
            command_surface::ArgumentForm::Positional
                | command_surface::ArgumentForm::Words
                | command_surface::ArgumentForm::Rest
        ) {
            argv.push(Box::leak(
                format!("--{}", a.spec.name.replace('_', "-")).into_boxed_str(),
            ));
        }
        if a.spec.form == command_surface::ArgumentForm::Flag {
            continue;
        }
        let value = a.choices.first().cloned().unwrap_or_else(|| {
            if matches!(&a.kind, command_surface::ValueKind::List(_))
                && matches!(
                    a.spec.form,
                    command_surface::ArgumentForm::Positional
                        | command_surface::ArgumentForm::Named
                )
            {
                return "[]".into();
            }
            fn sample(kind: &command_surface::ValueKind) -> &'static str {
                match kind {
                    command_surface::ValueKind::Text => "claude-code:s1",
                    command_surface::ValueKind::Boolean => "true",
                    command_surface::ValueKind::Json => "1",
                    command_surface::ValueKind::List(item) => sample(item),
                }
            }
            sample(&a.kind).into()
        });
        argv.push(Box::leak(value.into_boxed_str()));
    }
    argv
}

/// The request arguments a row declares, calling-process options left out.
fn request_arguments_of(c: &Command) -> Vec<&'static command_surface::ArgumentSpec> {
    declarations()
        .spec_named(c.name)
        .unwrap_or_else(|| panic!("{} is declared", c.name))
        .arguments
        .iter()
        .filter(|argument| !argument.calling_process)
        .collect()
}

/// The arguments a command line sends, prepared in a fixed directory.
fn cli(
    argv: &[&str],
) -> (
    &'static Command,
    command_surface::ParsedCommand,
    serde_json::Value,
) {
    let (c, values, arguments) =
        crate::cli::request_arguments(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
    (c, values, serde_json::Value::Object(arguments))
}

#[test]
fn every_declared_cli_command_is_reachable() {
    for c in COMMANDS.iter().filter(|c| c.cli != Cli::Line) {
        let argv = argv(c);
        let (found, _) = crate::cli::parse(&argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        assert_eq!(found.name, c.name, "{argv:?}");
    }
}

#[test]
fn the_command_line_has_nothing_outside_the_table() {
    fn visit(command: &clap::Command, prefix: &str) {
        for sub in command.get_subcommands() {
            let full = if prefix.is_empty() {
                sub.get_name().to_owned()
            } else {
                format!("{prefix} {}", sub.get_name())
            };
            if sub.has_subcommands() {
                assert!(
                    COMMANDS.iter().any(|command| command
                        .cli_words()
                        .join(" ")
                        .starts_with(&format!("{full} "))),
                    "group {full} has no declared commands"
                );
                visit(sub, &full);
            } else {
                assert!(
                    COMMANDS
                        .iter()
                        .any(|command| command.cli_words().join(" ") == full),
                    "{full} is not in the table"
                );
            }
        }
    }
    visit(&crate::cli::command(), "");
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
            !matches!(
                c.run,
                Run::Local(..) | Run::ConfiguredLocal(..) | Run::ClientLocal(..)
            ),
            "{} runs locally",
            c.name
        );
    }
    // Every CLI row has one shared input declaration. Only daemon rows
    // expose MCP and HTTP; local rows remain process-owned.
    let declared: Vec<&str> = declarations()
        .descriptions()
        .iter()
        .map(|description| description.spec.name.as_str())
        .collect();
    let rows: Vec<&str> = COMMANDS.iter().map(|c| c.name).collect();
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
fn mcp_schemas_follow_the_arguments() {
    let tools = crate::mcp::tools();
    for command in COMMANDS.iter().filter(|command| command.tool.is_some()) {
        let tool = tools
            .iter()
            .find(|tool| tool["name"] == command.tool.unwrap().name())
            .unwrap();
        let schema = &tool["inputSchema"];
        assert_eq!(schema["additionalProperties"], false, "{}", command.name);
        for arg in request_arguments_of(command)
            .into_iter()
            .filter(|arg| arg.listed)
        {
            assert!(
                schema["properties"]["arguments"]["properties"]
                    .get(&arg.name)
                    .is_some(),
                "{} {}",
                command.name,
                arg.name
            );
        }
    }
}

#[test]
fn line_listing_carries_its_selection_and_page_size() {
    let words = ["sessions", "--page-size", "5"].map(str::to_owned);
    let parsed = crate::cli::line::parse(&words, &Default::default(), false).unwrap();
    let request = crate::cli::run_line::request(&parsed).unwrap();
    assert_eq!(request.operation, "rows");
    let rows: crate::protocol::line::Rows = serde_json::from_value(request.arguments).unwrap();
    assert_eq!(rows.selection.as_deref(), Some("sessions"));
    assert_eq!(rows.page_size.unwrap().get(), 5);
    assert!(rows.steps.is_empty());
}

mod client_answers;

#[test]
fn request_id_travels_beside_the_arguments_on_every_surface() {
    let words = [
        "session:claude-code:s1",
        "note",
        "b",
        "--request-id",
        "retry-one",
    ]
    .map(str::to_owned);
    let parsed =
        crate::cli::line::parse(&words, &crate::cli::line::SavedNames::default(), false).unwrap();
    let request = crate::cli::run_line::request(&parsed).unwrap();
    assert_eq!(request.request_id, "retry-one");
    assert_eq!(
        request.arguments["steps"],
        json!([{"verb":"note","text":"b"}])
    );
    let root = crate::cli::command();
    assert!(
        root.get_arguments()
            .any(|arg| arg.get_long() == Some("request-id") && arg.is_global_set())
    );
    for command in COMMANDS.iter().filter(|command| command.writes) {
        assert!(
            !declarations()
                .spec_named(command.name)
                .unwrap()
                .arguments
                .iter()
                .any(|arg| arg.name == "request_id"),
            "{}",
            command.name
        );
        if let Some(tool) = command.tool {
            let listed = crate::mcp::tools()
                .into_iter()
                .find(|listed| listed["name"] == tool.name())
                .unwrap();
            let schema = &listed["inputSchema"];
            assert!(schema["properties"]["request_id"].is_object());
            // A call may leave the ID out; parsing derives it (docs/interfaces.md,
            // Operation table: `{operation, arguments, request_id?, …}`).
            assert!(
                !schema["required"]
                    .as_array()
                    .unwrap()
                    .contains(&json!("request_id"))
            );
        }
    }
}

mod annotations;

#[test]
fn status_parses_verification_manifest_and_explicit_vectors() {
    let (_, _, arguments) = cli(&[
        "polylogue",
        "status",
        "--verify",
        "--manifest",
        "checked.tsv",
    ]);
    let arguments: crate::protocol::status::Arguments = serde_json::from_value(arguments).unwrap();
    assert!(arguments.verify);
    assert_eq!(
        arguments.manifest.unwrap(),
        std::path::PathBuf::from("checked.tsv")
    );
    let (_, _, arguments) = cli(&["polylogue", "status", "--vectors"]);
    let arguments: crate::protocol::status::Arguments = serde_json::from_value(arguments).unwrap();
    assert!(arguments.vectors && !arguments.verify);
}

/// The value an option takes, from its schema: the type of a present value
/// (null leaves it out), its format and its choices; a repeatable option
/// takes the unit of one value.
fn unit(property: &serde_json::Value) -> String {
    use serde_json::Value;
    if let Some(reference) = property["$ref"].as_str() {
        return reference.rsplit('/').next().unwrap_or(reference).to_owned();
    }
    if let Some(branches) = property["anyOf"]
        .as_array()
        .or_else(|| property["oneOf"].as_array())
    {
        let mut units: Vec<String> = branches
            .iter()
            .filter(|branch| branch["type"] != "null")
            .map(unit)
            .collect();
        units.sort();
        units.dedup();
        return units.join("|");
    }
    let kinds: Vec<&str> = match &property["type"] {
        Value::String(kind) => vec![kind.as_str()],
        Value::Array(kinds) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => vec![],
    };
    let kind = kinds
        .into_iter()
        .filter(|kind| *kind != "null")
        .collect::<Vec<_>>()
        .join("|");
    let mut text = kind.clone();
    if let Some(format) = property["format"].as_str() {
        text = format!("{text}:{format}");
    }
    if let Some(choices) = property["enum"].as_array() {
        text = format!("{text}{choices:?}");
    }
    // A repeatable option takes the unit of one value.
    if kind == "array" {
        return unit(&property["items"]);
    }
    text
}

/// One option name has one unit wherever a person types it: in every
/// command a person types and in every verb's step. Program rows under
/// `internal` keep their own fields.
#[test]
fn no_flag_name_has_two_units() {
    use std::collections::BTreeMap;
    let mut units: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();
    let mut note = |name: &str, unit: String, owner: String| {
        units
            .entry(name.to_owned())
            .or_default()
            .entry(unit)
            .or_default()
            .push(owner);
    };
    let declarations = crate::commands::declarations();
    for command in COMMANDS.iter().filter(|command| !command.cli.hidden()) {
        let Some(description) = declarations
            .descriptions()
            .iter()
            .find(|description| description.spec.name == command.name)
        else {
            continue;
        };
        if let Some(properties) = description.input_schema["properties"].as_object() {
            for (name, property) in properties {
                note(name, unit(property), command.name.to_owned());
            }
        }
    }
    let steps = crate::daemon::verbs::declarations();
    for verb in steps.verbs() {
        if let Some(properties) = verb.input_schema()["properties"].as_object() {
            for (name, property) in properties {
                note(name, unit(property), verb.verb().to_owned());
            }
        }
    }
    // `status --after N` counts changes while the feed's and a page's
    // `--after` name a position; the status design owns that name.
    if let Some(after) = units.get_mut("after") {
        after.retain(|_, owners| owners.as_slice() != ["status"]);
    }
    let twice: Vec<_> = units
        .into_iter()
        .filter(|(_, units)| units.len() > 1)
        .collect();
    assert!(twice.is_empty(), "{twice:#?}");
}
