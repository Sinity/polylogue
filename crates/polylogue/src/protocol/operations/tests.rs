use super::*;
use serde_json::json;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn typed_query_runs_through_the_shared_table_and_socket() {
    let daemon = crate::test_daemon::start("typed operation witness");
    let table = table(Arc::clone(&daemon.daemon));
    let envelope = command_surface::Request {
        operation: "query".into(),
        arguments: json!({"query":"sessions"}),
        request_id: "query-witness".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let answer = runtime().block_on(table.call(
        envelope.clone(),
        command_surface::Via::Cli,
        Default::default(),
    ));
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let typed: crate::protocol::query::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert_eq!(typed.rows().unwrap().len(), 1);
    let socket = crate::client::call_ready(&daemon.daemon.archive, &envelope).unwrap();
    assert_eq!(
        socket.value().unwrap()["rows"],
        answer.value().unwrap()["rows"]
    );
    assert_eq!(socket.value().unwrap()["count"], 1);
}

#[test]
fn typed_query_refuses_unknown_fields_and_another_operations_arguments() {
    let daemon = crate::test_daemon::start("typed operation refusal");
    let table = table(Arc::clone(&daemon.daemon));
    for arguments in [
        json!({"author":"operator"}),
        json!({"paths":["some-input"]}),
        json!({"query":42}),
    ] {
        let envelope = command_surface::Request {
            operation: "query".into(),
            arguments,
            request_id: "refusal".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let answer =
            runtime().block_on(table.call(envelope, command_surface::Via::Mcp, Default::default()));
        assert_eq!(answer.exit_code(), 2, "{answer:?}");
        assert!(answer.value().is_none());
    }
}

#[test]
fn declaration_and_typed_query_schema_have_the_same_argument_names() {
    let command = crate::commands::command("query").unwrap();
    let declared = command
        .args_for(Surface::Mcp)
        .filter(|arg| !matches!(arg.name, "request_id" | "cursor"))
        .map(|arg| arg.name.to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let schema = (command.typed_schemas.unwrap())().0;
    let typed = schema["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(typed, declared);
    assert_eq!(schema["additionalProperties"], false);
    let output = (command.typed_schemas.unwrap())().1;
    assert!(output.to_string().contains("\"read_at\""));
}

#[test]
fn status_operation_uses_its_typed_answer_on_the_shared_socket_path() {
    let daemon = crate::test_daemon::start("typed status witness");
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &command_surface::Request {
            operation: "status".into(),
            arguments: json!({}),
            request_id: "status-wire".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
    )
    .unwrap();
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let status: crate::protocol::status::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert_eq!(status.pid, std::process::id());
    assert!(matches!(
        status.mode,
        crate::protocol::status::Mode::Standalone
    ));
    if let Some(counts) = status.store {
        assert_eq!(counts.messages, 1);
    }
    assert!(
        serde_json::from_value::<crate::protocol::status::Arguments>(json!({"author":"operator"}))
            .is_err()
    );
}

fn closed_argument_names(
    root: &serde_json::Value,
    schema: &serde_json::Value,
) -> std::collections::BTreeSet<String> {
    if let Some(reference) = schema["$ref"].as_str() {
        let pointer = reference.strip_prefix('#').expect("local schema reference");
        return closed_argument_names(
            root,
            root.pointer(pointer).expect("declared argument shape"),
        );
    }
    if let Some(branches) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(serde_json::Value::as_array)
    {
        return branches
            .iter()
            .flat_map(|branch| closed_argument_names(root, branch))
            .collect();
    }
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["additionalProperties"], false);
    schema["properties"]
        .as_object()
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

#[test]
fn each_typed_declaration_exposes_exactly_its_runtime_arguments() {
    for command in crate::commands::COMMANDS
        .iter()
        .filter(|command| command.register.is_some())
    {
        let declared = command
            .args_for(Surface::Mcp)
            .filter(|arg| !matches!(arg.name, "request_id" | "cursor"))
            .map(|arg| arg.name.to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let schema = (command.typed_schemas.unwrap())().0;
        let typed = closed_argument_names(&schema, &schema);
        assert_eq!(typed, declared, "{}", command.name);
    }
}

#[test]
fn typed_sql_empty_rows_are_successful_empty_reads() {
    let daemon = crate::test_daemon::start("SQL operation witness");
    let socket = crate::client::send(
        &daemon.daemon.archive.socket(),
        &crate::daemon::Request::Sql {
            statement: "SELECT 1 AS witness WHERE 0".into(),
        },
    )
    .unwrap();
    assert_eq!(socket.exit_code(), 1, "{socket:?}");
    assert_eq!(socket.http_status(), Some(200));
    assert_eq!(socket.value(), Some(&json!([])));
}

#[test]
fn canonical_socket_rejects_another_operations_arguments_before_execution() {
    let daemon = crate::test_daemon::start("typed socket validation");
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &command_surface::Request {
            operation: "status".into(),
            arguments: json!({"query":"sessions"}),
            request_id: "bad-status-wire".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
    )
    .unwrap();
    assert!(matches!(
        answer,
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(_),
            ..
        }
    ));
    assert_eq!(answer.http_status(), Some(400));
}

#[test]
fn typed_schema_selection_refuses_conflicting_origin_controls_on_the_socket() {
    let daemon = crate::test_daemon::start("typed schema selection");
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &command_surface::Request {
            operation: "schema infer".into(),
            arguments: json!({"origin":"claude-code", "all":true}),
            request_id: "schema-selection-wire".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
    )
    .unwrap();
    assert_eq!(answer.http_status(), Some(400));
    assert!(matches!(
        answer,
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(_),
            ..
        }
    ));
}

#[test]
fn local_schema_output_path_stays_out_of_the_typed_request() {
    let command = crate::commands::command("schema infer").unwrap();
    let values = Values::from_json(
        command,
        Surface::Cli,
        json!({"all":true,"output":"/synthetic/schema.json"})
            .as_object()
            .unwrap()
            .clone(),
    )
    .unwrap();
    let call = request(
        command,
        values,
        "schema-local-output".into(),
        command_surface::Deadline::None,
    );
    assert!(call.arguments.get("output").is_none());
    assert!(serde_json::from_value::<crate::protocol::schema::Infer>(call.arguments).is_ok());
}

#[test]
fn typed_setting_catalog_is_a_real_table_answer() {
    let daemon = crate::test_daemon::start("typed settings catalog witness");
    let raw = crate::daemon::service::handle(
        crate::daemon::Request::Setting {
            request_id: None,
            key: None,
            value: None,
            machine: None,
        },
        &daemon.daemon,
    );
    let Some(data) = raw.value().cloned() else {
        panic!("catalog producer failed: {raw:?}");
    };
    let _: crate::protocol::setting::Answer =
        serde_json::from_value(data).expect("catalog producer emits its declared DTO");
    let table = table(Arc::clone(&daemon.daemon));
    let answer = runtime().block_on(table.call(
        command_surface::Request {
            operation: "setting".into(),
            arguments: json!({}),
            request_id: "catalog-witness".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
        command_surface::Via::Cli,
        Default::default(),
    ));
    let catalog: Vec<polylogue_model::settings::Current> = serde_json::from_value(
        answer
            .value()
            .unwrap_or_else(|| panic!("catalog failed: {answer:?}"))
            .clone(),
    )
    .unwrap();
    assert!(
        catalog
            .iter()
            .any(|current| current.key == "browser.actions")
    );
    let invalid = runtime().block_on(table.call(
        command_surface::Request {
            operation: "setting".into(),
            arguments: json!({"key":"browser.actions", "machine":true}),
            request_id: "invalid-catalog-witness".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
        command_surface::Via::Cli,
        Default::default(),
    ));
    assert!(invalid.value().is_none());
    let invalid_host = runtime().block_on(table.call(
        command_surface::Request {
            operation: "setting".into(),
            arguments: json!({"machine":"*"}),
            request_id: "invalid-host-witness".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        },
        command_surface::Via::Cli,
        Default::default(),
    ));
    assert!(invalid_host.value().is_none());
}

#[test]
fn typed_import_uses_the_same_path_authority_and_recursive_answer_schema() {
    let command = crate::commands::command("import").unwrap();
    let build = command.build_typed.unwrap();
    let remote = crate::commands::Caller {
        surface: crate::commands::Surface::Mcp,
        cwd: None,
        json: true,
    };
    assert!(
        build(
            json!({"paths":["relative.json"]}),
            &remote,
            "import-witness"
        )
        .is_err()
    );
    let native = crate::commands::Caller {
        cwd: Some("/synthetic/native".into()),
        ..remote
    };
    let request = build(
        json!({"paths":["relative.json"]}),
        &native,
        "import-witness",
    )
    .unwrap();
    let crate::daemon::Request::Import { paths, request_id } = request else {
        panic!("import operation");
    };
    assert_eq!(request_id, "import-witness");
    assert_eq!(
        paths,
        vec![std::path::PathBuf::from("/synthetic/native/relative.json")]
    );
    let (_, answer) = command.typed_schemas.unwrap()();
    let schema = answer.to_string();
    assert!(schema.contains("ImportReport"));
    assert!(schema.contains("user_state"));
    assert!(
        build(
            json!({"paths":[],"author":"operator"}),
            &native,
            "import-witness"
        )
        .is_err()
    );
}

#[test]
fn mixed_setting_reads_use_scope_fences_from_the_admitted_request() {
    let command = crate::commands::command("setting").unwrap();
    assert!(command.writes);
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Http,
        cwd: None,
        json: true,
    };
    let build = command.build_typed.unwrap();
    let read = build(json!({}), &caller, "catalog-read").unwrap();
    let write = build(
        json!({"key":"browser.actions","value":"on"}),
        &caller,
        "catalog-write",
    )
    .unwrap();
    assert!(read_scope_required(Some(command), Some(&read)));
    assert!(!read_scope_required(Some(command), Some(&write)));
    assert_eq!(
        admitted_deadline(
            Some(command),
            Some(&read),
            command_surface::Deadline::Milliseconds(0)
        ),
        command_surface::Deadline::Milliseconds(0),
    );
    assert_eq!(
        admitted_deadline(
            Some(command),
            Some(&write),
            command_surface::Deadline::Milliseconds(0)
        ),
        command_surface::Deadline::None,
    );
}

#[test]
fn typed_write_identity_comes_from_the_envelope_on_every_surface() {
    let command = crate::commands::command("setting").unwrap();
    let build = command.build_typed.unwrap();
    for surface in [Surface::Cli, Surface::Http, Surface::Mcp] {
        let caller = Caller {
            surface,
            cwd: None,
            json: true,
        };
        assert!(
            !command
                .args_for(surface)
                .any(|arg| arg.name == "request_id")
        );
        let read = build(json!({}), &caller, "catalog-envelope-id").unwrap();
        assert!(matches!(
            read,
            crate::daemon::Request::Setting {
                request_id: None,
                ..
            }
        ));
        let write = build(
            json!({"key":"browser.actions","value":"on"}),
            &caller,
            "write-envelope-id",
        )
        .unwrap();
        assert!(
            matches!(write,crate::daemon::Request::Setting{request_id:Some(ref id),..} if id=="write-envelope-id")
        );
        assert!(
            build(
                json!({"key":"browser.actions","request_id":"in-arguments"}),
                &caller,
                "actual-envelope-id"
            )
            .is_err()
        );
    }
}

#[test]
fn measurement_answers_retain_the_actual_resolved_context_on_the_shared_socket() {
    let daemon = crate::test_daemon::start_with_clock(
        "measurement provenance witness",
        polylogue_model::WallClock::fixed("2026-03-01T12:00:00Z".parse().unwrap()),
    );
    for (text, paged) in [
        ("sessions | measure usage", false),
        ("sessions | measure time", true),
    ] {
        let answer = crate::client::call_ready(
            &daemon.daemon.archive,
            &command_surface::Request {
                operation: "query".into(),
                arguments: json!({"query":text}),
                request_id: format!("measurement-context-{paged}"),
                deadline: command_surface::Deadline::None,
                budget_tokens: Default::default(),
                cursor: None,
                fresh: false,
                format: Default::default(),
            },
        )
        .unwrap();
        let value = answer
            .value()
            .unwrap_or_else(|| panic!("{text}: {answer:?}"));
        let typed: crate::protocol::query::Answer = serde_json::from_value(value.clone()).unwrap();
        match &typed {
            crate::protocol::query::Answer::Measure(measure) => {
                assert!(!paged);
                assert_eq!(
                    measure.resolved.context.ran_at.to_string(),
                    "2026-03-01T12:00:00Z"
                );
            }
            crate::protocol::query::Answer::PagedMeasure(measure) => {
                assert!(paged);
                assert_eq!(
                    measure.resolved.context.ran_at.to_string(),
                    "2026-03-01T12:00:00Z"
                );
            }
            _ => panic!("actual measurement answer: {typed:?}"),
        }
        assert_eq!(serde_json::to_value(&typed).unwrap(), *value);
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("resolved");
        assert!(serde_json::from_value::<crate::protocol::query::Answer>(missing).is_err());
        let mut unknown = value.clone();
        unknown["caller_supplied_provenance"] = json!(true);
        assert!(serde_json::from_value::<crate::protocol::query::Answer>(unknown).is_err());
    }
    for schema in [
        schemars::schema_for!(crate::protocol::query::Measure).to_value(),
        schemars::schema_for!(crate::protocol::query::PagedMeasure).to_value(),
    ] {
        assert!(schema["properties"]["resolved"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("resolved"))
        );
        assert_eq!(schema["additionalProperties"], false);
    }
}

#[test]
fn named_context_budget_and_identity_survive_admission_and_socket_execution() {
    let daemon = crate::test_daemon::start("context budget witness");
    let command = crate::commands::command("brief").unwrap();
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Hook,
        cwd: None,
        json: true,
    };
    let envelope = command_surface::Request {
        operation: "brief".into(),
        arguments: json!({"cwd":"/synthetic","source":"resume"}),
        request_id: "context-budget-witness".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: command_surface::TokenBudget::Tokens(200),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let (cli_command, cli_values) = crate::cli::parse(&[
        "polylogue",
        "brief",
        "--cwd",
        "/synthetic",
        "--source",
        "resume",
        "--budget",
        "200",
    ])
    .unwrap();
    assert_eq!(
        request(
            cli_command,
            cli_values,
            envelope.request_id.clone(),
            envelope.deadline
        ),
        envelope
    );
    let original = envelope.clone();
    let internal = admit(command, &envelope, &caller).unwrap();
    assert!(
        matches!(internal, crate::daemon::Request::Context { request_id, budget_tokens: command_surface::TokenBudget::Tokens(200), sender: Some(sender), .. }
        if request_id == envelope.request_id && sender == "hook")
    );
    assert_eq!(envelope, original);
    for (label, budget, option) in [
        ("default", command_surface::TokenBudget::Default, None),
        (
            "unlimited",
            command_surface::TokenBudget::Unlimited,
            Some("none"),
        ),
        (
            "numeric",
            command_surface::TokenBudget::Tokens(200),
            Some("200"),
        ),
    ] {
        let mut selected = envelope.clone();
        selected.request_id = format!("context-budget-{label}");
        selected.budget_tokens = budget;
        let encoded = serde_json::to_value(&selected).unwrap();
        match budget {
            command_surface::TokenBudget::Default => {
                assert!(encoded.get("budget_tokens").is_none())
            }
            command_surface::TokenBudget::Unlimited => assert!(encoded["budget_tokens"].is_null()),
            command_surface::TokenBudget::Tokens(tokens) => {
                assert_eq!(encoded["budget_tokens"], tokens)
            }
        }
        let mut words = vec![
            "polylogue",
            "brief",
            "--cwd",
            "/synthetic",
            "--source",
            "resume",
        ];
        if let Some(option) = option {
            words.extend(["--budget", option]);
        }
        let (selected_command, values) = crate::cli::parse(&words).unwrap();
        assert_eq!(
            request(
                selected_command,
                values,
                selected.request_id.clone(),
                selected.deadline
            ),
            selected
        );
        assert!(matches!(admit(command, &selected, &caller).unwrap(),
            crate::daemon::Request::Context { budget_tokens, .. } if budget_tokens == budget));
    }
    let answer = crate::client::call_ready(&daemon.daemon.archive, &envelope).unwrap();
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let prepared: crate::protocol::context::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert!(prepared.prepared.text.len() <= 800);
    assert!(prepared.prepared.text.contains("context-budget-witness"));
    let mut unsupported = envelope;
    unsupported.operation = "query".into();
    unsupported.arguments = json!({"query":"sessions"});
    assert!(
        admit(
            crate::commands::command("query").unwrap(),
            &unsupported,
            &caller
        )
        .is_err()
    );
}
