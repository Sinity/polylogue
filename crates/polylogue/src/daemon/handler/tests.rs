//! Each operation is declared once with its handler; every surface reaches
//! the same typed handler. Ported case by case from the tests of the
//! typed adapters this module replaces.
use super::*;
use crate::commands::{COMMANDS, Cli};
use serde_json::json;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn request(
    operation: &str,
    arguments: Value,
    request_id: &str,
) -> command_surface::RunnableRequest {
    let mut request = crate::client::request(operation, arguments, request_id.into());
    request.deadline = command_surface::Deadline::None;
    request
}

fn description(name: &str) -> &'static command_surface::OperationDescription {
    crate::commands::declarations()
        .descriptions()
        .iter()
        .find(|description| description.spec.name == name)
        .unwrap()
}

fn policy(name: &str, arguments: Value, surface: Surface) -> Result<Policy, Failure> {
    let row = crate::commands::command(name).unwrap();
    row.registration()
        .unwrap()
        .admit(&arguments, row, &Caller { surface, cwd: None })
}

#[test]
fn typed_query_runs_through_the_shared_table_and_socket() {
    let daemon = crate::test_daemon::start("typed operation example");
    let table = crate::commands::table(Arc::clone(&daemon.daemon));
    let envelope = request("query", json!({"query":"sessions"}), "query-example");
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
    let table = crate::commands::table(Arc::clone(&daemon.daemon));
    for arguments in [
        json!({"author":"operator"}),
        json!({"paths":["some-input"]}),
        json!({"query":42}),
    ] {
        let answer = runtime().block_on(table.call(
            request("query", arguments, "refusal"),
            command_surface::Via::Mcp,
            Default::default(),
        ));
        assert_eq!(answer.exit_code(), 2, "{answer:?}");
        assert!(answer.value().is_none());
    }
}

#[test]
fn declaration_and_typed_query_schema_have_the_same_argument_names() {
    let command = crate::commands::command("query").unwrap();
    let declared = command
        .args_for(Surface::Mcp)
        .filter(|arg| !matches!(arg.name, "cursor" | "budget"))
        .map(|arg| arg.name.to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let schema = &description("query").input_schema;
    let typed = schema["properties"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(typed, declared);
    assert_eq!(schema["additionalProperties"], false);
    let output = &description("query").output_schema;
    assert!(output.to_string().contains("\"read_at\""));
}

#[test]
fn status_operation_uses_its_typed_answer_on_the_shared_socket_path() {
    let daemon = crate::test_daemon::start("typed status example");
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &request("status", json!({}), "status-wire"),
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

/// Every row a person types declares exactly its typed request's fields.
/// Rows the hooks and the extension send take their typed request as JSON;
/// `measure NAME` rows share one request with the options each offers;
/// `chat send` sends the files its `--attach` names.
#[test]
fn each_typed_declaration_exposes_exactly_its_runtime_arguments() {
    for command in COMMANDS.iter().filter(|command| {
        command.registration().is_some()
            && !(command.cli == Cli::Hidden && command.args.is_empty())
            && !command.name.starts_with("measure ")
            && command.prepare.is_none()
    }) {
        let declared = command
            .args_for(Surface::Mcp)
            .filter(|arg| !matches!(arg.name, "cursor" | "budget"))
            .map(|arg| arg.name.to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let schema = &description(command.name).input_schema;
        let typed = closed_argument_names(schema, schema);
        assert_eq!(typed, declared, "{}", command.name);
    }
}

#[test]
fn typed_sql_empty_rows_are_successful_empty_reads() {
    let daemon = crate::test_daemon::start("SQL operation example");
    let socket = crate::client::call(
        &daemon.daemon.archive.socket(),
        &request(
            "sql",
            json!({"statement": "SELECT 1 AS example WHERE 0"}),
            "sql-empty",
        ),
    )
    .unwrap();
    assert_eq!(socket.exit_code(), 1, "{socket:?}");
    assert_eq!(socket.http_status(), Some(200));
    assert_eq!(socket.value(), Some(&json!([])));
}

#[test]
fn the_socket_rejects_another_operations_arguments_before_execution() {
    let daemon = crate::test_daemon::start("typed socket validation");
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &request("status", json!({"query":"sessions"}), "bad-status-wire"),
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
        &request(
            "schema infer",
            json!({"origin":"claude-code", "all":true}),
            "schema-selection-wire",
        ),
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
    let (_, _, arguments) = crate::cli::request_arguments(
        &[
            "polylogue",
            "schema",
            "infer",
            "--all",
            "--output",
            "/synthetic/schema.json",
        ],
        std::path::Path::new("/"),
    )
    .unwrap();
    assert!(arguments.get("output").is_none());
    assert!(
        serde_json::from_value::<crate::protocol::schema::Infer>(Value::Object(arguments)).is_ok()
    );
}

#[test]
fn typed_setting_catalog_is_a_real_table_answer() {
    let daemon = crate::test_daemon::start("typed settings catalog example");
    let raw = crate::test_daemon::call(&daemon.daemon, "setting", json!({}));
    let Some(data) = raw.value().cloned() else {
        panic!("catalog producer failed: {raw:?}");
    };
    let _: crate::protocol::setting::Answer =
        serde_json::from_value(data).expect("catalog producer emits its declared answer");
    let table = crate::commands::table(Arc::clone(&daemon.daemon));
    let answer = runtime().block_on(table.call(
        request("setting", json!({}), "catalog-example"),
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
    for (arguments, id) in [
        (
            json!({"key":"browser.actions", "machine":true}),
            "invalid-catalog-example",
        ),
        (json!({"machine":"*"}), "invalid-host-example"),
    ] {
        let invalid = runtime().block_on(table.call(
            request("setting", arguments, id),
            command_surface::Via::Cli,
            Default::default(),
        ));
        assert!(invalid.value().is_none(), "{invalid:?}");
    }
}

#[test]
fn typed_import_uses_the_same_path_authority_and_recursive_answer_schema() {
    let relative = std::path::Path::new("relative.json");
    assert!(resolve_path(None, relative, "import path").is_err());
    assert_eq!(
        resolve_path(
            Some(std::path::Path::new("/synthetic/native")),
            relative,
            "import path"
        )
        .unwrap(),
        std::path::PathBuf::from("/synthetic/native/relative.json")
    );
    let daemon = crate::test_daemon::start("relative import refusal");
    let refused = crate::test_daemon::Call::new("import", json!({"paths":["relative.json"]}))
        .surface(Surface::Mcp)
        .answer(&daemon.daemon);
    assert_eq!(
        refused
            .failure()
            .and_then(|failure| failure.code.as_deref()),
        Some("bad_request"),
        "{refused:?}"
    );
    let schema = description("import").output_schema.to_string();
    assert!(schema.contains("ImportReport"));
    assert!(schema.contains("user_state"));
    assert!(
        policy(
            "import",
            json!({"paths":[],"author":"operator"}),
            Surface::Mcp
        )
        .is_err()
    );
}

#[test]
fn mixed_setting_reads_use_scope_fences_from_the_decoded_request() {
    assert!(crate::commands::command("setting").unwrap().writes);
    let read = policy("setting", json!({}), Surface::Http).unwrap();
    let write = policy(
        "setting",
        json!({"key":"browser.actions","value":"on"}),
        Surface::Http,
    )
    .unwrap();
    // A read takes a read scope and keeps the caller's read deadline; a
    // write keeps its result instead of inheriting one.
    assert!(read.read_only && read.read_phase);
    assert!(!write.read_only && !write.read_phase);
}

#[test]
fn typed_write_identity_comes_from_the_request_on_every_surface() {
    let command = crate::commands::command("setting").unwrap();
    for surface in [Surface::Cli, Surface::Http, Surface::Mcp] {
        assert!(
            !command
                .args_for(surface)
                .any(|arg| arg.name == "request_id")
        );
        assert!(policy("setting", json!({}), surface).is_ok());
        assert!(
            policy(
                "setting",
                json!({"key":"browser.actions","request_id":"in-arguments"}),
                surface
            )
            .is_err()
        );
    }
    let daemon = crate::test_daemon::start("write identity from the request");
    let written =
        crate::test_daemon::Call::new("setting", json!({"key":"browser.actions","value":"on"}))
            .id("write-request-id")
            .answer(&daemon.daemon);
    crate::test_answers::ok(written);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request("write-request-id")
            .unwrap()
            .is_some()
    );
}

#[test]
fn measurement_answers_retain_the_actual_resolved_context_on_the_shared_socket() {
    let daemon = crate::test_daemon::start_with_clock(
        "measurement provenance example",
        polylogue_model::WallClock::fixed("2026-03-01T12:00:00Z".parse().unwrap()),
    );
    for (text, paged) in [
        ("sessions | measure usage", false),
        ("sessions | measure time", true),
    ] {
        let answer = crate::client::call_ready(
            &daemon.daemon.archive,
            &request(
                "query",
                json!({"query":text}),
                &format!("measurement-context-{paged}"),
            ),
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
fn named_context_budget_and_identity_survive_the_socket() {
    let daemon = crate::test_daemon::start("context budget example");
    let mut envelope = request(
        "brief",
        json!({"cwd":"/synthetic","source":"resume"}),
        "context-budget-example",
    );
    envelope.budget_tokens = command_surface::TokenBudget::Tokens(200);
    // A hook's brief records its delivery, so it writes; its read phase
    // still has a deadline.
    let hook = policy("brief", envelope.arguments.clone(), Surface::Hook).unwrap();
    assert!(!hook.read_only && hook.read_phase);
    let person = policy("brief", envelope.arguments.clone(), Surface::Cli).unwrap();
    assert!(person.read_only);
    for (budget, option) in [
        (command_surface::TokenBudget::Default, None),
        (command_surface::TokenBudget::Unlimited, Some("none")),
        (command_surface::TokenBudget::Tokens(200), Some("200")),
    ] {
        let mut selected = envelope.clone();
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
        let (_, values, arguments) =
            crate::cli::request_arguments(&words, std::path::Path::new("/")).unwrap();
        assert_eq!(Value::Object(arguments), selected.arguments);
        assert_eq!(
            crate::cli::parse_budget(values.text("budget").as_deref()).unwrap(),
            budget
        );
    }
    let answer = crate::client::call_ready(&daemon.daemon.archive, &envelope).unwrap();
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let prepared: crate::protocol::context::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert!(prepared.prepared.text.len() <= 800);
    assert!(prepared.prepared.text.contains("context-budget-example"));
    let mut unsupported = envelope;
    unsupported.operation = "query".into();
    unsupported.arguments = json!({"query":"sessions"});
    let refused = crate::client::call_ready(&daemon.daemon.archive, &unsupported).unwrap();
    assert_eq!(
        refused.failure().map(|failure| failure.message.as_str()),
        Some("operation does not prepare budgeted context"),
        "{refused:?}"
    );
}

#[test]
fn brief_arguments_carry_the_start_source() {
    let arguments: crate::protocol::context::Brief =
        serde_json::from_value(json!({"cwd":"/x","source":"resume"})).unwrap();
    assert_eq!(arguments.source, crate::briefing::StartSource::Resume);
}

#[test]
fn answer_serialization_keeps_source_values_keys_and_empty_fields() {
    let token = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let data = json!({
        "nested": [token.clone(), ""], (token.clone()): {"code": 0, "known": false},
        "cursor": token.clone(), "authored": {"cursor": token.clone()}
    });
    let response = command_surface::Answer::ok(data.clone());
    let encoded = serde_json::to_vec(&response).unwrap();
    let decoded: command_surface::Answer<Value> = serde_json::from_slice(&encoded).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Ok(actual),
        ..
    } = decoded
    else {
        panic!("expected success");
    };
    assert_eq!(actual, data);
    assert_eq!(
        actual["nested"][0].as_str().unwrap().as_bytes(),
        token.as_bytes()
    );
}

#[test]
fn answer_errors_keep_credential_shaped_messages_and_hints() {
    let token = format!("ghp_{}", "q".repeat(36));
    let mut failure = crate::protocol::failure("bad_request", token.clone());
    failure.hint = Some(token.clone());
    let response = command_surface::Answer::<Value>::error(failure);
    let encoded = serde_json::to_value(response).unwrap();
    assert_eq!(encoded["error"]["message"], token);
    assert_eq!(encoded["error"]["hint"], token);
}
