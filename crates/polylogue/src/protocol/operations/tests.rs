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
    };
    let answer = runtime().block_on(table.call(
        envelope.clone(),
        command_surface::Origin::Cli,
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
        };
        let answer = runtime().block_on(table.call(
            envelope,
            command_surface::Origin::Mcp,
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

#[test]
fn each_typed_declaration_exposes_exactly_its_runtime_arguments() {
    for command in crate::commands::COMMANDS
        .iter()
        .filter(|command| command.register.is_some())
    {
        let declared = command
            .args_for(Surface::Mcp)
            .filter(|arg| arg.name != "request_id")
            .map(|arg| arg.name.to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        let schema = (command.typed_schemas.unwrap())().0;
        let typed = schema["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(typed, declared, "{}", command.name);
        assert_eq!(schema["additionalProperties"], false, "{}", command.name);
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
    .unwrap()
    .answer();
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
        },
    )
    .unwrap();
    assert!(matches!(answer, Answer::Error { .. }));
    assert_eq!(answer.http_status(), Some(400));
}
