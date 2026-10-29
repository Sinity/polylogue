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
    let answer =
        runtime().block_on(table.call(envelope, command_surface::Origin::Cli, Default::default()));
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let typed: crate::protocol::query::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert_eq!(typed.rows().unwrap().len(), 1);
    let command = crate::commands::command("query").unwrap();
    let values = Values::from_json(
        command,
        Surface::Cli,
        json!({"query":"sessions"}).as_object().unwrap().clone(),
    )
    .unwrap();
    let Action::Request(build) = command.action else {
        unreachable!()
    };
    let request = build(
        command,
        &values,
        &Caller {
            surface: Surface::Cli,
            cwd: None,
            json: true,
        },
    )
    .unwrap();
    let socket = crate::client::send(&daemon.daemon.archive.socket(), &request)
        .unwrap()
        .answer();
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
    let response = crate::client::send(
        &daemon.daemon.archive.socket(),
        &crate::daemon::Request::Status,
    )
    .unwrap();
    let answer = response.answer();
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
