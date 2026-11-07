//! MCP tool calls against a real daemon on a temporary archive.

use super::*;
use crate::test_daemon;
use serde_json::json;

/// The `tools/call` parameters that run `operation` (a command name) with
/// these arguments through its tool.
pub(crate) fn tool_call(operation: &str, arguments: Value, request_id: &str) -> Value {
    let command = crate::commands::command(operation).expect("a declared command");
    let tool = command.tool.expect("an MCP operation");
    json!({"name": tool.name(), "arguments": {
        "operation": command.name, "arguments": arguments, "request_id": request_id,
    }})
}

/// Runs `operation` through its tool; a name that is no command is called
/// as a tool name, so its refusal can be checked.
pub(crate) fn call(
    archive: &Archive,
    operation: &str,
    arguments: serde_json::Map<String, Value>,
) -> Result<String, String> {
    call_as(
        archive,
        operation,
        arguments,
        &uuid::Uuid::now_v7().to_string(),
    )
}

/// [`call`] with the caller's retry identity.
pub(crate) fn call_as(
    archive: &Archive,
    operation: &str,
    arguments: serde_json::Map<String, Value>,
    request_id: &str,
) -> Result<String, String> {
    let params = match crate::commands::command(operation).filter(|c| c.tool.is_some()) {
        Some(_) => tool_call(operation, Value::Object(arguments), request_id),
        None => {
            json!({"name": operation, "arguments": {"arguments": arguments, "request_id": request_id}})
        }
    };
    call_params(archive, params)
}

pub(crate) fn call_page(
    archive: &Archive,
    operation: &str,
    arguments: Value,
    cursor: Option<&str>,
) -> Result<String, String> {
    let mut params = tool_call(operation, arguments, &uuid::Uuid::now_v7().to_string());
    if let Some(cursor) = cursor {
        params["arguments"]["cursor"] = json!(cursor);
    }
    call_params(archive, params)
}

fn call_params(archive: &Archive, params: Value) -> Result<String, String> {
    call_with_limit(
        archive,
        params["name"].as_str().unwrap(),
        params["arguments"].as_object().unwrap().clone(),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )
    .and_then(|(text, _, failed)| if failed { Err(text) } else { Ok(text) })
}

/// The completion protocol method's answer for `text` at `position`, as
/// the completions value of the `complete` operation.
pub(crate) fn complete(
    archive: &Archive,
    text: &str,
    position: Option<u64>,
) -> Result<Value, String> {
    let mut meta = json!({});
    if let Some(position) = position {
        meta["polylogue/position"] = json!(position);
    }
    let reply = Session {
        initialized: true,
        ..Default::default()
    }
    .prepare(
        json!({"jsonrpc":"2.0", "id":1, "method":"completion/complete", "params":{
            "argument": {"name": "query", "value": text},
            "_meta": meta,
        }}),
    )
    .answer(archive)
    .unwrap();
    match reply.get("error") {
        Some(error) => Err(error["message"].as_str().unwrap_or_default().to_owned()),
        None => Ok(reply["result"]["_meta"]["polylogue/completions"].clone()),
    }
}

fn args(v: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn tools_answer_with_the_daemon_and_refuse_bad_calls() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);

    let status: serde_json::Value =
        serde_json::from_str(&call(&archive, "status", args(json!({}))).unwrap()).unwrap();
    assert_eq!(status["store"]["messages"], 1);

    // Every option of `read` reaches the daemon.
    let text = call(
        &archive,
        "read",
        args(json!({"target": "session:claude-code:s1", "format": "markdown"})),
    )
    .unwrap();
    let read: polylogue_model::read::ReadPage = serde_json::from_str(&text).unwrap();
    assert!(read_pages::body(&read).contains("walrus"), "{text}");

    let text = call(&archive, "query", args(json!({"query": "walrus"}))).unwrap();
    assert!(text.contains("claude-code:s1"), "{text}");

    let e = call(
        &archive,
        "read",
        args(json!({"target": "session:claude-code:s1", "pages": 1})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (bad_request)"), "{e}");
    let e = call(
        &archive,
        "excise",
        args(json!({"target": "session:claude-code:s1"})),
    )
    .unwrap_err();
    assert!(e.contains("no tool excise"), "{e}");
    let e = call(
        &archive,
        "read",
        args(json!({"target": "session:claude-code:nope"})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (not_found)"), "{e}");
}

pub(crate) mod read_pages;

mod events;

mod attachments;

mod explain;

fn initialize() -> Value {
    json!({"jsonrpc":"2.0", "id":"init", "method":"initialize", "params":{
        "protocolVersion":"2025-11-25", "capabilities":{}, "clientInfo":{"name":"test", "version":"1"}
    }})
}

#[test]
fn stdio_answers_initialize_tools_calls_and_errors() {
    let w = test_daemon::start("the walrus migration");
    let archive = Archive::at(&w.daemon.archive.root);
    let requests = [
        initialize().to_string(),
        json!({"jsonrpc":"2.0", "method":"notifications/initialized"}).to_string(),
        json!({"jsonrpc":"2.0", "id":2, "method":"tools/list"}).to_string(),
        json!({"jsonrpc":"2.0", "id":3, "method":"tools/call", "params":tool_call("status", json!({}), "status-jsonrpc")}).to_string(),
        json!({"jsonrpc":"2.0", "id":4, "method":"tools/call", "params":{"name":"unknown"}}).to_string(),
        "not JSON".to_owned(),
        json!({"jsonrpc":"2.0", "id":5, "method":"unknown"}).to_string(),
        json!({"jsonrpc":"2.0", "id":6, "method":"tools/call", "params":{"name":"polylogue_read", "arguments":[]}}).to_string(),
        json!({"jsonrpc":"2.0", "id":7, "method":"ping"}).to_string(),
    ].join("\n");
    let replies: Vec<Value> = std::thread::scope(|scope| {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixStream;
        let (mut client, server) = UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        let output = server.try_clone().unwrap();
        let worker = scope.spawn(|| serve(BufReader::new(server), output, &archive));
        writeln!(client, "{requests}").unwrap();
        let mut input = BufReader::new(client.try_clone().unwrap());
        let replies = (0..8)
            .map(|_| {
                let mut line = String::new();
                input.read_line(&mut line).unwrap();
                serde_json::from_str(&line).unwrap()
            })
            .collect();
        client.shutdown(std::net::Shutdown::Write).unwrap();
        worker.join().unwrap().unwrap();
        replies
    });
    assert_eq!(replies.len(), 8);
    let reply = |id: Value| replies.iter().find(|reply| reply["id"] == id).unwrap();
    assert_eq!(
        reply(json!("init"))["result"]["protocolVersion"],
        "2025-11-25"
    );
    assert_eq!(reply(json!(2))["result"]["tools"], json!(tools()));
    let result = &reply(json!(3))["result"];
    assert_eq!(result["isError"], false);
    let status: Value =
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(status["store"]["messages"], 1);
    assert_eq!(reply(json!(4))["result"]["isError"], true);
    assert_eq!(reply(Value::Null)["error"]["code"], -32700);
    assert_eq!(reply(json!(5))["error"]["code"], -32601);
    assert_eq!(reply(json!(6))["error"]["code"], -32602);
    assert_eq!(reply(json!(7))["result"], json!({}));
}

#[test]
fn discovery_and_inline_requests_keep_protocol_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let mut session = Session::default();
    let discovery = session
        .prepare(
            json!({"jsonrpc":"2.0", "id":1, "method":"server/discover", "params":{"_meta":meta}}),
        )
        .answer(&archive)
        .unwrap();
    assert_eq!(discovery["result"]["supportedVersions"], json!(VERSIONS));
    assert_eq!(discovery["result"]["resultType"], "complete");
    let list = session
        .prepare(json!({"jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{"_meta":meta}}))
        .answer(&archive)
        .unwrap();
    assert_eq!(list["result"]["tools"], json!(tools()));
    assert_eq!(list["result"]["resultType"], "complete");
    let invalid = session
        .prepare(json!({"jsonrpc":"2.0", "id":3, "method":"tools/list"}))
        .answer(&archive)
        .unwrap();
    assert_eq!(invalid["error"]["code"], -32602);
    let mut init = initialize();
    init["params"]["protocolVersion"] = json!("future");
    let answer = Session::default().prepare(init).answer(&archive).unwrap();
    assert_eq!(answer["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn protocol_ids_null_arguments_and_ping_follow_the_request_version() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let mut session = Session::default();
    session.prepare(initialize()).answer(&archive).unwrap();
    let id = json!(u64::MAX);
    let ping = session
        .prepare(json!({"jsonrpc":"2.0", "id":id, "method":"ping"}))
        .answer(&archive)
        .unwrap();
    assert_eq!(ping["id"], id);
    assert_eq!(ping["result"], json!({}));
    let call = session.prepare(json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":{"name":"polylogue_read", "arguments":null}}));
    match call {
        Reply::Tool { arguments, .. } => assert!(arguments.is_empty()),
        _ => panic!("null arguments must be an empty argument map"),
    }
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let ping = session
        .prepare(json!({"jsonrpc":"2.0", "id":3, "method":"ping", "params":{"_meta":meta}}))
        .answer(&archive)
        .unwrap();
    assert_eq!(ping["error"]["code"], -32601);
}

#[test]
fn malformed_request_ids_are_not_echoed_in_error_responses() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    for id in [json!({"wrong":1}), json!([1]), json!(true)] {
        let reply = Session::default()
            .prepare(json!({"jsonrpc":"2.0", "id":id, "method":"ping"}))
            .answer(&archive)
            .unwrap();
        assert_eq!(reply["id"], Value::Null);
        assert_eq!(reply["error"]["code"], -32600);
    }
}

/// law: mcp-instructions-are-the-card
#[test]
fn server_instructions_are_the_card_naming_every_tool() {
    let Reply::Message(info) = Session::default().prepare(initialize()) else {
        panic!("initialize must return server information");
    };
    let text = info["result"]["instructions"].as_str().unwrap();
    assert_eq!(text, crate::manual::card());
    for tool in Tool::ALL {
        assert!(text.contains(tool.name()), "{}", tool.name());
    }
    assert!(text.contains("`explain capabilities`"));
    assert!(text.len() <= 800, "the card is {} bytes", text.len());
}

#[test]
fn manual_query_examples_compile_through_the_public_command() {
    let w = test_daemon::start("manual query examples");
    let archive = Archive::at(&w.daemon.archive.root);
    for query in crate::manual::text()
        .split('`')
        .filter(|text| text.starts_with("sessions where ") || text.starts_with("messages where "))
    {
        call(
            &archive,
            "explain",
            args(json!({"subject":"query", "expression":query})),
        )
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    }
}

#[test]
fn errors_show_archived_controls_as_text() {
    let hostile = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";
    let mut failure = crate::protocol::failure("bad_request", hostile)
        .with_suggestion(command_surface::Suggestion::Query(hostile.into()));
    failure.code = Some(hostile.into());
    let failure = failure.with_candidates(vec![hostile.into()]);
    let text = super::error_text(&failure);
    assert!(
        !text
            .chars()
            .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}')
    );
    assert!(text.contains("\\x1b]52;c;YQ==\\x07"));
}

/// law: mcp-tool-access-is-uniform
#[test]
fn tool_annotations_follow_the_command_table() {
    let tools = tools();
    for tool in Tool::ALL {
        let listed = tools.iter().find(|t| t["name"] == tool.name()).unwrap();
        assert_eq!(listed["annotations"]["readOnlyHint"], !tool.writes());
        assert_eq!(listed["annotations"]["openWorldHint"], tool.open_world());
        assert_eq!(listed["annotations"]["idempotentHint"], tool.idempotent());
        for command in operations(tool) {
            assert_eq!(command.writes, tool.writes(), "{}", command.name);
        }
    }
    for (name, read_only) in [
        ("polylogue_read", true),
        ("polylogue_context", true),
        ("polylogue_note", false),
        ("polylogue_chat", false),
        ("polylogue_act", false),
    ] {
        let tool = tools.iter().find(|t| t["name"] == name).unwrap();
        assert_eq!(tool["annotations"]["readOnlyHint"], read_only, "{name}");
    }
}

/// law: mcp-covers-command-table
#[test]
fn every_tool_operation_is_reachable_from_exactly_one_tool() {
    let tools = tools();
    for command in tool_commands() {
        let offering: Vec<&str> = tools
            .iter()
            .filter(|tool| {
                tool["inputSchema"]["properties"]["operation"]["enum"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(command.name))
            })
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(offering, [command.tool.unwrap().name()], "{}", command.name);
    }
    // Every daemon request the command line or HTTP can make, apart from
    // query forms and the protocol's own completion, is a tool operation.
    let query_forms = ["list", "live", "notices", "sql", "changes"];
    for command in COMMANDS
        .iter()
        .filter(|c| matches!(c.action, Action::Request(_) | Action::Stream { .. }) && c.http)
    {
        assert!(
            command.tool.is_some()
                || query_forms.contains(&command.name)
                || command.name == "complete",
            "{} is not reachable over MCP",
            command.name
        );
    }
}

/// law: mcp-arguments-are-listed-or-explained
#[test]
fn every_argument_is_in_its_tool_schema_or_unlisted() {
    let tools = tools();
    for command in tool_commands() {
        let tool = tools
            .iter()
            .find(|tool| tool["name"] == command.tool.unwrap().name())
            .unwrap();
        let properties = &tool["inputSchema"]["properties"]["arguments"]["properties"];
        for arg in command.args_for(Surface::Mcp) {
            if matches!(arg.name, "request_id" | "cursor") {
                assert!(properties.get(arg.name).is_none());
                assert!(tool["inputSchema"]["properties"].get(arg.name).is_some());
                continue;
            }
            assert_eq!(
                properties.get(arg.name).is_some(),
                !arg.unlisted
                    || operations(command.tool.unwrap()).any(|other| {
                        other
                            .args_for(Surface::Mcp)
                            .any(|a| a.name == arg.name && !a.unlisted)
                    }),
                "{} {}",
                command.name,
                arg.name
            );
        }
    }
}

/// law: mcp-arguments-are-listed-or-explained
#[test]
fn typed_operations_declare_every_field_of_their_arguments() {
    fn fields(schema: &Value, root: &Value, names: &mut std::collections::BTreeSet<String>) {
        if let Some(reference) = schema["$ref"].as_str() {
            let name = reference.strip_prefix("#/$defs/").unwrap();
            return fields(&root["$defs"][name], root, names);
        }
        names.extend(
            schema["properties"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, _)| k.clone()),
        );
        for key in ["anyOf", "oneOf"] {
            for choice in schema[key].as_array().into_iter().flatten() {
                fields(choice, root, names);
            }
        }
    }
    for command in tool_commands().filter(|c| c.register.is_some()) {
        let schema = command.typed_schemas.unwrap()().0;
        let mut typed = std::collections::BTreeSet::new();
        fields(&schema, &schema, &mut typed);
        let declared: std::collections::BTreeSet<String> = command
            .args_for(Surface::Mcp)
            .map(|arg| arg.name.to_owned())
            .filter(|name| !matches!(name.as_str(), "request_id" | "cursor"))
            .collect();
        assert_eq!(typed, declared, "{}", command.name);
    }
}

/// law: mcp-operation-dispatch
#[test]
fn a_tool_call_names_one_of_its_operations() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let run = |name: &str, arguments: Value| {
        call_with_limit(
            &archive,
            name,
            arguments.as_object().unwrap().clone(),
            None,
            &tokio_util::sync::CancellationToken::new(),
            None,
        )
        .map_or_else(
            |error| error,
            |(text, answer, failed)| {
                assert!(failed, "{answer}");
                assert_eq!(answer["error"]["kind"], "bad_request", "{answer}");
                text
            },
        )
    };
    let missing = run(
        "polylogue_read",
        json!({"arguments": {}, "request_id": "no-operation"}),
    );
    assert!(missing.contains("needs `operation`"), "{missing}");
    let foreign = run(
        "polylogue_read",
        json!({"operation": "assert", "arguments": {}, "request_id": "foreign"}),
    );
    assert!(foreign.contains("no operation `assert`"), "{foreign}");
    let inside = run(
        "polylogue_note",
        json!({"operation": "save", "arguments": {"name": "x", "query": "sessions", "request_id": "inner"}, "request_id": "outer"}),
    );
    assert!(inside.contains("beside arguments"), "{inside}");
    let unknown = run(
        "polylogue_context",
        json!({"operation": "brief", "arguments": {"cwd": "/synthetic", "detail": "full"}, "request_id": "unknown-argument"}),
    );
    assert!(unknown.contains("unknown field `detail`"), "{unknown}");
    let live = crate::test_daemon::start("finite context deadline");
    let rows = call(&live.daemon.archive, "list", args(json!({}))).unwrap();
    let rows: Value = serde_json::from_str(&rows).unwrap();
    assert_eq!(rows["count"], 1, "{rows}");
    let (_, answer, failed) = call_with_limit(
        &live.daemon.archive, "polylogue_context",
        json!({"operation":"brief", "arguments":{"cwd":"/synthetic"}, "request_id":"finite", "deadline":1000}).as_object().unwrap().clone(),
        None, &tokio_util::sync::CancellationToken::new(), None,
    ).unwrap();
    assert!(!failed, "{answer}");
    let (_, lifted, failed) = call_with_limit(
        &live.daemon.archive, "polylogue_context",
        json!({"operation":"brief", "arguments":{"cwd":"/synthetic"}, "request_id":"lifted", "deadline":null, "budget_tokens":300}).as_object().unwrap().clone(),
        None, &tokio_util::sync::CancellationToken::new(), None,
    ).unwrap();
    assert!(!failed, "{lifted}");
    let value = &lifted["value"];
    assert!(value["prepared"]["text"].as_str().unwrap().len() <= 1200);
}

mod cancellation;

#[test]
fn tool_requests_and_answers_keep_terminal_controls_visible() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    let hostile = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}";
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let escaped = polylogue_render::terminal::escape(hostile);
    let error = call(&archive, hostile, args(json!({}))).unwrap_err();
    assert!(error.contains(&escaped), "{error}");
    for response in [
        serde_json::to_value(command_surface::Answer::ok(json!({"value":hostile}))).unwrap(),
        serde_json::to_value(command_surface::Answer::<Value>::error(
            command_surface::Failure::new(command_surface::FailureKind::Cancelled, hostile)
                .with_suggestion(command_surface::Suggestion::Query(hostile.into())),
        ))
        .unwrap(),
    ] {
        let listener = UnixListener::bind(archive.socket()).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(&stream).read_line(&mut request).unwrap();
            writeln!(stream, "{response}").unwrap();
        });
        let answer = call(&archive, "status", args(json!({})));
        let text = answer.unwrap_or_else(|error| error);
        assert!(
            !text
                .chars()
                .any(|c| (c.is_control() && c != '\n' && c != '\t') || c == '\u{202e}'),
            "{text}"
        );
        assert!(text.contains("\\x1b") || text.contains("\\u001b"), "{text}");
        server.join().unwrap();
        std::fs::remove_file(archive.socket()).unwrap();
    }
}

#[test]
fn generated_ready_mcp_call_executes_its_exact_typed_envelope() {
    let daemon = test_daemon::start("MCP ready witness");
    let ready = daemon
        .daemon
        .ready_call(&command_surface::Request {
            operation: "query".into(),
            arguments: json!({"query":"sessions"}),
            request_id: "mcp-ready-id".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .unwrap();
    let mut session = Session {
        initialized: true,
        ..Default::default()
    };
    let reply = session
        .prepare(json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":ready.mcp}))
        .answer(&daemon.daemon.archive)
        .unwrap();
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    assert_eq!(
        reply["result"]["structuredContent"]["value"]["count"], 1,
        "{reply}"
    );
    assert_eq!(ready.mcp["name"], "polylogue_read");
    assert_eq!(ready.mcp["arguments"]["operation"], "query");
    // Every declared request retains an executable MCP form.
    let sql = daemon
        .daemon
        .ready_call(&command_surface::Request {
            operation: "sql".into(),
            arguments: json!({"statement":"SELECT 1"}),
            request_id: "mcp-ready-sql".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .unwrap();
    assert_eq!(sql.mcp["name"], "polylogue_read");
    assert_eq!(sql.mcp["arguments"]["operation"], "sql");
    let reply = session
        .prepare(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":sql.mcp}))
        .answer(&daemon.daemon.archive)
        .unwrap();
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    assert_eq!(sql.http.path, "/api/v1/sql");
    let legacy = call_with_limit(
        &daemon.daemon.archive,
        "polylogue_read",
        args(json!({"operation":"query","query":"sessions","request_id":"outside"})),
        None,
        &Default::default(),
        None,
    );
    assert!(
        legacy.is_err(),
        "typed MCP operation accepted arguments outside its declared envelope"
    );
    let tool = tools()
        .into_iter()
        .find(|tool| tool["name"] == "polylogue_read")
        .unwrap();
    assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    assert!(tool["inputSchema"]["properties"].get("arguments").is_some());
}

#[test]
fn typed_admission_failure_has_the_same_shared_answer_without_opening_a_daemon() {
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let (text, structured, failed) = call_with_limit(
        &archive,
        "polylogue_read",
        args(json!({
            "operation":"read",
            "arguments":{"target":"session:claude-code:synthetic", "author":"invented"},
            "request_id":"invalid-admission-witness"
        })),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )
    .unwrap();
    assert!(failed);
    assert!(text.starts_with("error (bad_request)"), "{text}");
    let answer: command_surface::Answer<Value> = serde_json::from_value(structured).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = answer
    else {
        panic!("canonical admission error");
    };
    assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
}
