//! MCP tool calls against a real daemon on a temporary archive.

use super::*;
use crate::test_daemon;
use serde_json::json;

/// The `tools/call` parameters that run `operation` (a command name) with
/// these arguments through its tool.
pub(crate) fn tool_call(operation: &str, arguments: Value, request_id: &str) -> Value {
    let command = crate::commands::command(operation).expect("a declared command");
    let home = command.tool.expect("an MCP operation");
    let access = crate::daemon::handler::policy(
        command,
        &arguments,
        polylogue_model::authorship::Channel::Mcp,
    )
    .map(|policy| policy.access)
    // The daemon still owns the refusal of malformed arguments.
    .unwrap_or_else(|_| home.access());
    let tool = Tool::ALL
        .into_iter()
        .find(|tool| tool.access() == access)
        .unwrap();
    let request = crate::client::request(operation, arguments, request_id.into());
    let mut call = crate::commands::declarations().mcp_call(&request).unwrap();
    call["name"] = json!(tool.name());
    call
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

/// The tool result's complete answer, independent of its human text.
pub(crate) fn structured_call_as(
    archive: &Archive,
    operation: &str,
    arguments: serde_json::Map<String, Value>,
    request_id: &str,
) -> command_surface::Answer<Value> {
    let reply = Peer::initialized(archive).request(json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": tool_call(operation, Value::Object(arguments), request_id),
    }));
    let answer: command_surface::Answer<Value> =
        serde_json::from_value(reply["result"]["structuredContent"].clone())
            .unwrap_or_else(|error| panic!("invalid structured tool answer: {error}: {reply}"));
    let failed_step = operation == "rows"
        && answer.value().is_some_and(|value| {
            let line: crate::protocol::line::LineAnswer =
                serde_json::from_value(value.clone()).expect("typed line answer");
            matches!(line, crate::protocol::line::LineAnswer::Steps(steps)
                if steps.steps.iter().any(|step| matches!(step.outcome,
                    crate::protocol::line::StepOutcome::Failed { .. })))
        });
    assert_eq!(
        reply["result"]["isError"],
        answer.failure().is_some() || failed_step,
        "{reply}",
    );
    answer
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

/// Runs one tool call on its own runtime, as the server's handler runs it.
pub(crate) fn call_with_limit(
    archive: &Archive,
    name: &str,
    arguments: serde_json::Map<String, Value>,
    deadline_ms: Option<i64>,
    cancelled: &CancellationToken,
    tool_call_id: Option<String>,
) -> Result<(String, Value, bool), String> {
    super::runtime().unwrap().block_on(answer_call(
        archive,
        name,
        arguments,
        deadline_ms,
        cancelled,
        tool_call_id,
    ))
}

/// Serves MCP on one end of a socket pair, as `polylogue mcp` serves stdio.
pub(crate) fn serve_socket(
    socket: std::os::unix::net::UnixStream,
    archive: &Archive,
) -> std::io::Result<()> {
    super::runtime()?.block_on(async {
        socket.set_nonblocking(true)?;
        let (input, output) = tokio::net::UnixStream::from_std(socket)?.into_split();
        serve(input, output, archive.clone()).await
    })
}

/// An MCP client session with the real server over a socket pair.
pub(crate) struct Peer {
    client: std::os::unix::net::UnixStream,
    input: std::io::BufReader<std::os::unix::net::UnixStream>,
    server: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}

impl Peer {
    pub(crate) fn start(archive: &Archive) -> Self {
        let (client, socket) = std::os::unix::net::UnixStream::pair().unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(60)))
            .unwrap();
        let archive = archive.clone();
        let server = std::thread::spawn(move || serve_socket(socket, &archive));
        Self {
            input: std::io::BufReader::new(client.try_clone().unwrap()),
            client,
            server: Some(server),
        }
    }

    /// A session that has completed `initialize`.
    pub(crate) fn initialized(archive: &Archive) -> Self {
        let mut peer = Self::start(archive);
        let reply = peer.request(initialize());
        assert!(reply["result"].is_object(), "{reply}");
        peer.send(&json!({"jsonrpc":"2.0", "method":"notifications/initialized"}).to_string());
        peer
    }

    pub(crate) fn send(&mut self, line: &str) {
        use std::io::Write;
        writeln!(self.client, "{line}").unwrap();
    }

    /// Sends a request and returns the reply that carries its id, or the
    /// first reply without one.
    pub(crate) fn request(&mut self, request: Value) -> Value {
        use std::io::BufRead;
        self.send(&request.to_string());
        loop {
            let mut line = String::new();
            assert_ne!(self.input.read_line(&mut line).unwrap(), 0, "MCP closed");
            let reply: Value = serde_json::from_str(&line).unwrap();
            if reply["id"] == request["id"] || reply["id"].is_null() {
                return reply;
            }
        }
    }
}

impl Drop for Peer {
    fn drop(&mut self) {
        let _ = self.client.shutdown(std::net::Shutdown::Write);
        if let Some(server) = self.server.take() {
            let result = server.join();
            if !std::thread::panicking() {
                result.unwrap().unwrap();
            }
        }
    }
}

fn call_params(archive: &Archive, params: Value) -> Result<String, String> {
    call_with_limit(
        archive,
        params["name"].as_str().unwrap(),
        params["arguments"].as_object().unwrap().clone(),
        None,
        &CancellationToken::new(),
        None,
    )
    .and_then(|(text, _, failed)| if failed { Err(text) } else { Ok(text) })
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
        "rows",
        args(json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "format":"markdown"}]})),
    )
    .unwrap();
    assert!(text.starts_with(polylogue_render::ARCHIVE_NOTICE), "{text}");
    assert!(text.contains("walrus"), "{text}");
    assert!(text.contains("Times "), "{text}");
    let structured = structured_call_as(
        &archive,
        "rows",
        args(json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read"}]})),
        &uuid::Uuid::now_v7().to_string(),
    );
    let line: crate::protocol::line::LineAnswer =
        serde_json::from_value(structured.value().unwrap().clone()).unwrap();
    let crate::protocol::line::LineAnswer::Steps(steps) = line else {
        panic!("read steps")
    };
    let crate::protocol::line::StepOutcome::Done {
        reply: crate::protocol::line::StepAnswer::Read(read),
        ..
    } = &steps.steps[0].outcome
    else {
        panic!("finished read")
    };
    assert!(read_pages::body(read).contains("walrus"));

    let found = structured_call_as(
        &archive,
        "rows",
        args(json!({"selection": "walrus"})),
        &uuid::Uuid::now_v7().to_string(),
    );
    assert_eq!(found.value().unwrap()["reply"]["count"], 1, "{found:?}");
    let text = call(&archive, "rows", args(json!({"selection": "walrus"}))).unwrap();
    assert!(text.contains("claude-code:s1"), "{text}");

    let e = call(
        &archive,
        "rows",
        args(json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "pages":1}]})),
    )
    .unwrap_err();
    assert!(e.starts_with("error (bad_request)"), "{e}");
    let e = call(
        &archive,
        "rows",
        args(json!({"selection": "session:claude-code:nope", "steps":[{"verb":"read"}]})),
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
        let worker = scope.spawn(|| serve_socket(server, &archive));
        writeln!(client, "{requests}").unwrap();
        let mut input = BufReader::new(client.try_clone().unwrap());
        // Unparsable input gets no reply: it has no id to answer.
        let replies = (0..7)
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
    assert_eq!(replies.len(), 7);
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
    assert_eq!(reply(json!(5))["error"]["code"], -32601);
    assert_eq!(reply(json!(6))["error"]["code"], -32602);
    assert_eq!(reply(json!(7))["result"], json!({}));
}

#[test]
fn discovery_and_inline_requests_keep_protocol_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let mut session = Peer::start(&archive);
    let discovery = session.request(
        json!({"jsonrpc":"2.0", "id":1, "method":"server/discover", "params":{"_meta":meta}}),
    );
    assert_eq!(
        discovery["result"]["supportedVersions"],
        json!([
            "2024-11-05",
            "2025-03-26",
            "2025-06-18",
            "2025-11-25",
            "2026-07-28"
        ])
    );
    assert_eq!(discovery["result"]["resultType"], "complete");
    let list = session
        .request(json!({"jsonrpc":"2.0", "id":2, "method":"tools/list", "params":{"_meta":meta}}));
    assert_eq!(list["result"]["tools"], json!(tools()));
    assert_eq!(list["result"]["resultType"], "complete");
    let invalid = session.request(json!({"jsonrpc":"2.0", "id":3, "method":"tools/list"}));
    assert_eq!(invalid["error"]["code"], -32602);
    let mut init = initialize();
    init["params"]["protocolVersion"] = json!("future");
    let answer = Peer::start(&archive).request(init);
    assert_eq!(answer["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn protocol_ids_null_arguments_and_ping_follow_the_request_version() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let mut session = Peer::initialized(&archive);
    let id = json!(i64::MAX);
    let ping = session.request(json!({"jsonrpc":"2.0", "id":id, "method":"ping"}));
    assert_eq!(ping["id"], id);
    assert_eq!(ping["result"], json!({}));
    // Null arguments are an empty argument map: the call reaches the tool,
    // which answers it, rather than failing as a protocol error.
    let call = session.request(json!({"jsonrpc":"2.0", "id":2, "method":"tools/call", "params":{"name":"polylogue_read", "arguments":null}}));
    assert!(call["result"]["content"][0]["text"].is_string(), "{call}");
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28", "io.modelcontextprotocol/clientCapabilities":{}});
    let ping =
        session.request(json!({"jsonrpc":"2.0", "id":3, "method":"ping", "params":{"_meta":meta}}));
    assert_eq!(ping["error"]["code"], -32601);
}

#[test]
fn malformed_request_ids_get_no_reply_and_keep_the_session() {
    let temp = tempfile::tempdir().unwrap();
    let archive = Archive::at(temp.path());
    let mut session = Peer::start(&archive);
    // A reply without the caller's id could not be matched to its request.
    for id in [json!({"wrong":1}), json!([1]), json!(true)] {
        session.send(&json!({"jsonrpc":"2.0", "id":id, "method":"ping"}).to_string());
    }
    let reply = session.request(initialize());
    assert_eq!(reply["id"], "init", "{reply}");
    assert_eq!(reply["result"]["protocolVersion"], "2025-11-25");
}

#[test]
fn server_instructions_are_the_card_naming_every_tool() {
    let temp = tempfile::tempdir().unwrap();
    let info = Peer::start(&Archive::at(temp.path())).request(initialize());
    let text = info["result"]["instructions"].as_str().unwrap();
    assert_eq!(text, crate::manual::card());
    for tool in [Tool::Read, Tool::Write, Tool::Act] {
        assert!(text.contains(tool.name()), "{}", tool.name());
    }
    assert!(text.contains("operation defaults to \"rows\""));
    assert!(text.len() <= 800, "the card is {} bytes", text.len());
}

#[test]
fn manual_query_examples_compile_through_the_public_command() {
    let w = test_daemon::start("manual query examples");
    let archive = Archive::at(&w.daemon.archive.root);
    let manual = crate::manual::text();
    let mut queries: Vec<&str> = manual
        .split('`')
        .filter(|text| text.starts_with("sessions where ") || text.starts_with("messages where "))
        .collect();
    // The guide gives this one inside a call, so the split cannot find it.
    let in_a_call = "sessions where repo:example since:7d";
    assert!(manual.contains(in_a_call));
    queries.push(in_a_call);
    assert!(
        queries.contains(&"messages where \"timeout\""),
        "{queries:?}"
    );
    for query in queries {
        call(
            &archive,
            "rows",
            args(json!({"selection":query, "explain":true})),
        )
        .unwrap_or_else(|e| panic!("{query}: {e}"));
    }
}

#[test]
fn errors_show_archived_controls_as_text() {
    let hostile = "\x1b]0;title\x07\x1b]2;title\x1b\\\x1b]52;c;YQ==\x07\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\x1b[2J\x1b\u{9b}2J\u{202e}";
    let mut failure = crate::protocol::failure(polylogue_model::Code::BadRequest, hostile)
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

#[test]
fn tool_annotations_follow_the_command_table() {
    let listed = tools();
    for tool in Tool::ALL {
        let entry = listed
            .iter()
            .find(|entry| entry["name"] == tool.name())
            .unwrap();
        let expected = match tool.access() {
            ToolAccess::Read => json!({"readOnlyHint":true,"openWorldHint":false}),
            ToolAccess::Write => {
                json!({"readOnlyHint":false,"idempotentHint":true,"openWorldHint":false})
            }
            ToolAccess::Act => {
                json!({"readOnlyHint":false,"destructiveHint":false,"openWorldHint":true})
            }
        };
        assert_eq!(entry["annotations"], expected, "{}", tool.name());
    }
}

#[test]
fn larger_tools_offer_smaller_operations() {
    let listed = tools();
    for command in tool_commands() {
        let home = command.tool.unwrap();
        for tool in Tool::ALL {
            let entry = listed
                .iter()
                .find(|entry| entry["name"] == tool.name())
                .unwrap();
            let offered = entry["inputSchema"]["properties"]["operation"]["enum"]
                .as_array()
                .unwrap();
            let expected = home == tool
                || (matches!(tool, Tool::Write | Tool::Act) && home.access() <= tool.access())
                || (tool == Tool::Read && command.name == "set");
            assert_eq!(
                offered.contains(&json!(command.name)),
                expected,
                "{} on {}",
                command.name,
                tool.name()
            );
        }
    }
    for command in COMMANDS
        .iter()
        // A row the command line follows as a stream is not a tool call.
        .filter(|command| {
            command.http && command.operation().is_some() && command.deliver.is_none()
        })
    {
        if command.name.starts_with("internal ") {
            assert_eq!(
                command.cli,
                crate::commands::Cli::Hidden,
                "{}",
                command.name
            );
            assert_eq!(command.tool, None, "{}", command.name);
        } else {
            assert!(
                command.tool.is_some(),
                "{} is not reachable over MCP",
                command.name
            );
        }
    }
}

/// A row's request arguments as declared, without its calling-process options.
fn arguments(
    command: &crate::commands::Command,
) -> impl Iterator<Item = &'static command_surface::ArgumentSpec> {
    crate::commands::declarations()
        .spec_named(command.name)
        .expect("declared row")
        .arguments
        .iter()
        .filter(|arg| !arg.calling_process)
}

#[test]
fn every_argument_is_in_its_tool_schema_or_unlisted() {
    let tools = tools();
    for command in tool_commands() {
        let tool = tools
            .iter()
            .find(|tool| tool["name"] == command.tool.unwrap().name())
            .unwrap();
        let properties = &tool["inputSchema"]["properties"]["arguments"]["properties"];
        for arg in arguments(command).filter(|argument| argument.listed) {
            assert!(
                properties.get(&arg.name).is_some(),
                "{} {}",
                command.name,
                arg.name
            );
        }
    }
}

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
            &CancellationToken::new(),
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
    assert!(
        run("polylogue_read", json!({"operation":7,"arguments":{}}))
            .contains("operation must be text")
    );
    let wrong_tool = run(
        "polylogue_read",
        json!({"arguments": {"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"a decision"}]}, "request_id": "wrong-tool"}),
    );
    assert!(
        wrong_tool.contains("rows with these arguments needs polylogue_write"),
        "{wrong_tool}"
    );
    let inside = run(
        "polylogue_write",
        json!({"operation": "rows", "arguments": {"selection": "sessions", "steps":[{"verb":"save", "name":"x"}], "request_id": "inner"}, "request_id": "outer"}),
    );
    assert!(inside.contains("beside arguments"), "{inside}");
    let unknown = run(
        "polylogue_read",
        json!({"operation": "rows", "arguments": {"steps":[{"verb":"brief", "cwd": "/synthetic", "detail": "full"}]}, "request_id": "unknown-argument"}),
    );
    assert!(
        unknown.contains("brief: unknown argument detail"),
        "{unknown}"
    );
    let live = crate::test_daemon::start("finite context deadline");
    let rows = structured_call_as(
        &live.daemon.archive,
        "rows",
        args(json!({"selection":"sessions"})),
        "finite-context-listing",
    );
    assert_eq!(rows.value().unwrap()["reply"]["count"], 1, "{rows:?}");
    let (_, answer, failed) = call_with_limit(
        &live.daemon.archive, "polylogue_read",
        json!({"operation":"rows", "arguments":{"steps":[{"verb":"brief","cwd":"/synthetic"}]}, "request_id":"finite", "deadline":1000}).as_object().unwrap().clone(),
        None, &CancellationToken::new(), None,
    ).unwrap();
    assert!(!failed, "{answer}");
    let manual = format!("\nAgent manual:\n{}\n", crate::manual::text());
    let budget = manual.len().div_ceil(4) + 300;
    let (_, lifted, failed) = call_with_limit(
        &live.daemon.archive, "polylogue_read",
        json!({"operation":"rows", "arguments":{"steps":[{"verb":"brief","cwd":"/synthetic"}]}, "request_id":"lifted", "deadline":null, "budget_tokens":budget}).as_object().unwrap().clone(),
        None, &CancellationToken::new(), None,
    ).unwrap();
    assert!(!failed, "{lifted}");
    let value = &lifted["value"];
    let text = value["reply"]["steps"][0]["outcome"]["reply"]["reply"]["prepared"]["text"]
        .as_str()
        .unwrap();
    assert!(text.contains(&manual));
    assert!(text.len() <= budget * 4);
}

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
            write!(stream, "event: answer\ndata: {response}\n\n").unwrap();
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
fn generated_ready_mcp_call_executes_its_typed_envelope() {
    let daemon = test_daemon::start("MCP ready example");
    let ready = daemon
        .daemon
        .ready_call(&command_surface::Request {
            operation: "rows".into(),
            arguments: json!({"selection":"sessions"}),
            request_id: "mcp-ready-id".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .unwrap();
    let mut session = Peer::initialized(&daemon.daemon.archive);
    let reply = session.request(json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":crate::commands::declarations().mcp_call(&ready.request).unwrap()}));
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    assert_eq!(
        reply["result"]["structuredContent"]["value"]["reply"]["count"], 1,
        "{reply}"
    );
    assert_eq!(
        crate::commands::declarations()
            .mcp_call(&ready.request)
            .unwrap()["name"],
        "polylogue_read"
    );
    assert_eq!(
        crate::commands::declarations()
            .mcp_call(&ready.request)
            .unwrap()["arguments"]["operation"],
        "rows"
    );
    // Shared metadata survives transport. Fresh remains unsupported; JSON is
    // an accepted output format for the current line operation.
    for option in ["fresh", "format"] {
        let mut request = ready.request.clone();
        request.request_id = format!("metadata-{option}");
        if option == "fresh" {
            request.fresh = true;
        } else {
            request.format = command_surface::Format::Json;
        }
        let params = crate::commands::declarations().mcp_call(&request).unwrap();
        assert_eq!(
            params["arguments"][option],
            if option == "fresh" {
                json!(true)
            } else {
                json!("json")
            },
        );
        let reply =
            session.request(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":params}));
        if option == "fresh" {
            assert_eq!(reply["result"]["isError"], true, "{reply}");
            assert_eq!(
                reply["result"]["structuredContent"]["error"]["kind"],
                "bad_request"
            );
            assert_eq!(
                reply["result"]["structuredContent"]["error"]["message"],
                "polylogue does not take fresh"
            );
        } else {
            assert_eq!(reply["result"]["isError"], false, "{reply}");
            let answer =
                serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
            let listing = crate::test_answers::ok(crate::test_answers::listing(answer));
            assert_eq!(listing["count"], 1);
        }
    }
    // Every declared request keeps an executable MCP form.
    let sql = daemon
        .daemon
        .ready_call(&command_surface::Request {
            operation: "dev sql".into(),
            arguments: json!({"statement":"SELECT 1"}),
            request_id: "mcp-ready-sql".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .unwrap();
    assert_eq!(
        crate::commands::declarations()
            .mcp_call(&sql.request)
            .unwrap()["name"],
        "polylogue_read"
    );
    assert_eq!(
        crate::commands::declarations()
            .mcp_call(&sql.request)
            .unwrap()["arguments"]["operation"],
        "dev sql"
    );
    let reply = session.request(json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":crate::commands::declarations().mcp_call(&sql.request).unwrap()}));
    assert_eq!(reply["result"]["isError"], false, "{reply}");
    let mut unsupported = sql.request.clone();
    unsupported.cursor = Some("a SQL request has no continuation".into());
    unsupported.request_id = "sql-with-unsupported-cursor".into();
    let params = crate::commands::declarations()
        .mcp_call(&unsupported)
        .unwrap();
    let refused =
        session.request(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":params}));
    assert_eq!(refused["result"]["isError"], true, "{refused}");
    let answer = serde_json::from_value(refused["result"]["structuredContent"].clone()).unwrap();
    let error = crate::test_answers::refusal(answer);
    assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
    assert_eq!(error.message, "dev sql does not take cursor");
    assert!(
        polylogue_render::ready::http_url(crate::commands::declarations(), &sql.request)
            .unwrap()
            .starts_with("/api/v1/dev/sql?")
    );
    let outside_arguments = call_with_limit(
        &daemon.daemon.archive,
        "polylogue_read",
        args(json!({"operation":"rows","selection":"sessions","request_id":"outside"})),
        None,
        &Default::default(),
        None,
    );
    let refusal = outside_arguments
        .expect_err("typed MCP operation accepted arguments outside its declared envelope");
    assert!(refusal.contains("selection"), "{refusal}");
    let tool = tools()
        .into_iter()
        .find(|tool| tool["name"] == "polylogue_read")
        .unwrap();
    assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    assert!(tool["inputSchema"]["properties"].get("arguments").is_some());
}

#[test]
fn typed_acceptance_failure_answers_without_a_daemon() {
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let params = json!({
        "name":"polylogue_read",
        "arguments":{
            "operation":"rows",
            "arguments":{"selection":"session:claude-code:synthetic", "steps":[{"verb":"read", "author":"invented"}]},
            "request_id":"invalid-request-example"
        }
    });
    let reply = Peer::initialized(&archive)
        .request(json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":params}));
    assert_eq!(reply["result"]["isError"], true, "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.starts_with("error (bad_request)"), "{text}");
    let answer = serde_json::from_value(reply["result"]["structuredContent"].clone()).unwrap();
    let error = crate::test_answers::refusal(answer);
    assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
    assert!(error.message.contains("read: unknown argument author"));
}

/// The tool list is the shared declarations rendered from the rows: the
/// same tools with the declared descriptions; each tool offers exactly the
/// operations of its rows and of the smaller tools' rows (the read tool also
/// listing and showing settings), the steps within its access, and takes
/// each row's typed arguments; `operation` is never required.
#[test]
fn tool_schema_from_operation_table() {
    tools_follow_the_operation_table(&tools());
}

fn tools_follow_the_operation_table(listed: &[Value]) {
    let declarations = crate::commands::declarations();
    let declared = declarations.mcp_tools();
    let names = |tools: &[Value]| -> Vec<String> {
        tools
            .iter()
            .map(|tool| tool["name"].as_str().unwrap().to_owned())
            .collect()
    };
    assert_eq!(names(listed), names(&declared));
    let offered = crate::daemon::verbs::offered_over_mcp();
    for (entry, home) in listed.iter().zip(&declared) {
        assert_eq!(entry["description"], home["description"]);
        let tool = Tool::named(entry["name"].as_str().unwrap()).unwrap();
        let mut rows: Vec<&str> = Tool::ALL
            .into_iter()
            .filter(|smaller| smaller.access() <= tool.access())
            .flat_map(operations)
            .map(|row| row.name)
            .collect();
        if tool == Tool::Read {
            rows.push("set");
        }
        let mut operations: Vec<&str> = entry["inputSchema"]["properties"]["operation"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap())
            .collect();
        rows.sort_unstable();
        operations.sort_unstable();
        assert_eq!(operations, rows, "{}", tool.name());
        assert!(
            !entry["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .contains(&json!("operation")),
            "{}",
            tool.name()
        );
        let verbs: Vec<&str> = offered
            .iter()
            .filter(|(_, access)| *access <= tool.access())
            .map(|(verb, _)| *verb)
            .collect();
        assert_eq!(
            entry["inputSchema"]["properties"]["arguments"]["properties"]["steps"]["items"]["properties"]
                ["verb"]["enum"],
            json!(verbs),
            "{}",
            tool.name()
        );
        let family = declarations.family_schemas(tool.family());
        for row in operations_of(tool) {
            let description = declarations
                .descriptions()
                .iter()
                .find(|description| description.spec.name == row)
                .unwrap();
            assert_eq!(family[row], description.input_schema, "{row}");
        }
    }
}

fn operations_of(tool: Tool) -> Vec<&'static str> {
    operations(tool).map(|row| row.name).collect()
}

/// The descriptions offer only what the build does: every verb signature
/// and operation signature they print names declared fields, and nothing on
/// the list of what is not built appears as an option of its verb.
#[test]
fn tool_descriptions_name_only_built_options() {
    let verbs = crate::daemon::verbs::declarations();
    for tool in tools() {
        let description = tool["description"].as_str().unwrap();
        for line in description.lines() {
            let Some((verb, fields)) = line
                .trim_start()
                .split_once('(')
                .filter(|(name, _)| !name.contains(' ') || name.starts_with("dev "))
            else {
                continue;
            };
            let Some(declared) = verbs
                .verbs()
                .iter()
                .find(|declared| declared.verb() == verb)
            else {
                continue;
            };
            let fields = fields.split_once(')').unwrap().0;
            for field in fields.split(", ").filter(|field| !field.is_empty()) {
                let field = field.trim_end_matches('?');
                assert!(
                    declared.input_schema()["properties"].get(field).is_some(),
                    "{verb}({field}) in {}",
                    tool["name"]
                );
                assert!(
                    crate::unbuilt::option(verb, field).is_none(),
                    "{verb}({field}) is not built"
                );
            }
            assert!(
                line.trim_start().starts_with(&declared.signature()),
                "{line}"
            );
        }
    }
    // Each verb a tool adds is described there, and pick, which needs a
    // person, nowhere.
    let all: String = tools()
        .iter()
        .map(|tool| tool["description"].as_str().unwrap().to_owned())
        .collect();
    for (verb, _) in crate::daemon::verbs::offered_over_mcp() {
        assert!(all.contains(&format!("  {verb}(")), "{verb}");
    }
    assert!(!all.contains("  pick("));
}

/// docs/agents.md, The tools as they ship: the page states the server
/// instructions and each tool's name, annotations, description and input
/// schema; the server lists exactly these.
#[test]
fn the_tools_as_they_ship_equal_the_page() {
    let page = include_str!("../../../../docs/agents.md");
    let section = page.split_once("\n### The tools as they ship\n").unwrap().1;
    let section = &section[..section.find("\n### ").unwrap()];
    let blocks = |text: &str| -> Vec<String> {
        text.split("\n```")
            .skip(1)
            .step_by(2)
            .map(|block| block.split_once('\n').unwrap().1.trim_end().to_owned())
            .collect()
    };
    let headings = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|line| line.starts_with("**`polylogue_"))
            .map(str::to_owned)
            .collect()
    };
    // What the server lists, written as the page writes it.
    let listed = tools();
    let mut served = format!("\n```\n{}\n```\n", crate::manual::card().trim_end());
    for tool in &listed {
        served.push_str(&format!(
            "\n**`{}`** · annotations `{}`\n\n```\n{}\n```\n\n```json\n{}\n```\n",
            tool["name"].as_str().unwrap(),
            tool["annotations"],
            tool["description"].as_str().unwrap(),
            tool["inputSchema"],
        ));
    }
    println!("{served}");
    assert_eq!(headings(section), headings(&served));
    let (on_page, from_server) = (blocks(section), blocks(&served));
    assert_eq!(on_page.len(), from_server.len());
    for (page, server) in on_page.iter().zip(&from_server) {
        match serde_json::from_str::<Value>(page) {
            Ok(schema) => assert_eq!(schema, serde_json::from_str::<Value>(server).unwrap()),
            Err(_) => assert_eq!(page, server),
        }
    }
    let size = serde_json::to_string(&listed).unwrap().len();
    let card = crate::manual::card().len();
    assert!(card <= 800, "the instructions are {card} bytes");
    println!("tool list {size} bytes as compact JSON, instructions {card} bytes");
}

#[test]
fn generated_mcp_follow_up_preserves_the_accepted_request() {
    use std::io::{BufRead, BufReader, Write};
    let directory = tempfile::tempdir().unwrap();
    let archive = Archive::at(directory.path());
    let request = command_surface::RunnableRequest {
        deadline: command_surface::Deadline::None,
        budget_tokens: command_surface::TokenBudget::Unlimited,
        cursor: Some("opaque-follow-up".into()),
        fresh: true,
        format: command_surface::Format::Json,
        ..command_surface::RunnableRequest::new(
            "rows",
            json!({"selection":"session:claude-code:synthetic", "steps":[{"verb":"read"}]}),
            "accepted-child",
        )
    };
    let projected = crate::commands::declarations().mcp_call(&request).unwrap();
    let parent_deadline = i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
    )
    .unwrap()
        + 60_000;
    let listener = std::os::unix::net::UnixListener::bind(archive.socket()).unwrap();
    let peer = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        let mut line = String::new();
        BufReader::new(&stream).read_line(&mut line).unwrap();
        let incoming = crate::protocol::wire::incoming(&line).unwrap();
        assert_eq!(incoming.parent_deadline_ms, Some(parent_deadline));
        assert_eq!(
            incoming.caller.tool_call_id.as_deref(),
            Some("synthetic-mcp-call")
        );
        assert_eq!(incoming.request, request);
        let refusal = command_surface::Answer::<Value>::error(command_surface::Failure::new(
            command_surface::FailureKind::Refused,
            "synthetic refusal",
        ));
        write!(
            stream,
            "event: answer\ndata: {}\n\n",
            serde_json::to_string(&refusal).unwrap()
        )
        .unwrap();
    });
    let (text, _, failed) = call_with_limit(
        &archive,
        projected["name"].as_str().unwrap(),
        projected["arguments"].as_object().unwrap().clone(),
        Some(parent_deadline),
        &CancellationToken::new(),
        Some("synthetic-mcp-call".into()),
    )
    .unwrap();
    assert!(failed, "{text}");
    assert!(text.contains("synthetic refusal"), "{text}");
    peer.join().unwrap();
}

#[test]
fn set_mcp_keeps_typed_write_repeat_and_read_results() {
    let daemon = test_daemon::start("MCP settings example");
    let mut session = Peer::initialized(&daemon.daemon.archive);
    let mut invoke = |id, arguments| {
        session.request(json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":tool_call("set", arguments,"mcp-setting-write-result")}))
    };
    let first = invoke(1, json!({"key":"reading.alternates","value":"[]"}));
    assert_eq!(first["result"]["isError"], false, "{first}");
    let first: crate::protocol::setting::Answer =
        serde_json::from_value(first["result"]["structuredContent"]["value"].clone()).unwrap();
    let crate::protocol::setting::Answer::Outcome(first) = first else {
        panic!("write must return a write result")
    };
    assert!(!first.pending);
    let repeat = invoke(2, json!({"key":"reading.alternates","value":"[]"}));
    let repeat: crate::protocol::setting::Answer =
        serde_json::from_value(repeat["result"]["structuredContent"]["value"].clone()).unwrap();
    let crate::protocol::setting::Answer::Outcome(repeat) = repeat else {
        panic!("repeat must return a write result")
    };
    assert_eq!(first.id, repeat.id);
    assert!(repeat.repeat);
    assert!(!repeat.pending);
    let shown = session.request(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":tool_call("set",json!({"key":"reading.alternates"}),"mcp-setting-current")}));
    assert_eq!(shown["result"]["isError"], false, "{shown}");
    let current: crate::protocol::setting::Answer =
        serde_json::from_value(shown["result"]["structuredContent"]["value"].clone()).unwrap();
    let crate::protocol::setting::Answer::Current(current) = current else {
        panic!("MCP returns setting details");
    };
    assert_eq!(current.current.key, "reading.alternates");
    assert!(current.allowed_values.is_empty());
    assert_eq!(current.history.len(), 1);
    assert_eq!(
        current.history[0].value,
        polylogue_model::settings::Value::Alternates(vec![])
    );
    assert_eq!(
        current.current.authored.as_ref().unwrap().reference,
        current.history[0].authored.reference
    );
}

mod rows;

#[test]
fn status_keeps_verification_out_of_the_normal_mcp_tool_path() {
    let daemon = crate::test_daemon::start("synthetic source");
    let manifest = daemon._dir.path().join("not-created.tsv");
    let arguments = json!({"verify":true, "manifest":manifest})
        .as_object()
        .unwrap()
        .clone();
    let refused = structured_call_as(
        &daemon.daemon.archive,
        "status",
        arguments,
        "status-verify-refused",
    );
    let error = refused.failure().expect("MCP refuses archive verification");
    assert_eq!(error.kind, command_surface::FailureKind::BadRequest);
    assert!(error.message.contains("command line"));
    assert!(!manifest.exists());
    let answer = structured_call_as(
        &daemon.daemon.archive,
        "status",
        json!({"vectors":true}).as_object().unwrap().clone(),
        "status-vectors",
    );
    let status: crate::protocol::status::Answer =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    assert_eq!(
        status.vectors,
        Some(polylogue_store::vectors::VectorStatus::default())
    );
    assert!(status.verification.is_none());
}
