//! MCP over stdio, as a proxy to the daemon. The commands the table offers
//! over MCP are grouped into five tools; a call names the operation (the
//! command) and carries its arguments, which are checked against the table
//! and sent as the same request the CLI sends. Answers are the daemon's
//! JSON; context (briefs, takeovers, recalls) is answered with its text and
//! recorded as a delivery from `mcp`. Refusals are tool errors.

mod completion;

use crate::commands::{self, Action, COMMANDS, Caller, Command, Surface, Tool, Values};
use crate::daemon::Request;
use crate::protocol::operations::ToolCall;
use command_surface::{Answer, Deadline};
use polylogue_store::Archive;
use serde_json::{Map, Value, json};
use std::io::{self, BufRead, Write};

/// The commands offered as tool operations.
pub fn tool_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS.iter().filter(|c| {
        c.tool.is_some() && matches!(c.action, Action::Request(_) | Action::Stream { .. })
    })
}

/// The operations of one tool.
pub fn operations(tool: Tool) -> impl Iterator<Item = &'static Command> {
    tool_commands().filter(move |c| c.tool == Some(tool))
}

// The tool listing has the shape of command-surface's tool families, with
// the envelope fields Polylogue's requests carry (`request_id`, `deadline`);
// it is replaced by `OperationTable::mcp_tools` when Polylogue adopts them.
pub fn tools() -> Vec<Value> {
    Tool::ALL
        .into_iter()
        .map(|tool| {
            json!({
                "name": tool.name(),
                "description": tool.description(),
                "inputSchema": input_schema(tool),
                "annotations": {
                    "readOnlyHint": !tool.writes(),
                    "destructiveHint": false,
                    "idempotentHint": tool.idempotent(),
                    "openWorldHint": tool.open_world(),
                },
            })
        })
        .collect()
}

/// An argument's distinct schemas, each with the operations that use it.
type ArgumentTypes<'a> = Vec<(Value, Vec<&'a str>)>;

/// A tool's input: `operation` naming the command (required when the tool
/// has several), `arguments` holding the listed arguments of its operations
/// merged by name, and the request's identity and deadline. An argument two
/// operations type differently is a choice of its types, naming the
/// operations that use each. Each operation checks its own arguments,
/// unlisted ones included, when it is called.
fn input_schema(tool: Tool) -> Value {
    let rows: Vec<&Command> = operations(tool).collect();
    let mut fields: Vec<(&str, ArgumentTypes<'_>)> = Vec::new();
    for row in &rows {
        for arg in row
            .args_for(Surface::Mcp)
            .filter(|arg| !arg.unlisted && !matches!(arg.name, "request_id" | "cursor"))
        {
            // A list is listed by its bare type, as command-surface lists it.
            let schema = match arg.kind {
                commands::Kind::List => json!({"type": "array"}),
                kind => commands::kind_schema(kind),
            };
            let index = match fields.iter().position(|(name, _)| *name == arg.name) {
                Some(index) => index,
                None => {
                    fields.push((arg.name, Vec::new()));
                    fields.len() - 1
                }
            };
            let kinds = &mut fields[index].1;
            match kinds.iter_mut().find(|(kind, _)| *kind == schema) {
                Some((_, users)) => users.push(row.name),
                None => kinds.push((schema, vec![row.name])),
            }
        }
    }
    let arguments: Map<String, Value> = fields
        .into_iter()
        .map(|(name, mut kinds)| {
            let schema = if kinds.len() == 1 {
                kinds.pop().expect("one kind").0
            } else {
                let used_by = kinds
                    .iter()
                    .map(|(_, users)| users.join(", "))
                    .collect::<Vec<_>>()
                    .join("; ");
                json!({
                    "anyOf": kinds.into_iter().map(|(kind, _)| kind).collect::<Vec<_>>(),
                    "description": format!("typed per operation: {used_by}"),
                })
            };
            (name.to_owned(), schema)
        })
        .collect();
    let mut required = vec![json!("arguments"), json!("request_id")];
    if rows.len() > 1 {
        required.insert(0, json!("operation"));
    }
    json!({
        "type": "object",
        "properties": {
            "operation": {"enum": rows.iter().map(|row| row.name).collect::<Vec<_>>()},
            "arguments": {"type": "object", "properties": arguments},
            "request_id": {"type": "string"},
            "deadline": {"type": ["integer", "null"], "description": "ms; null: none"},
            "cursor": {"type": ["string", "null"]},
            "budget_tokens": {"type":"integer", "minimum":0},
        },
        "required": required,
        "additionalProperties": false,
    })
}

/// An error as tool text, with the corrected query when the daemon formed
/// one, so an agent can retry with it as given.
/// One text form for a failure: its code, message and runnable correction.
fn error_text(failure: &command_surface::Failure) -> String {
    let code = failure.code.clone().unwrap_or_else(|| {
        serde_json::to_value(failure.kind)
            .ok()
            .and_then(|kind| kind.as_str().map(str::to_owned))
            .unwrap_or_default()
    });
    let mut text = format!("error ({code}): {}", failure.message);
    if let Some(suggestion) = crate::protocol::suggestion_text(failure).as_deref() {
        text.push_str(&format!("\ntry: {suggestion}"));
    }
    for candidate in failure.candidates() {
        text.push_str(&format!("\n  {candidate}"));
    }
    polylogue_render::terminal::escape(&text)
}

fn bad_request(message: &str) -> String {
    polylogue_render::terminal::escape(&format!("error (bad_request): {message}"))
}

/// The request an untyped command builds from a call's arguments. Its
/// retry identity is the call's `request_id`.
fn build_request(command: &Command, call: &ToolCall) -> Result<Request, String> {
    let build = match command.action {
        Action::Request(build) | Action::Stream { build, .. } => build,
        Action::Local(_) | Action::ConfiguredLocal(_) => {
            return Err(bad_request(&format!(
                "`{}` is not a tool operation",
                command.name
            )));
        }
    };
    let Value::Object(mut arguments) = call.arguments.clone() else {
        return Err(bad_request("arguments must be an object"));
    };
    if arguments.contains_key("cursor") {
        return Err(bad_request("cursor goes beside arguments, not inside them"));
    }
    if arguments.contains_key("request_id") {
        return Err(bad_request(
            "request_id goes beside arguments, not inside them",
        ));
    }
    if command
        .args_for(Surface::Mcp)
        .any(|arg| arg.name == "request_id")
    {
        arguments.insert("request_id".into(), Value::String(call.request_id.clone()));
    }
    let values =
        Values::from_json(command, Surface::Mcp, arguments).map_err(|e| bad_request(&e))?;
    let caller = Caller {
        surface: Surface::Mcp,
        cwd: std::env::current_dir().ok(),
        json: true,
    };
    crate::protocol::operations::with_cursor(
        build(command, &values, &caller).map_err(|e| bad_request(&e))?,
        call.cursor.as_deref(),
    )
    .map_err(|error| bad_request(&error.message))
}

/// The text content beside the structured answer. A value stays parseable
/// JSON; its notes are in the structured answer. A failure's text names its
/// notes too.
fn format_answer(
    request: &Request,
    answer: Result<&Answer<Value>, &std::io::Error>,
) -> Result<String, String> {
    let answer = answer.map_err(|e| {
        polylogue_render::terminal::escape(&format!(
            "error (daemon_required): no Polylogue daemon answered ({e}); start it with `polylogue daemon`"
        ))
    })?;
    let notes = crate::protocol::notes(answer)
        .map(|note| format!("\nnote: {}", polylogue_render::terminal::escape(note)))
        .collect::<String>();
    match answer {
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(error),
            ..
        } => Err(format!("{}{notes}", error_text(error))),
        answer => {
            let value = answer.value().expect("a non-error answer has a value");
            Ok(if matches!(request, Request::Context { .. }) {
                polylogue_render::terminal::escape(
                    value["prepared"]["text"].as_str().unwrap_or_default(),
                )
            } else {
                polylogue_render::terminal::json(value, true)
            })
        }
    }
}

/// A `tools/call`: the tool's operation named by the call, run with its
/// arguments.
fn call_with_limit(
    archive: &Archive,
    name: &str,
    arguments: Map<String, Value>,
    deadline_ms: Option<i64>,
    cancelled: &tokio_util::sync::CancellationToken,
    tool_call_id: Option<String>,
) -> Result<(String, Value, bool), String> {
    let tool = Tool::named(name).ok_or_else(|| bad_request(&format!("no tool {name}")))?;
    let call: ToolCall = serde_json::from_value(Value::Object(arguments))
        .map_err(|error| bad_request(&format!("{name}: {error}")))?;
    let names = || {
        operations(tool)
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let command = match call.operation.as_deref() {
        Some(operation) => operations(tool)
            .find(|c| c.name == operation)
            .ok_or_else(|| {
                bad_request(&format!(
                    "{name} has no operation `{operation}`; it has: {}",
                    names()
                ))
            })?,
        None => {
            let mut rows = operations(tool);
            match (rows.next(), rows.next()) {
                (Some(only), None) => only,
                _ => {
                    return Err(bad_request(&format!(
                        "{name} needs `operation`: one of {}",
                        names()
                    )));
                }
            }
        }
    };
    run(archive, command, call, deadline_ms, cancelled, tool_call_id)
}

/// Runs one operation: a typed command as its canonical request, any other
/// as the request its arguments build.
fn run(
    archive: &Archive,
    command: &Command,
    call: ToolCall,
    deadline_ms: Option<i64>,
    cancelled: &tokio_util::sync::CancellationToken,
    tool_call_id: Option<String>,
) -> Result<(String, Value, bool), String> {
    if call.request_id.is_empty() {
        return Err(bad_request("request_id must not be empty"));
    }
    let canonical = command
        .register
        .is_some()
        .then(|| command_surface::Request {
            operation: command.name.into(),
            arguments: call.arguments.clone(),
            request_id: call.request_id.clone(),
            deadline: call.deadline,
            budget_tokens: call.budget_tokens,
            cursor: call.cursor.clone(),
            fresh: false,
            format: Default::default(),
        });
    if canonical.is_none() && call.budget_tokens.is_some() {
        return Err(bad_request("operation does not prepare budgeted context"));
    }
    let request = if let Some(canonical) = &canonical {
        let caller = Caller {
            surface: Surface::Mcp,
            cwd: None,
            json: true,
        };
        match crate::protocol::operations::admit(command, canonical, &caller) {
            Ok(request) => request,
            Err(failure) => {
                let text = error_text(&failure);
                let answer = command_surface::Answer::<Value>::error(failure);
                return Ok((
                    text,
                    serde_json::to_value(answer).expect("answer serializes"),
                    true,
                ));
            }
        }
    } else {
        build_request(command, &call)?
    };
    // An untyped request carries an absolute deadline; a lifted one is
    // expressed only by the typed envelope.
    let parent_deadline_ms = deadline_ms;
    let deadline_ms = match call.deadline {
        Deadline::Default => deadline_ms,
        Deadline::Milliseconds(ms) => Some(crate::daemon::execution::deadline(
            std::time::Duration::from_millis(ms),
        )),
        Deadline::None if canonical.is_some() => deadline_ms,
        Deadline::None => {
            return Err(bad_request(&format!(
                "`{}` takes its deadline in milliseconds; it cannot be lifted",
                command.name
            )));
        }
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let mut caller = crate::protocol::caller::evidence();
    if tool_call_id.is_some() {
        caller.tool_call_id = tool_call_id;
    }
    let answer = if let Some(canonical) = canonical {
        runtime.block_on(crate::client::call_ready_as_bounded(
            archive,
            &canonical,
            cancelled,
            caller,
            parent_deadline_ms,
        ))
    } else {
        runtime.block_on(crate::client::send_ready_as(
            archive,
            &request,
            deadline_ms,
            cancelled,
            caller,
        ))
    };
    let structured = match &answer {
        Ok(answer) => serde_json::to_value(answer).expect("answer serializes"),
        Err(error) => serde_json::to_value(command_surface::Answer::<Value>::error(
            crate::protocol::failure("not_running", error.to_string()),
        ))
        .expect("failure serializes"),
    };
    let failed = matches!(
        answer,
        Ok(command_surface::Answer {
            outcome: command_surface::Outcome::Error(_),
            ..
        }) | Err(_)
    );
    let text = match format_answer(&request, answer.as_ref()) {
        Ok(text) | Err(text) => text,
    };
    Ok((text, structured, failed))
}

const VERSIONS: &[&str] = &[
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
];

#[derive(Default)]
struct Session {
    initialized: bool,
    request_metadata: bool,
}

enum Reply {
    Message(Value),
    Completion {
        id: Value,
        arguments: crate::protocol::complete::Arguments,
        deadline_ms: Option<i64>,
        with_result_type: bool,
    },
    Tool {
        id: Value,
        name: String,
        arguments: serde_json::Map<String, Value>,
        with_result_type: bool,
        deadline_ms: Option<i64>,
        tool_call_id: Option<String>,
    },
    None,
}

fn success(id: Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

impl Session {
    fn prepare(&mut self, request: Value) -> Reply {
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        let valid_id = id.is_null() || id.is_string() || id.is_number();
        let id = if valid_id { id } else { Value::Null };
        let invalid = |code, message| Reply::Message(error(id.clone(), code, message));
        let Some(method) = request["method"].as_str() else {
            // Client responses to server requests have no method.
            return if request.get("result").is_some() || request.get("error").is_some() {
                Reply::None
            } else {
                invalid(-32600, "Invalid request")
            };
        };
        if request["jsonrpc"] != "2.0" || !valid_id {
            return invalid(-32600, "Invalid request");
        }
        if request.get("id").is_none() {
            return Reply::None;
        }
        let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
        if !params.is_object() {
            return invalid(-32602, "Invalid params");
        }
        if method == "initialize" {
            let Some(version) = params["protocolVersion"].as_str() else {
                return invalid(-32602, "protocolVersion is required");
            };
            if !params["capabilities"].is_object()
                || !params["clientInfo"]["name"].is_string()
                || !params["clientInfo"]["version"].is_string()
            {
                return invalid(-32602, "clientInfo and capabilities are required");
            }
            let version = if VERSIONS[..4].contains(&version) {
                version
            } else {
                "2025-11-25"
            };
            self.initialized = true;
            return Reply::Message(success(
                id,
                json!({
                    "protocolVersion": version, "capabilities": {"tools": {}, "completions": {}},
                    "serverInfo": {"name": "polylogue", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": crate::manual::card(),
                }),
            ));
        }
        let meta = &params["_meta"];
        let version = meta["io.modelcontextprotocol/protocolVersion"].as_str();
        if method == "ping"
            && !self.request_metadata
            && (!self.initialized || version.is_none_or(|version| version < "2026-07-28"))
        {
            return Reply::Message(success(id, json!({})));
        }
        let with_result_type = self.request_metadata || version.is_some_and(|v| v >= "2026-07-28");
        if !self.initialized || with_result_type || method == "server/discover" {
            let mut missing = Vec::new();
            if version.is_none() {
                missing.push("io.modelcontextprotocol/protocolVersion");
            }
            if !meta["io.modelcontextprotocol/clientCapabilities"].is_object() {
                missing.push("io.modelcontextprotocol/clientCapabilities");
            }
            if !missing.is_empty() {
                return invalid(
                    -32602,
                    &format!(
                        "request _meta is missing or has malformed required fields: {}",
                        missing.join(", ")
                    ),
                );
            }
            if !VERSIONS.contains(&version.unwrap()) {
                let mut reply = error(id, -32022, "Unsupported protocol version");
                reply["error"]["data"] = json!({"requested":version, "supported":VERSIONS});
                return Reply::Message(reply);
            }
            if method != "server/discover" && !self.initialized {
                self.request_metadata = true;
            }
        }
        if method == "server/discover" {
            return Reply::Message(success(
                id,
                json!({
                    "resultType": "complete", "supportedVersions": VERSIONS,
                    "capabilities": {"tools": {}, "completions": {}}, "instructions": crate::manual::card(),
                    "ttlMs": 0, "cacheScope": "private",
                    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "polylogue", "version": env!("CARGO_PKG_VERSION")}},
                }),
            ));
        }
        let with_result_type = self.request_metadata || with_result_type;
        match method {
            "completion/complete" => match completion::arguments(&params) {
                Ok(arguments) => Reply::Completion {
                    id,
                    arguments,
                    deadline_ms: meta["deadline_ms"].as_i64(),
                    with_result_type,
                },
                Err(message) => invalid(-32602, message),
            },
            "tools/list" => {
                let mut result = json!({"tools": tools()});
                if with_result_type {
                    result["resultType"] = json!("complete");
                }
                Reply::Message(success(id, result))
            }
            "tools/call" => {
                let Some(name) = params["name"].as_str() else {
                    return invalid(-32602, "tool name is required");
                };
                let arguments = match params.get("arguments") {
                    None | Some(Value::Null) => serde_json::Map::new(),
                    Some(Value::Object(arguments)) => arguments.clone(),
                    _ => return invalid(-32602, "tool arguments must be an object"),
                };
                Reply::Tool {
                    id,
                    name: name.to_owned(),
                    arguments,
                    with_result_type,
                    deadline_ms: meta["deadline_ms"].as_i64(),
                    tool_call_id: meta["claudecode/toolUseId"].as_str().map(str::to_owned),
                }
            }
            _ => invalid(-32601, "Method not found"),
        }
    }
}

impl Reply {
    fn answer(self, archive: &Archive) -> Option<Value> {
        self.answer_with_limit(archive, &tokio_util::sync::CancellationToken::new())
    }

    fn answer_with_limit(
        self,
        archive: &Archive,
        cancelled: &tokio_util::sync::CancellationToken,
    ) -> Option<Value> {
        match self {
            Self::None => None,
            Self::Message(message) => Some(message),
            Self::Completion {
                id,
                arguments,
                deadline_ms,
                with_result_type,
            } => Some(completion::answer(
                id,
                archive,
                arguments,
                deadline_ms,
                cancelled,
                with_result_type,
            )),
            Self::Tool {
                id,
                name,
                arguments,
                with_result_type,
                deadline_ms,
                tool_call_id,
            } => {
                let (text, structured, failed) = match call_with_limit(
                    archive,
                    &name,
                    arguments,
                    deadline_ms,
                    cancelled,
                    tool_call_id,
                ) {
                    Ok(answer) => answer,
                    Err(text) => {
                        let answer = command_surface::Answer::<Value>::error(
                            crate::protocol::failure("bad_request", text.clone()),
                        );
                        (
                            text,
                            serde_json::to_value(answer).expect("failure serializes"),
                            true,
                        )
                    }
                };
                let mut result = json!({"content": [{"type": "text", "text": text}], "structuredContent": structured, "isError": failed});
                if with_result_type {
                    result["resultType"] = json!("complete");
                }
                Some(success(id, result))
            }
        }
    }
}

type Worker<'scope> = (
    Value,
    tokio_util::sync::CancellationToken,
    std::thread::ScopedJoinHandle<'scope, io::Result<()>>,
);

fn serve(input: impl BufRead, output: impl Write + Send, archive: &Archive) -> io::Result<()> {
    let output = std::sync::Mutex::new(output);
    let write = |message: Value| -> io::Result<()> {
        let mut output = output
            .lock()
            .map_err(|_| io::Error::other("MCP output lock failed"))?;
        serde_json::to_writer(&mut *output, &message)?;
        output.write_all(b"\n")?;
        output.flush()
    };
    std::thread::scope(|scope| {
        let mut session = Session::default();
        let mut workers: Vec<Worker<'_>> = Vec::new();
        let result = (|| {
            for line in input.lines() {
                let mut index = 0;
                while index < workers.len() {
                    if workers[index].2.is_finished() {
                        workers
                            .swap_remove(index)
                            .2
                            .join()
                            .map_err(|_| io::Error::other("MCP tool thread failed"))??;
                    } else {
                        index += 1;
                    }
                }
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let reply = match serde_json::from_str::<Value>(&line) {
                    Ok(request) => {
                        if request["method"] == "notifications/cancelled"
                            && request.get("id").is_none()
                        {
                            for (id, cancelled, _) in &workers {
                                if id == &request["params"]["requestId"] {
                                    cancelled.cancel();
                                }
                            }
                            continue;
                        }
                        session.prepare(request)
                    }
                    Err(_) => Reply::Message(error(Value::Null, -32700, "Parse error")),
                };
                if let Reply::Tool { ref id, .. } | Reply::Completion { ref id, .. } = reply {
                    let id = id.clone();
                    let cancelled = tokio_util::sync::CancellationToken::new();
                    let worker_cancelled = cancelled.clone();
                    let write = &write;
                    let worker = scope.spawn(move || {
                        if let Some(message) = reply.answer_with_limit(archive, &worker_cancelled) {
                            write(message)?;
                        }
                        Ok(())
                    });
                    workers.push((id, cancelled, worker));
                } else if let Some(message) = reply.answer(archive) {
                    write(message)?;
                }
            }
            Ok(())
        })();
        for (_, cancelled, _) in &workers {
            cancelled.cancel();
        }
        for (_, _, worker) in workers {
            worker
                .join()
                .map_err(|_| io::Error::other("MCP tool thread failed"))??;
        }
        result
    })
}

pub fn serve_stdio(archive: &Archive) -> io::Result<()> {
    serve(io::stdin().lock(), io::stdout(), archive)
}

#[cfg(test)]
pub(crate) mod tests;
