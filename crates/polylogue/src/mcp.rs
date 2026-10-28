//! MCP over stdio, as a proxy to the daemon: one tool per command the
//! table offers over MCP, named by its words joined with `_`. Each call is
//! checked against the table, sent as the same request the CLI sends, and
//! answered with the daemon's JSON; context (briefs, takeovers, recalls) is
//! answered with its text and recorded as a delivery from `mcp`. Refusals
//! are tool errors.

use crate::commands::{self, Action, COMMANDS, Caller, Command, Surface, Values};
use crate::daemon::{Request, Response};
use polylogue_store::Archive;
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

/// The commands offered as tools.
pub fn tool_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS
        .iter()
        .filter(|c| c.mcp && matches!(c.action, Action::Request(_)))
}

pub fn tools() -> Vec<Value> {
    tool_commands()
        .map(|c| json!({"name": c.mcp_name(), "description": c.help, "inputSchema": commands::schema(c), "annotations": {"readOnlyHint": !c.writes}}))
        .collect()
}

/// An error as tool text, with the corrected query when the daemon formed
/// one, so an agent can retry with it as given.
fn error_text(kind: &str, message: &str, detail: Option<&serde_json::Value>) -> String {
    let text = match detail.and_then(|d| d["suggestion"].as_str()) {
        Some(s) => format!("error ({kind}): {message}\ntry: {s}"),
        None => format!("error ({kind}): {message}"),
    };
    polylogue_render::terminal::escape(&text)
}

fn build_request(
    name: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
) -> Result<Request, String> {
    let command = tool_commands()
        .find(|c| c.mcp_name() == name)
        .ok_or_else(|| {
            polylogue_render::terminal::escape(&format!("error (bad_request): no tool {name}"))
        })?;
    let Action::Request(build) = command.action else {
        return Err(polylogue_render::terminal::escape(&format!(
            "error (bad_request): no tool {name}"
        )));
    };
    let bad = |e: String| polylogue_render::terminal::escape(&format!("error (bad_request): {e}"));
    let values = Values::from_json(command, Surface::Mcp, arguments).map_err(bad)?;
    let caller = Caller {
        surface: Surface::Mcp,
        cwd: std::env::current_dir().ok(),
        json: true,
    };
    build(command, &values, &caller).map_err(bad)
}

fn format_answer(request: &Request, response: std::io::Result<Response>) -> Result<String, String> {
    match response {
        Ok(Response::Ok { data, .. }) if matches!(request, Request::Context { .. }) => {
            Ok(polylogue_render::terminal::escape(
                data["prepared"]["text"].as_str().unwrap_or_default(),
            ))
        }
        Ok(Response::Ok { data, .. }) => Ok(polylogue_render::terminal::json(&data, true)),
        Ok(Response::Error {
            kind,
            message,
            detail,
        }) => Err(error_text(&kind, &message, detail.as_ref())),
        Err(e) => Err(polylogue_render::terminal::escape(&format!(
            "error (daemon_required): no Polylogue daemon answered ({e}); start it with `polylogue daemon`"
        ))),
    }
}

fn call_with_limit(
    archive: &Archive,
    name: &str,
    arguments: serde_json::Map<String, Value>,
    deadline_ms: Option<i64>,
    cancelled: &tokio_util::sync::CancellationToken,
) -> Result<String, String> {
    let request = build_request(name, arguments)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    format_answer(
        &request,
        runtime.block_on(crate::client::send_ready(
            archive,
            &request,
            deadline_ms,
            cancelled,
        )),
    )
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
    Tool {
        id: Value,
        name: String,
        arguments: serde_json::Map<String, Value>,
        with_result_type: bool,
        deadline_ms: Option<i64>,
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
                    "protocolVersion": version, "capabilities": {"tools": {}},
                    "serverInfo": {"name": "polylogue", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": crate::manual::text(),
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
                    "capabilities": {"tools": {}}, "instructions": crate::manual::text(),
                    "ttlMs": 0, "cacheScope": "private",
                    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "polylogue", "version": env!("CARGO_PKG_VERSION")}},
                }),
            ));
        }
        let with_result_type = self.request_metadata || with_result_type;
        match method {
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
            Self::Tool {
                id,
                name,
                arguments,
                with_result_type,
                deadline_ms,
            } => {
                let (text, failed) =
                    match call_with_limit(archive, &name, arguments, deadline_ms, cancelled) {
                        Ok(text) => (text, false),
                        Err(text) => (text, true),
                    };
                let mut result =
                    json!({"content": [{"type": "text", "text": text}], "isError": failed});
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
                if let Reply::Tool { ref id, .. } = reply {
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

pub fn serve_stdio() -> io::Result<()> {
    serve(
        io::stdin().lock(),
        io::stdout(),
        &Archive::at(&Archive::default_root()),
    )
}

#[cfg(test)]
mod tests;
