//! MCP over stdio, as a proxy to the daemon: one tool per command the
//! table offers over MCP, named by its words joined with `_`. Each call is
//! checked against the table, sent as the same request the CLI sends, and
//! answered with the daemon's JSON; context (briefs, takeovers, recalls) is
//! answered with its text and recorded as a delivery from `mcp`. Refusals
//! are tool errors.

use crate::commands::{self, Action, COMMANDS, Caller, Command, Surface, Values};
use crate::daemon::{Request, Response};
use polylogue_store::Archive;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler, ServiceExt};
use std::path::PathBuf;
use std::sync::Arc;

pub struct Polylogue {
    archive: PathBuf,
    disconnected: tokio_util::sync::CancellationToken,
}

/// The commands offered as tools.
pub fn tool_commands() -> impl Iterator<Item = &'static Command> {
    COMMANDS
        .iter()
        .filter(|c| c.mcp && matches!(c.action, Action::Request(_)))
}

pub fn tools() -> Vec<Tool> {
    tool_commands()
        .map(|c| Tool::new(c.mcp_name(), c.help, Arc::new(commands::schema(c))))
        .collect()
}

/// An error as tool text, with the corrected query when the daemon formed
/// one, so an agent can retry with it as given.
fn error_text(kind: &str, message: &str, detail: Option<&serde_json::Value>) -> String {
    match detail.and_then(|d| d["suggestion"].as_str()) {
        Some(s) => format!("error ({kind}): {message}\ntry: {s}"),
        None => format!("error ({kind}): {message}"),
    }
}

/// In-process tests use the production async socket proxy.
#[cfg(test)]
pub(crate) fn call(
    archive: &Archive,
    name: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
    let request = build_request(name, arguments)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    format_answer(&request, runtime.block_on(send(archive, &request, None)))
}

fn build_request(
    name: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
) -> Result<Request, String> {
    let command = tool_commands()
        .find(|c| c.mcp_name() == name)
        .ok_or_else(|| format!("error (bad_request): no tool {name}"))?;
    let Action::Request(build) = command.action else {
        return Err(format!("error (bad_request): no tool {name}"));
    };
    let bad = |e: String| format!("error (bad_request): {e}");
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
        Ok(Response::Ok { data }) if matches!(request, Request::Context { .. }) => {
            Ok(data["prepared"]["text"]
                .as_str()
                .unwrap_or_default()
                .to_owned())
        }
        Ok(Response::Ok { data }) => Ok(serde_json::to_string_pretty(&data).unwrap_or_default()),
        Ok(Response::Error {
            kind,
            message,
            detail,
        }) => Err(error_text(&kind, &message, detail.as_ref())),
        Err(e) => Err(format!(
            "error (daemon_required): no Polylogue daemon answered ({e}); start it with `polylogue daemon`"
        )),
    }
}

impl ServerHandler for Polylogue {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("polylogue", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Polylogue: the archive of your past AI sessions. Use brief at the start of work, recall to find earlier work, takeover to continue another session.",
            )
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tools()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let answer = match build_request(&request.name, request.arguments.unwrap_or_default()) {
            Ok(request) => {
                let archive = Archive::at(&self.archive);
                tokio::select! {
                    response = send(&archive, &request, context.meta.get("deadline_ms").and_then(serde_json::Value::as_i64)) => format_answer(&request, response),
                    _ = context.ct.cancelled() => Err("error (cancelled): the read was cancelled".into()),
                    _ = self.disconnected.cancelled() => Err("error (cancelled): the MCP client disconnected".into()),
                }
            }
            Err(error) => Err(error),
        };
        Ok(match answer {
            Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
            Err(text) => CallToolResult::error(vec![ContentBlock::text(text)]),
        }
        .into())
    }
}

/// Dropping this future, including on MCP cancellation or EOF, closes the
/// socket. No blocking proxy worker can outlive the MCP request.
async fn send(
    archive: &Archive,
    request: &Request,
    deadline_ms: Option<i64>,
) -> std::io::Result<Response> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let duration = crate::daemon::execution::remaining(deadline_ms);
    let outgoing = crate::daemon::execution::Outgoing {
        request,
        deadline_ms: deadline_ms.unwrap_or_else(|| crate::daemon::execution::deadline(duration)),
    };
    let mut bytes = serde_json::to_vec(&outgoing).map_err(std::io::Error::other)?;
    bytes.push(b'\n');
    match tokio::time::timeout(duration + std::time::Duration::from_secs(1), async {
        let mut pause = std::time::Duration::from_millis(50);
        let mut stream = loop {
            match tokio::net::UnixStream::connect(archive.socket()).await {
                Ok(stream) => break stream,
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
                    ) && archive.lock_held() =>
                {
                    tokio::time::sleep(pause).await;
                    pause = (pause * 2).min(std::time::Duration::from_millis(500));
                }
                Err(error) => return Err(error),
            }
        };
        stream.write_all(&bytes).await?;
        let mut response = String::new();
        BufReader::new(stream).read_line(&mut response).await?;
        serde_json::from_str(&response).map_err(std::io::Error::other)
    })
    .await
    {
        Ok(result) => result,
        Err(_) => Ok(crate::daemon::service::error(
            "deadline_exceeded",
            "the daemon did not answer before the deadline",
        )),
    }
}

/// EOF must stop read work before the MCP library drains its handlers.
struct Input<R> {
    inner: R,
    disconnected: tokio_util::sync::CancellationToken,
}

impl<R: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for Input<R> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let before = buffer.filled().len();
        let has_room = buffer.remaining() > 0;
        let result = std::pin::Pin::new(&mut self.inner).poll_read(cx, buffer);
        if matches!(&result, std::task::Poll::Ready(Err(_)))
            || matches!(&result, std::task::Poll::Ready(Ok(())))
                && has_room
                && buffer.filled().len() == before
        {
            self.disconnected.cancel();
        }
        result
    }
}

impl<R> Drop for Input<R> {
    fn drop(&mut self) {
        self.disconnected.cancel();
    }
}

pub fn serve_stdio() -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let disconnected = tokio_util::sync::CancellationToken::new();
        let (input, output) = rmcp::transport::stdio();
        let input = Input {
            inner: input,
            disconnected: disconnected.clone(),
        };
        let service = Polylogue {
            archive: Archive::default_root(),
            disconnected,
        }
        .serve((input, output))
        .await
        .map_err(std::io::Error::other)?;
        service.waiting().await.map_err(std::io::Error::other)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests;
