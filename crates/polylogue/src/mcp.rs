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

/// One tool call: the answer's text, or the refusal.
pub fn call(
    archive: &Archive,
    name: &str,
    arguments: serde_json::Map<String, serde_json::Value>,
) -> Result<String, String> {
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
    let request = build(command, &values, &caller).map_err(bad)?;
    match crate::cli::send_when_ready(archive, &request) {
        Ok(Response::Ok { data, .. }) if matches!(request, Request::Context { .. }) => {
            Ok(data["prepared"]["text"]
                .as_str()
                .unwrap_or_default()
                .to_owned())
        }
        Ok(Response::Ok { data, .. }) => {
            Ok(serde_json::to_string_pretty(&data).unwrap_or_default())
        }
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
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let archive = Archive::at(&self.archive);
        let name = request.name.into_owned();
        let arguments = request.arguments.unwrap_or_default();
        let answer = tokio::task::spawn_blocking(move || call(&archive, &name, arguments)).await;
        Ok(match answer {
            Ok(Ok(text)) => CallToolResult::success(vec![ContentBlock::text(text)]),
            Ok(Err(text)) => CallToolResult::error(vec![ContentBlock::text(text)]),
            Err(e) => {
                CallToolResult::error(vec![ContentBlock::text(format!("error (internal): {e}"))])
            }
        }
        .into())
    }
}

pub fn serve_stdio() -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let service = Polylogue {
            archive: Archive::default_root(),
        }
        .serve(rmcp::transport::stdio())
        .await
        .map_err(std::io::Error::other)?;
        service.waiting().await.map_err(std::io::Error::other)?;
        Ok(())
    })
}

#[cfg(test)]
mod tests;
