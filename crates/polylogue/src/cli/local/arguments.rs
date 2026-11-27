//! Arguments of operations that run in the calling process. The same schemas
//! drive common's CLI rendering and word decoding.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};

fn claude_code_harness() -> crate::hook::Harness {
    crate::hook::Harness::ClaudeCode
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Empty {}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Daemon {
    /// Stop once the loop has no work, printing what it did
    #[serde(default)]
    pub until_idle: bool,
    /// Read no root locations, spools, tool databases or collectors: parse and derive only what the archive stores, and import what is given
    #[serde(default)]
    pub no_sources: bool,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SchemaDiff {
    /// Earlier schema JSON file
    #[schemars(extend("argument_form" = "positional"))]
    pub before: String,
    /// Later schema JSON file
    #[schemars(extend("argument_form" = "positional"))]
    pub after: String,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Tour {
    /// A new directory for the example home, archive and transcript
    #[schemars(extend("argument_form" = "positional"))]
    pub directory: String,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Install {
    /// The harness, browser, shell or account to install into
    #[schemars(extend("argument_form" = "positional"))]
    pub target: String,
    /// Remove the installation
    #[serde(default)]
    pub remove: bool,
    /// Print what would be written
    #[serde(default)]
    pub dry_run: bool,
    /// Add the MCP guidance to Claude Code, Codex or Gemini CLI global instructions
    #[serde(default)]
    pub instructions: bool,
    /// Chrome extension ID (defaults to the packaged extension)
    pub extension_id: Option<String>,
    /// The wrapper command for Chrome or shell completion (default: polylogue)
    pub program: Option<String>,
    /// Use this harness settings file
    #[schemars(extend("unlisted" = true))]
    pub settings: Option<String>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Hook {
    /// The hook event's name
    #[schemars(extend("argument_form" = "positional"))]
    pub event: String,
    /// The harness calling: its payload, output and session IDs
    #[serde(default = "claude_code_harness")]
    pub harness: crate::hook::Harness,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum HostHarness {
    ClaudeCode,
    GeminiCli,
    Hermes,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Host {
    /// The harness to run
    #[schemars(extend("argument_form" = "positional"))]
    pub harness: HostHarness,
    /// The socket the daemon connects to
    pub socket: Option<String>,
    /// The agent's working directory
    pub cwd: Option<String>,
    /// The hook spool its status lines go to
    pub spool: Option<String>,
    /// The session ID to start the agent with, where the harness takes one
    pub session_id: Option<String>,
}
