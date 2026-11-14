//! Arguments of operations that run in the calling process. The same schemas
//! drive common's CLI projection and word decoding.
use command_surface::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Empty {}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Daemon {
    #[serde(default)]
    pub until_idle: bool,
    #[serde(default)]
    pub no_sources: bool,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SchemaDiff {
    pub before: String,
    pub after: String,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Tour {
    pub directory: String,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstallAgent {
    pub harness: String,
    pub extension_id: Option<String>,
    #[serde(default)]
    pub remove: bool,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Hook {
    pub event: String,
    pub harness: Option<crate::hook::Harness>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Hooks {
    pub harness: Option<crate::hook::Harness>,
    pub settings: Option<String>,
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
    pub harness: HostHarness,
    pub socket: Option<String>,
    pub cwd: Option<String>,
    pub spool: Option<String>,
    pub session_id: Option<String>,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Shell {
    Bash,
    Elvish,
    Fish,
    Powershell,
    Zsh,
}
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Completions {
    pub shell: Shell,
    pub program: Option<String>,
}
