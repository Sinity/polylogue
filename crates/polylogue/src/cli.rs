//! The command tree. A bare query (`polylogue repo:x since:7d`) runs the
//! query language; subcommands cover everything else. Every command takes
//! `--json` for programs and agents.

use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "polylogue",
    version,
    about = "Search, read and measure your AI sessions"
)]
pub struct Args {
    /// Machine-readable output.
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Option<Command>,
    /// A query, when no subcommand is given.
    #[arg(trailing_var_arg = true)]
    pub query: Vec<String>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// What the archive holds and what the daemon is doing.
    Status,
    /// Run the daemon in the foreground.
    Daemon,
    /// Bring in an export (ChatGPT, claude.ai, Gemini, a Polylogue export).
    Import { paths: Vec<PathBuf> },
    /// Sessions matching a query, newest first.
    List(QueryArgs),
    /// Read one session.
    Read(ReadArgs),
    /// Write sessions out as files.
    Export {
        sessions: Vec<String>,
        #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
        format: ExportFormat,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Notes and tags on sessions, messages and tool calls.
    #[command(subcommand)]
    Note(NoteCommand),
    /// The session-start brief for a directory or session.
    Brief {
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    /// Resume a session through its tool's own command.
    Continue { session: String },
    /// Search past work for something relevant now, with a receipt.
    Recall { query: Vec<String> },
    /// Token use, cost and subscription windows.
    #[command(subcommand)]
    Measure(MeasureCommand),
    /// The work graph: goals, attempts, verifications.
    #[command(subcommand)]
    Work(WorkCommand),
    /// Change notices and where they were delivered.
    Notices,
    /// Read-only SQL over the stable views.
    Sql { statement: String },
    /// Remove content permanently from every file and derived product.
    Excise {
        target: String,
        #[arg(long)]
        yes: bool,
    },
    /// Embeddings: cost preview, build, status.
    #[command(subcommand)]
    Embed(EmbedCommand),
    /// The Sinex backend: status and the one-time move to primary.
    #[command(subcommand)]
    Sinex(SinexCommand),
    /// Open the web reader.
    Web,
    /// Serve MCP over stdio.
    Mcp,
    /// Called by a harness hook; reads the event from stdin.
    Hook { event: String },
    /// Shell completions.
    Completions { shell: clap_complete::Shell },
}

#[derive(Debug, ClapArgs)]
pub struct QueryArgs {
    pub query: Vec<String>,
    #[arg(long)]
    pub limit: Option<u64>,
}

#[derive(Debug, ClapArgs)]
pub struct ReadArgs {
    pub session: String,
    #[arg(long, value_enum, default_value_t = ViewArg::Full)]
    pub view: ViewArg,
    #[arg(long, value_enum, default_value_t = ExportFormat::Terminal)]
    pub format: ExportFormat,
    /// Pages sized for a model, at most this many tokens each.
    #[arg(long)]
    pub budget: Option<usize>,
    #[arg(long)]
    pub cursor: Option<String>,
    /// Include messages inherited from a parent session.
    #[arg(long)]
    pub logical: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ExportFormat {
    Terminal,
    Markdown,
    Html,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ViewArg {
    Full,
    Dialogue,
    Prompts,
    Actions,
}

#[derive(Debug, Subcommand)]
pub enum NoteCommand {
    Add {
        target: String,
        body: String,
        #[arg(long)]
        kind: Option<String>,
    },
    List {
        target: String,
    },
    Tag {
        target: String,
        tag: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum MeasureCommand {
    Usage(QueryArgs),
    Cost(QueryArgs),
    /// The 5-hour and weekly subscription windows: burn and forecast.
    Quota,
}

#[derive(Debug, Subcommand)]
pub enum WorkCommand {
    List,
    Show { id: i64 },
}

#[derive(Debug, Subcommand)]
pub enum EmbedCommand {
    Preview(QueryArgs),
    Build(QueryArgs),
    Status,
}

#[derive(Debug, Subcommand)]
pub enum SinexCommand {
    Status,
    Cutover,
}

pub fn run(_args: Args) -> ExitCode {
    // Architecture 6: send the request to the daemon; exit 6 when it is down.
    todo!()
}
