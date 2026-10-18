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

pub fn run(args: Args) -> ExitCode {
    let json = args.json;
    let request = match args.command {
        Some(Command::Daemon) => {
            return match crate::daemon::server::serve() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("polylogue: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        Some(Command::Completions { shell }) => {
            use clap::CommandFactory;
            clap_complete::generate(
                shell,
                &mut Args::command(),
                "polylogue",
                &mut std::io::stdout(),
            );
            return ExitCode::SUCCESS;
        }
        Some(Command::Status) => Request::Status,
        Some(Command::List(q)) if q.query.is_empty() => Request::List {
            limit: q.limit.unwrap_or(50),
            before: None,
        },
        Some(Command::List(q)) => Request::Query {
            text: q.query.join(" "),
            cursor: None,
        },
        Some(Command::Read(r)) => {
            let session = match SessionId::parse(&r.session) {
                Ok(s) => s,
                Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
            };
            Request::Read {
                session,
                view: format!("{:?}", r.view).to_lowercase(),
                format: format!("{:?}", r.format).to_lowercase(),
                logical: r.logical,
                budget: r.budget,
                cursor: r.cursor,
            }
        }
        Some(Command::Import { paths }) => Request::Import {
            request_id: request_id(),
            paths: paths
                .into_iter()
                .map(|p| std::path::absolute(&p).unwrap_or(p))
                .collect(),
        },
        Some(Command::Sql { statement }) => Request::Sql { statement },
        Some(Command::Recall { query }) => Request::Recall {
            text: query.join(" "),
        },
        Some(Command::Continue { session }) => match SessionId::parse(&session) {
            Ok(session) => Request::Continue { session },
            Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
        },
        Some(other) => {
            let name = format!("{other:?}");
            let name = name
                .split([' ', '(', '{'])
                .next()
                .unwrap_or("")
                .to_lowercase();
            return fail(json, "not_built", &format!("`{name}` is not built yet"), 2);
        }
        None => Request::Query {
            text: args.query.join(" "),
            cursor: None,
        },
    };
    let root = Archive::default_root();
    let response = match send(&Archive::at(&root).socket(), &request) {
        Ok(r) => r,
        Err(e) if is_down(&e) => {
            return fail(
                json,
                "daemon_required",
                &format!(
                    "no daemon is serving {}; start it with `polylogue daemon`",
                    root.display()
                ),
                crate::EXIT_DAEMON_REQUIRED,
            );
        }
        Err(e) => return fail(json, "io", &e.to_string(), 1),
    };
    match response {
        Response::Ok { data } => {
            if json {
                println!("{data}");
            } else {
                print_plain(&request, &data);
            }
            ExitCode::SUCCESS
        }
        Response::Accepted { request_id } => {
            println!("accepted {request_id}");
            ExitCode::SUCCESS
        }
        Response::Error { kind, message } => fail(json, &kind, &message, 1),
    }
}

use crate::daemon::{Request, Response};
use polylogue_model::SessionId;
use polylogue_store::Archive;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

fn request_id() -> String {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{ns:x}-{:x}", std::process::id())
}

fn is_down(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
    )
}

/// Sends one request and reads its response.
pub fn send(socket: &std::path::Path, request: &Request) -> std::io::Result<Response> {
    let mut stream = UnixStream::connect(socket)?;
    let mut line = serde_json::to_string(request).map_err(std::io::Error::other)?;
    line.push('\n');
    stream.write_all(line.as_bytes())?;
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply)?;
    serde_json::from_str(&reply).map_err(std::io::Error::other)
}

fn fail(json: bool, kind: &str, message: &str, code: u8) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({ "error": { "kind": kind, "message": message } })
        );
    } else {
        eprintln!("polylogue: {kind}: {message}");
    }
    ExitCode::from(code)
}

/// Plain terminal output until the renderer (architecture 7, `render`)
/// takes these over: one line per session, one paragraph per message.
fn print_plain(request: &Request, data: &serde_json::Value) {
    let s = |v: &serde_json::Value| v.as_str().unwrap_or("").to_owned();
    match request {
        Request::List { .. } => {
            for row in data.as_array().into_iter().flatten() {
                println!(
                    "{:<25} {:>6}  {}  {}",
                    s(&row["last_activity"]),
                    row["messages"],
                    s(&row["id"]),
                    s(&row["title"])
                );
            }
        }
        Request::Read { .. } => {
            for m in data["messages"].as_array().into_iter().flatten() {
                println!("── {} {}", s(&m["role"]), s(&m["at"]));
                for b in m["blocks"].as_array().into_iter().flatten() {
                    match b["type"].as_str() {
                        Some("text") => println!("{}", s(&b["text"])),
                        Some("tool_call") => println!(
                            "→ {} {}",
                            s(&b["tool"]),
                            b["command"].as_str().or(b["path"].as_str()).unwrap_or("")
                        ),
                        Some("tool_result") => println!("← {}", b["outcome"]["kind"]),
                        Some(other) => println!("[{other}]"),
                        None => {}
                    }
                }
                println!();
            }
        }
        _ => println!("{}", serde_json::to_string_pretty(data).unwrap_or_default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_daemon_is_daemon_required() {
        let dir = tempfile::tempdir().unwrap();
        let err = send(&dir.path().join("daemon.sock"), &Request::Status).unwrap_err();
        assert!(is_down(&err));
    }
}
