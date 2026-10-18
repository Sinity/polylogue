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
        /// Why the session starts: startup, resume, clear or compact.
        #[arg(long, default_value = "startup")]
        source: String,
    },
    /// Context for taking over another session: its prompts, notes, what
    /// it left open and its last answer.
    Takeover { session: String },
    /// Resume a session through its tool's own command. Without a session,
    /// the sessions in this directory that left something open.
    Continue {
        session: Option<String>,
        /// Run the native resume command instead of printing it.
        #[arg(long)]
        exec: bool,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    /// Search past work for something relevant now, with a receipt.
    Recall {
        query: Vec<String>,
        #[arg(long)]
        cwd: Option<PathBuf>,
    },
    /// Read or set a setting (`briefing.session_start off`).
    Setting {
        key: String,
        value: Option<String>,
        /// Only on this machine.
        #[arg(long)]
        machine: bool,
    },
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
    /// Add or remove Polylogue's hooks in Claude Code's settings.
    #[command(subcommand)]
    Hooks(HooksCommand),
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
    /// Add a note to one or more targets: session:<id>, message:<id>,
    /// tool:<session>#<call>, repo:<name>, cwd:<path>, work:<n>.
    Add {
        #[arg(required = true, num_args = 1..)]
        targets: Vec<String>,
        /// The note's text.
        #[arg(long, short = 'm')]
        body: String,
        /// note, lesson, handoff, decision, blocker, goal, question, finding,
        /// outcome, correction, judgment, title, claim, problem, forecast.
        #[arg(long, default_value = "note")]
        kind: String,
        /// Replace an earlier note: the new one joins its revision family.
        #[arg(long)]
        revises: Option<String>,
        /// Written by an agent (by its session when given).
        #[arg(long)]
        agent: Option<Option<String>>,
    },
    /// Tag targets: `key` or `key:value`.
    Tag {
        #[arg(required = true, num_args = 1..)]
        targets: Vec<String>,
        #[arg(long)]
        tag: String,
        /// Confidence or affinity, as JSON (`{"confidence": 0.8}`).
        #[arg(long)]
        value: Option<String>,
    },
    /// Withdraw a note and every revision of it.
    Withdraw {
        note: String,
        #[arg(long)]
        reason: Option<String>,
    },
    /// Notes, current revisions only unless --history.
    List {
        targets: Vec<String>,
        #[arg(long)]
        kind: Vec<String>,
        /// Only notes agents wrote.
        #[arg(long)]
        agents: bool,
        #[arg(long)]
        history: bool,
        #[arg(long)]
        withdrawn: bool,
        /// Notes on this directory and its repository.
        #[arg(long)]
        cwd: Option<PathBuf>,
        #[arg(long)]
        limit: Option<u64>,
    },
}

#[derive(Debug, Subcommand)]
pub enum HooksCommand {
    /// Merge Polylogue's hook entries into the settings file, keeping
    /// everything else. Installing twice changes nothing.
    Install {
        /// Default: `$CLAUDE_CONFIG_DIR/settings.json` or
        /// `~/.claude/settings.json`.
        #[arg(long)]
        settings: Option<PathBuf>,
    },
    /// Remove Polylogue's hook entries, keeping everything else.
    Uninstall {
        #[arg(long)]
        settings: Option<PathBuf>,
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
    let mut exec_native = false;
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
        Some(Command::Mcp) => {
            return match crate::mcp::serve_stdio() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("polylogue mcp: {e}");
                    ExitCode::FAILURE
                }
            };
        }
        Some(Command::Hook { event }) => return crate::hook::run(&event),
        Some(Command::Hooks(h)) => {
            let (path, on) = match h {
                HooksCommand::Install { settings } => (settings, true),
                HooksCommand::Uninstall { settings } => (settings, false),
            };
            let path = path.unwrap_or_else(crate::hook::settings_path);
            return match crate::hook::install(&path, on) {
                Ok(()) => {
                    println!(
                        "{} Polylogue's hooks {} {}",
                        if on { "installed" } else { "removed" },
                        if on { "in" } else { "from" },
                        path.display()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => fail(json, "io", &e.to_string(), 1),
            };
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
        Some(Command::Recall { query, cwd }) => Request::Context {
            request_id: request_id(),
            ask: Ask::Recall {
                text: query.join(" "),
                cwd: Some(cwd.unwrap_or_else(current_dir)),
                for_session: None,
            },
            sender: None,
        },
        Some(Command::Brief {
            session,
            cwd,
            source,
        }) => {
            let session = match session.map(|s| SessionId::parse(&s)).transpose() {
                Ok(s) => s,
                Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
            };
            Request::Context {
                request_id: request_id(),
                ask: Ask::SessionStart {
                    cwd: cwd.unwrap_or_else(current_dir),
                    session,
                    source: StartSource::parse(&source),
                },
                sender: None,
            }
        }
        Some(Command::Takeover { session }) => match SessionId::parse(&session) {
            Ok(session) => Request::Context {
                request_id: request_id(),
                ask: Ask::Takeover {
                    session,
                    for_session: None,
                },
                sender: None,
            },
            Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
        },
        Some(Command::Continue {
            session: None, cwd, ..
        }) => Request::Candidates {
            cwd: cwd.unwrap_or_else(current_dir),
            limit: None,
        },
        Some(Command::Continue {
            session: Some(session),
            exec,
            ..
        }) => match SessionId::parse(&session) {
            Ok(session) => {
                exec_native = exec;
                Request::Continue { session }
            }
            Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
        },
        Some(Command::Setting {
            key,
            value,
            machine,
        }) => Request::Setting {
            request_id: value.as_ref().map(|_| request_id()),
            key,
            value,
            machine,
        },
        Some(Command::Note(n)) => match note_request(n) {
            Ok(r) => r,
            Err(message) => return fail(json, "bad_request", &message, 2),
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
        Response::Ok { data } if exec_native => exec_plan(&data),
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

use crate::briefing::{Ask, StartSource};
use crate::daemon::{Request, Response};
use polylogue_model::{Author, NewNote, NoteKind, SessionId, Target};
use polylogue_store::Archive;
use polylogue_store::NoteFilter;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

fn request_id() -> String {
    polylogue_store::user::new_request_id()
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"))
}

fn targets(texts: &[String]) -> Result<Vec<Target>, String> {
    texts
        .iter()
        .map(|t| Target::parse(t).map_err(|e| e.to_string()))
        .collect()
}

/// Builds the request for a `note` subcommand.
fn note_request(n: NoteCommand) -> Result<Request, String> {
    let author = |agent: Option<Option<String>>| -> Result<Author, String> {
        match agent {
            None => Ok(Author::Operator),
            Some(None) => Ok(Author::Agent(None)),
            Some(Some(s)) => SessionId::parse(&s)
                .map(|s| Author::Agent(Some(s)))
                .map_err(|e| e.to_string()),
        }
    };
    Ok(match n {
        NoteCommand::Add {
            targets: t,
            body,
            kind,
            revises,
            agent,
        } => Request::AddNote {
            request_id: request_id(),
            note: NewNote {
                kind: NoteKind::parse(&kind)
                    .ok_or_else(|| format!("unknown note kind {kind:?}"))?,
                key: None,
                value: Some(body),
                author: author(agent)?,
                targets: targets(&t)?,
                relations: vec![],
                revises,
                valid_until: None,
            },
        },
        NoteCommand::Tag {
            targets: t,
            tag,
            value,
        } => Request::AddNote {
            request_id: request_id(),
            note: NewNote {
                kind: NoteKind::Tag,
                key: Some(tag),
                value,
                author: Author::Operator,
                targets: targets(&t)?,
                relations: vec![],
                revises: None,
                valid_until: None,
            },
        },
        NoteCommand::Withdraw { note, reason } => Request::Withdraw {
            request_id: request_id(),
            note,
            author: Author::Operator,
            reason,
        },
        NoteCommand::List {
            targets: t,
            kind,
            agents,
            history,
            withdrawn,
            cwd,
            limit,
        } => {
            let mut t = targets(&t)?;
            if let Some(cwd) = cwd {
                for d in crate::briefing::scope_dirs(&cwd) {
                    t.push(Target::Cwd(d));
                }
                if let Some(name) = cwd.file_name().and_then(|n| n.to_str()) {
                    t.push(Target::Repo(name.to_owned()));
                }
            }
            Request::Notes {
                filter: NoteFilter {
                    targets: t,
                    kinds: kind
                        .iter()
                        .map(|k| {
                            NoteKind::parse(k).ok_or_else(|| format!("unknown note kind {k:?}"))
                        })
                        .collect::<Result<_, _>>()?,
                    agents: agents.then_some(true),
                    history,
                    withdrawn,
                    after_seq: None,
                    limit,
                },
            }
        }
    })
}

/// Replaces this process with the session's native resume command.
fn exec_plan(data: &serde_json::Value) -> ExitCode {
    use std::os::unix::process::CommandExt;
    let Some(native) = data.get("native").filter(|n| !n.is_null()) else {
        eprintln!(
            "polylogue: this session has no native resume; continue with context instead: {}",
            data["context_assisted"]["program"]
                .as_str()
                .map_or_else(String::new, |_| format!(
                    "polylogue takeover {}",
                    data["session"].as_str().unwrap_or("")
                ))
        );
        return ExitCode::from(1);
    };
    let program = native["program"].as_str().unwrap_or("");
    let args: Vec<&str> = native["args"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|a| a.as_str())
        .collect();
    let mut cmd = std::process::Command::new(program);
    cmd.args(&args);
    if let Some(dir) = native["cwd"].as_str() {
        cmd.current_dir(dir);
    }
    let err = cmd.exec();
    eprintln!("polylogue: could not run {program}: {err}");
    ExitCode::FAILURE
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
        Request::Context { .. } if data["off"].as_bool() == Some(true) => {
            println!(
                "session-start briefs are off (`polylogue setting briefing.session_start on`)"
            );
        }
        Request::Context { .. } => {
            print!("{}", s(&data["prepared"]["text"]));
        }
        Request::Notes { .. } => {
            for n in data.as_array().into_iter().flatten() {
                let targets: Vec<String> = n["targets"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(s)
                    .collect();
                let label = n["key"]
                    .as_str()
                    .map_or_else(|| s(&n["kind"]), str::to_owned);
                println!(
                    "{}  [{}] {}  ({}; {})",
                    s(&n["id"]),
                    label,
                    s(&n["value"]),
                    targets.join(", "),
                    s(&n["author"])
                );
            }
        }
        Request::Candidates { .. } => {
            for c in data.as_array().into_iter().flatten() {
                let open = &c["open"];
                let mut why = Vec::new();
                if open["unanswered_prompt"].as_bool() == Some(true) {
                    why.push("unanswered prompt".to_owned());
                }
                if open["open_tool_call"].as_bool() == Some(true) {
                    why.push("unfinished tool call".to_owned());
                }
                if let Some(t) = open["todo"].as_str() {
                    why.push(format!("todo: {t}"));
                }
                if let Some(w) = open["waiting_on"].as_str() {
                    why.push(format!("waiting on {w}"));
                }
                for f in c["follow_ups"].as_array().into_iter().flatten() {
                    why.push(s(f));
                }
                println!(
                    "{}  {}  {}\n    {}",
                    s(&c["session"]["last_activity"]),
                    s(&c["session"]["id"]),
                    s(&c["session"]["title"]),
                    why.join("; ")
                );
            }
        }
        Request::Continue { .. } => {
            let line = |c: &serde_json::Value| -> String {
                let mut words = vec![s(&c["program"])];
                words.extend(c["args"].as_array().into_iter().flatten().map(s));
                let cmd = words.join(" ");
                match c["cwd"].as_str() {
                    Some(d) => format!("cd {d} && {cmd}"),
                    None => cmd,
                }
            };
            if !data["native"].is_null() {
                println!("native resume:      {}", line(&data["native"]));
            }
            println!("with context:       {}", line(&data["context_assisted"]));
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
