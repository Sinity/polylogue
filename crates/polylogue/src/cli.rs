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
    /// Explain an empty result: how many rows each condition keeps.
    #[arg(long, global = true)]
    pub why: bool,
    /// Continue from a previous page's `next`.
    #[arg(long, global = true)]
    pub cursor: Option<String>,
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
    /// The work graph: goals, attempts, verifications and artifacts, by a
    /// query over the `work` unit (`polylogue work kind:attempt state:open`).
    Work(QueryArgs),
    /// Sessions running now and what each is doing: computed from hook
    /// and process records and the time.
    Live,
    /// What changed since a cursor: sessions, removals and notes.
    Changes {
        /// A previous answer's `next`.
        #[arg(long)]
        since: Option<String>,
    },
    /// Signal notices and what happened to each. A signal is a note of
    /// kind `signal` whose value names a query and a delivery rule.
    Notices {
        #[arg(long)]
        limit: Option<u64>,
    },
    /// Read-only SQL over the stable views.
    Sql { statement: String },
    /// Remove content permanently from every file and derived product:
    /// `session:<id>`, `input:<id>` or `text:<string>`. Without `--yes`,
    /// shows what would go and changes nothing.
    Excise {
        target: String,
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        reason: Option<String>,
    },
    /// Embeddings: cost preview, build, status.
    #[command(subcommand)]
    Embed(EmbedCommand),
    /// The Sinex backend: status and the one-time move to primary.
    #[command(subcommand)]
    Sinex(SinexCommand),
    /// Open the web reader.
    Web,
    /// Register Polylogue with a browser: `chrome` writes the extension's
    /// native messaging host manifest, admitting only `--extension-id`.
    InstallAgent {
        harness: String,
        #[arg(long)]
        extension_id: Option<String>,
        /// Delete the manifest instead.
        #[arg(long)]
        remove: bool,
    },
    /// The browser extension's native messaging host; Chrome starts it.
    #[command(hide = true)]
    BrowserHost,
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
    /// Token lanes and cost by model, over a sessions query or everything.
    Usage(QueryArgs),
    /// The same report as `usage`: cost sits beside the tokens it prices.
    Cost(QueryArgs),
    /// The 5-hour and weekly subscription windows: burn and forecast.
    Quota,
    /// Tool outcomes: ok, error, no result and unknown, never one rate.
    Outcomes(QueryArgs),
    /// Time to a session's first failed tool call (Kaplan-Meier).
    Survival {
        /// Count human prompts instead of minutes.
        #[arg(long)]
        turns: bool,
        query: Vec<String>,
    },
    /// A daily series with its 7-day baseline and level changes:
    /// tokens, api_cost, sessions, prompts or tool_errors.
    Daily { metric: String, query: Vec<String> },
    /// Which tool call follows which, and what follows a failure.
    Process(QueryArgs),
    /// Prompts by weekday and hour.
    Rhythm(QueryArgs),
}

#[derive(Debug, Subcommand)]
pub enum EmbedCommand {
    /// What a build would embed and what it would cost.
    Preview(EmbedArgs),
    /// Embed the texts not yet embedded. A paid provider needs `--yes`.
    Build(EmbedArgs),
    /// Vectors per model, and the provider settings.
    Status,
}

#[derive(Debug, ClapArgs)]
pub struct EmbedArgs {
    /// A sessions query limiting the texts (default: every prose text).
    pub query: Vec<String>,
    /// At most this many texts.
    #[arg(long)]
    pub limit: Option<u64>,
    #[arg(long)]
    pub yes: bool,
}

#[derive(Debug, Subcommand)]
pub enum SinexCommand {
    Status,
    Cutover,
}

pub fn run(args: Args) -> ExitCode {
    let json = args.json;
    let mut exec_native = false;
    let mut export_out: Option<PathBuf> = None;
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
        Some(Command::Web) => return open_reader(json),
        Some(Command::BrowserHost) => return crate::browser::native::run(),
        Some(Command::InstallAgent {
            harness,
            extension_id,
            remove,
        }) => return install_agent(json, &harness, extension_id.as_deref(), remove),
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
            cursor: args.cursor.clone(),
            page_size: q.limit,
            why: args.why,
            session: None,
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
        Some(Command::Export {
            sessions,
            format,
            out,
        }) => {
            let ids: Result<Vec<SessionId>, _> =
                sessions.iter().map(|s| SessionId::parse(s)).collect();
            let ids = match ids {
                Ok(ids) if !ids.is_empty() => ids,
                Ok(_) => return fail(json, "bad_request", "name at least one session", 2),
                Err(e) => return fail(json, "bad_request", &e.to_string(), 2),
            };
            export_out = out;
            Request::Export {
                sessions: ids,
                format: format!("{format:?}").to_lowercase(),
            }
        }
        Some(Command::Embed(e)) => {
            let (what, args) = match e {
                EmbedCommand::Status => ("status", None),
                EmbedCommand::Preview(a) => ("preview", Some(a)),
                EmbedCommand::Build(a) => ("build", Some(a)),
            };
            Request::Embed {
                what: what.into(),
                query: args.as_ref().map(|a| a.query.join(" ")),
                limit: args.as_ref().and_then(|a| a.limit),
                yes: args.as_ref().is_some_and(|a| a.yes),
            }
        }
        Some(Command::Excise {
            target,
            yes,
            reason,
        }) => Request::Excise {
            request_id: request_id(),
            target,
            reason,
            dry_run: !yes,
        },
        Some(Command::Sql { statement }) => Request::Sql { statement },
        Some(Command::Work(q)) => Request::Query {
            text: if q.query.is_empty() {
                "work".into()
            } else {
                format!("work where {}", q.query.join(" "))
            },
            cursor: args.cursor.clone(),
            page_size: q.limit,
            why: args.why,
            session: None,
        },
        Some(Command::Live) => Request::Live { now: None },
        Some(Command::Notices { limit }) => Request::Notices { limit },
        Some(Command::Changes { since }) => Request::Changes { since },
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
        Some(Command::Measure(m)) => {
            let q = |words: Vec<String>| (!words.is_empty()).then(|| words.join(" "));
            let (what, query) = match m {
                MeasureCommand::Usage(a) => ("usage".to_owned(), q(a.query)),
                MeasureCommand::Cost(a) => ("cost".to_owned(), q(a.query)),
                MeasureCommand::Quota => ("quota".to_owned(), None),
                MeasureCommand::Outcomes(a) => ("outcomes".to_owned(), q(a.query)),
                MeasureCommand::Survival { turns, query } => (
                    if turns { "survival-turns" } else { "survival" }.to_owned(),
                    q(query),
                ),
                MeasureCommand::Daily { metric, query } => (format!("daily:{metric}"), q(query)),
                MeasureCommand::Process(a) => ("process".to_owned(), q(a.query)),
                MeasureCommand::Rhythm(a) => ("rhythm".to_owned(), q(a.query)),
            };
            Request::Measure { what, query }
        }
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
            cursor: args.cursor.clone(),
            page_size: None,
            why: args.why,
            session: None,
        },
    };
    let root = Archive::default_root();
    let response = match send_when_ready(&Archive::at(&root), &request) {
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
        Response::Ok { data } if matches!(request, Request::Export { .. }) => {
            write_export(&data, export_out.as_deref(), json)
        }
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
        Response::Error {
            kind,
            message,
            detail,
        } => fail_detail(json, &kind, &message, detail, 1),
    }
}

/// Writes an export to `out`, or to standard output.
fn write_export(data: &serde_json::Value, out: Option<&std::path::Path>, json: bool) -> ExitCode {
    let text = match data.get("content").and_then(|c| c.as_str()) {
        Some(content) => content.to_owned(),
        None => match serde_json::to_string_pretty(data) {
            Ok(t) => t,
            Err(e) => return fail(json, "internal", &e.to_string(), 1),
        },
    };
    match out {
        Some(path) => match std::fs::write(path, text.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => fail(json, "io", &format!("{}: {e}", path.display()), 1),
        },
        None => {
            print!("{text}");
            ExitCode::SUCCESS
        }
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

/// Sends one request, waiting for a daemon that is still starting: one
/// that holds the archive lock but has not bound its socket yet (opening a
/// large archive takes seconds). With no daemon at all, the error is
/// returned at once.
pub fn send_when_ready(archive: &Archive, request: &Request) -> std::io::Result<Response> {
    let mut pause = std::time::Duration::from_millis(50);
    let mut told = false;
    loop {
        match send(&archive.socket(), request) {
            Err(e) if is_down(&e) && archive.lock_held() => {
                if !told {
                    eprintln!("polylogue: waiting for the daemon to finish opening the archive");
                    told = true;
                }
                std::thread::sleep(pause);
                pause = (pause * 2).min(std::time::Duration::from_millis(500));
            }
            other => return other,
        }
    }
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

/// Chrome's and Chromium's host manifest directories under `config`. An
/// existing browser config directory gets the manifest; with none, Chrome's
/// does.
pub fn host_manifest_dirs(config: &std::path::Path) -> Vec<std::path::PathBuf> {
    let dirs: Vec<_> = ["google-chrome", "chromium"]
        .iter()
        .map(|b| config.join(b))
        .filter(|d| d.is_dir())
        .collect();
    let dirs = if dirs.is_empty() {
        vec![config.join("google-chrome")]
    } else {
        dirs
    };
    dirs.into_iter()
        .map(|d| d.join("NativeMessagingHosts"))
        .collect()
}

fn install_agent(json: bool, harness: &str, extension_id: Option<&str>, remove: bool) -> ExitCode {
    if harness != "chrome" {
        return fail(
            json,
            "not_built",
            &format!(
                "`install-agent {harness}` is not built yet; for Claude Code hooks use `polylogue hooks install`"
            ),
            1,
        );
    }
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let file = format!("{}.json", crate::browser::HOST_NAME);
    let mut changed = Vec::new();
    for dir in host_manifest_dirs(&config) {
        let path = dir.join(&file);
        let result = if remove {
            match std::fs::remove_file(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                other => other,
            }
        } else {
            let Some(id) = extension_id
                .filter(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_lowercase()))
            else {
                return fail(
                    json,
                    "bad_request",
                    "`--extension-id` (the ID chrome://extensions shows) is required",
                    2,
                );
            };
            let binary = std::env::current_exe().and_then(std::fs::canonicalize);
            let Ok(binary) = binary else {
                return fail(json, "io", "cannot resolve this binary's path", 1);
            };
            let manifest = crate::browser::native::manifest(&binary, id);
            std::fs::create_dir_all(&dir).and_then(|()| {
                std::fs::write(
                    &path,
                    serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
                )
            })
        };
        if let Err(e) = result {
            return fail(json, "io", &format!("{}: {e}", path.display()), 1);
        }
        changed.push(path);
    }
    if json {
        println!(
            "{}",
            serde_json::json!({ "removed": remove, "files": changed })
        );
    } else {
        for p in &changed {
            println!(
                "{} {}",
                if remove { "removed" } else { "wrote" },
                p.display()
            );
        }
        if changed.is_empty() {
            println!("nothing to remove");
        }
    }
    ExitCode::SUCCESS
}

/// Opens the web reader: reads the running daemon's port and token, checks
/// that the port's listener is this user's, and opens `/auth#<token>` in
/// the browser. The fragment never reaches a server log or history.
fn open_reader(json: bool) -> ExitCode {
    use crate::web::access;
    let root = polylogue_store::Archive::default_root();
    let dir = access::runtime_dir(&access::archive_key(&root));
    let (Ok(token), Ok(port)) = (
        std::fs::read_to_string(dir.join("token")),
        std::fs::read_to_string(dir.join("port")),
    ) else {
        return fail(
            json,
            "daemon_required",
            "the daemon is not serving the web reader; start `polylogue daemon`",
            crate::EXIT_DAEMON_REQUIRED,
        );
    };
    let Ok(port) = port.trim().parse::<u16>() else {
        return fail(json, "io", "the port file is unreadable", 1);
    };
    if access::listener_owner(port).is_none() || access::listener_owner(port) != access::my_uid() {
        return fail(
            json,
            "refused",
            &format!("port {port} is not served by this user's daemon; not sending the token"),
            1,
        );
    }
    let url = format!("http://127.0.0.1:{port}/");
    let auth = format!("http://127.0.0.1:{port}/auth#{}", token.trim());
    let opened = std::process::Command::new("xdg-open")
        .arg(&auth)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if json {
        println!("{}", serde_json::json!({ "url": url, "opened": opened }));
    } else if opened {
        println!("opened {url}");
    } else {
        // The operator's own terminal: the sign-in link may be shown here.
        eprintln!("polylogue: could not start a browser; open {auth}");
    }
    ExitCode::SUCCESS
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

/// An error with structured detail: printed whole for programs; for a
/// person, the message and the corrected query when there is one.
fn fail_detail(
    json: bool,
    kind: &str,
    message: &str,
    detail: Option<serde_json::Value>,
    code: u8,
) -> ExitCode {
    if json {
        println!(
            "{}",
            serde_json::json!({ "error": { "kind": kind, "message": message, "detail": detail } })
        );
    } else {
        eprintln!("polylogue: {kind}: {message}");
        if let Some(s) = detail.as_ref().and_then(|d| d["suggestion"].as_str()) {
            eprintln!("polylogue: try: {s}");
        }
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
        Request::Measure { what, .. } => print!("{}", crate::measure_text::render(what, data)),
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
        Request::Query { .. } => {
            let _ = polylogue_render::results::render(
                data,
                polylogue_render::Format::Terminal,
                &mut std::io::stdout().lock(),
            );
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
        assert!(!Archive::at(dir.path()).lock_held());
        let err = send_when_ready(&Archive::at(dir.path()), &Request::Status).unwrap_err();
        assert!(is_down(&err));
    }

    #[test]
    fn a_starting_daemon_is_waited_for() {
        let dir = tempfile::tempdir().unwrap();
        let archive = Archive::at(dir.path());
        // The daemon's order: take the lock, open the archive, then bind.
        let (_, lock) = Archive::open_or_create(dir.path()).unwrap();
        assert!(archive.lock_held());
        let socket = archive.socket();
        let server = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
            let (stream, _) = listener.accept().unwrap();
            let mut line = String::new();
            std::io::BufRead::read_line(&mut std::io::BufReader::new(&stream), &mut line).unwrap();
            let reply = serde_json::to_string(&Response::Ok {
                data: serde_json::json!({"ready": true}),
            })
            .unwrap();
            std::io::Write::write_all(&mut &stream, format!("{reply}\n").as_bytes()).unwrap();
            drop(lock);
        });
        let response = send_when_ready(&archive, &Request::Status).unwrap();
        assert!(matches!(response, Response::Ok { .. }));
        server.join().unwrap();
    }
}
