//! The command table: every operation once, with its help and the
//! surfaces that offer it. The CLI and its completions, the MCP tools and
//! the HTTP API under `/api/v1/` are built from it, and its command names
//! are the words a query may not start with.
//!
//! A row's arguments are declared on its request type ([`arguments`]): a
//! field's doc comment is its help, its serde default its default. An
//! argument has one name everywhere: `--name` on the command line (with
//! `-` for `_`), the property of an MCP tool's input, the field of an HTTP
//! request. Arguments become one checked JSON object by the shared parser.
//! Options acting on the calling machine are typed structs too
//! (`cli::calling_process`). HTTP-only structured bodies use their owning
//! request type; binary inputs reach the daemon directly as bytes.

pub(crate) mod arguments;

use crate::cli::{calling_process, local};
use crate::daemon::handler::{FeedRegistration, Registration, daemon, feed, streams};
use crate::daemon::measure::{MEASURES, Measure};
use crate::daemon::service::handlers as H;
use crate::daemon::service::{acts, browser, feeds, follow, read};
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::LazyLock;

/// An MCP tool: a family of commands offered as its operations. Every
/// command of a tool writes, or every one reads, so its hints hold for
/// each operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Read,
    Context,
    Note,
    Chat,
    Act,
}

impl Tool {
    pub const ALL: [Tool; 5] = [Tool::Read, Tool::Context, Tool::Note, Tool::Chat, Tool::Act];

    pub fn name(self) -> &'static str {
        match self {
            Tool::Read => "polylogue_read",
            Tool::Context => "polylogue_context",
            Tool::Note => "polylogue_note",
            Tool::Chat => "polylogue_chat",
            Tool::Act => "polylogue_act",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Tool::Read => "Query and read the archive",
            Tool::Context => "Briefs, recall, takeovers and file advisories",
            Tool::Note => "Assertions, saved queries, watches, settings",
            Tool::Chat => "Web chats via the browser extension",
            Tool::Act => "Acts on live sessions: ping, answer, interrupt, start, hand-off",
        }
    }

    /// A retried `request_id` is applied once. Chat and backfill requests
    /// start a new browser action each time, and acts reach a running
    /// harness or start one.
    pub fn idempotent(self) -> bool {
        !matches!(self, Tool::Chat | Tool::Act)
    }

    /// Its operations act outside the archive (a web chat, a harness).
    pub fn open_world(self) -> bool {
        matches!(self, Tool::Chat | Tool::Act)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cli {
    Subcommand,
    /// A subcommand left out of help: programs run it, people do not.
    Hidden,
    /// What `polylogue` runs when no subcommand is given.
    NoSubcommand,
}

pub type OutputHandler = fn(
    &command_surface::ParsedCommand,
    &command_surface::Answer<Value>,
    &crate::config::Startup,
    bool,
) -> ExitCode;

/// What the command line does in place of one request and its answer: it
/// writes a file or a directory, or follows a stream.
pub type Deliver = fn(
    &command_surface::ParsedCommand,
    &crate::config::Startup,
    &command_surface::RunnableRequest,
) -> ExitCode;

/// Turns the command line's values into the request's arguments where
/// they differ.
pub type Prepare = fn(
    &command_surface::ParsedCommand,
    &std::path::Path,
    &mut Map<String, Value>,
) -> Result<(), String>;

/// What runs a command.
#[derive(Clone, Copy)]
pub enum Run {
    /// The daemon answers through the row's handler.
    Daemon(Registration),
    /// The daemon answers through the row's handler; the command line
    /// delivers the answer itself, as a page or a stream.
    Delivered {
        registration: Registration,
        deliver: Deliver,
    },
    /// A stream the command line follows over the socket.
    Feed {
        registration: FeedRegistration,
        deliver: Deliver,
    },
    /// Runs in the calling process (the CLI only), with `--json`.
    Local(
        fn(
            &Command,
            &command_surface::ParsedCommand,
            crate::config::OutputConfig,
            bool,
        ) -> ExitCode,
        fn(&mut command_surface::DeclarationsBuilder, &'static Command),
        fn(&Value) -> Result<(), String>,
    ),
    ConfiguredLocal(
        fn(&Command, &command_surface::ParsedCommand, &crate::config::Startup, bool) -> ExitCode,
        fn(&mut command_surface::DeclarationsBuilder, &'static Command),
        fn(&Value) -> Result<(), String>,
    ),
}

const fn local_command<A: command_surface::JsonSchema + serde::de::DeserializeOwned>(
    run: fn(
        &Command,
        &command_surface::ParsedCommand,
        crate::config::OutputConfig,
        bool,
    ) -> ExitCode,
) -> Run {
    Run::Local(run, declare_local::<A>, validate_local::<A>)
}
const fn configured_local<A: command_surface::JsonSchema + serde::de::DeserializeOwned>(
    run: fn(&Command, &command_surface::ParsedCommand, &crate::config::Startup, bool) -> ExitCode,
) -> Run {
    Run::ConfiguredLocal(run, declare_local::<A>, validate_local::<A>)
}
fn validate_local<A: serde::de::DeserializeOwned>(arguments: &Value) -> Result<(), String> {
    A::deserialize(arguments)
        .map(drop)
        .map_err(|error| error.to_string())
}
fn declare_local<A: command_surface::JsonSchema>(
    declarations: &mut command_surface::DeclarationsBuilder,
    command: &'static Command,
) {
    let mut spec = specification::<A>(command);
    spec.kind = command_surface::OperationKind::Local;
    spec.http_route = None;
    if spec.reason.is_empty() {
        spec.reason = command.help.into();
    }
    declarations
        .declare_local::<A>(spec)
        .expect("local arguments match their declaration");
}

#[derive(Clone, Copy)]
pub struct Command {
    pub reason: &'static str,
    pub latency: command_surface::LatencyClass,
    pub failures: &'static [command_surface::FailureKind],
    /// The command-line words (`assert-query`), also the operation's name
    /// within its MCP tool; the HTTP route joins them with `/`.
    pub name: &'static str,
    pub help: &'static str,
    /// Options acting on the calling machine (`--to DIR`, `--exec`): the
    /// command line reads them and keeps them out of the request.
    pub calling_process: Option<fn() -> schemars::Schema>,
    /// Command lines on the Morrow archive that help shows under the row.
    pub examples: &'static [&'static str],
    pub run: Run,
    pub cli: Cli,
    /// The MCP tool that offers this command as an operation.
    pub tool: Option<Tool>,
    pub http: bool,
    /// Mutates state: POST over HTTP, where the origin is checked.
    pub writes: bool,
    /// Turns the command line's values into the request's arguments where
    /// they differ (`chat send` reads its `--attach` files here).
    pub prepare: Option<Prepare>,
    /// What the command line does with the answer in place of printing it
    /// (`export` writes a file, `continue --exec` runs the plan).
    pub output: Option<OutputHandler>,
}

impl Command {
    /// The handler registration of a row the daemon answers once.
    pub(crate) fn registration(&self) -> Option<&Registration> {
        match &self.run {
            Run::Daemon(registration) | Run::Delivered { registration, .. } => Some(registration),
            Run::Feed { .. } | Run::Local(..) | Run::ConfiguredLocal(..) => None,
        }
    }

    pub fn cli_words(&self) -> Vec<String> {
        if self.cli == Cli::NoSubcommand {
            vec!["find".into()]
        } else {
            self.name.split(' ').map(str::to_owned).collect()
        }
    }

    pub fn http_path(&self) -> String {
        format!("/{}", self.name.replace(' ', "/"))
    }
}

/// A word that groups commands on the command line (`note`, `measure`).
pub struct Group {
    pub name: &'static str,
    pub help: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    Cli,
    Mcp,
    Http,
    Hook,
}

/// Who is calling, for what a command's arguments leave open.
#[derive(Clone, Debug)]
pub struct Caller {
    pub surface: Surface,
    /// The caller's working directory, which `cwd` defaults to; HTTP
    /// callers have none and must name one.
    pub cwd: Option<PathBuf>,
}

impl Caller {
    /// The sender a prepared context is recorded under: MCP hands it to a
    /// model; the command line and the reader show it to a person.
    pub fn sender(&self) -> Option<String> {
        match self.surface {
            Surface::Mcp => Some(polylogue_model::DeliverySender::Mcp.as_str().to_owned()),
            Surface::Hook => Some(polylogue_model::DeliverySender::Hook.as_str().to_owned()),
            Surface::Cli | Surface::Http => None,
        }
    }
}

/// The live view as a query: what `polylogue live` runs.
pub const LIVE: &str = "sessions where running:true | sort by live_rank, last_event_at desc";
/// Sessions waiting on the operator, the longest wait first.
pub const WAITING: &str = "sessions where waiting:true | sort by live_since";
/// Signal notices as a query: what `polylogue notices` runs.
pub const NOTICES: &str = "deliveries where sender:signal | sort by at desc";

/// A command on the command line only, sent to the daemon unless its
/// action is local.
const CLI: Command = Command {
    reason: "",
    latency: command_surface::LatencyClass::Local,
    failures: &command_surface::FailureKind::ALL,
    name: "",
    help: "",
    calling_process: None,
    examples: &[],
    run: local_command::<local::arguments::Empty>(local::not_built),
    cli: Cli::Subcommand,
    tool: None,
    http: false,
    writes: false,
    prepare: None,
    output: None,
};

pub const GROUPS: &[Group] = &[
    Group {
        name: "schema",
        help: "Structure, counts and unread paths from stored revisions",
    },
    Group {
        name: "demo",
        help: "Explore the synthetic Morrow archive",
    },
    Group {
        name: "measure",
        help: "Token use, cost and subscription windows: `QUERY | measure WHAT` over the sessions a query selects",
    },
    Group {
        name: "embed",
        help: "Embeddings: cost preview, build, status",
    },
    Group {
        name: "hooks",
        help: "Add or remove Polylogue's hooks in Claude Code's, Codex's or Gemini CLI's settings",
    },
    Group {
        name: "chat",
        help: "Chat actions: turns the browser extension drafts or sends in a web chat",
    },
    Group {
        name: "backfill",
        help: "Capture a web chat's older conversations through the browser extension",
    },
];

/// Every command, in the order help lists them: the written ones, with a
/// `measure NAME` for each measurement after `setting`.
pub static COMMANDS: LazyLock<Vec<Command>> = LazyLock::new(|| {
    let mut all = WRITTEN.to_vec();
    let at = all
        .iter()
        .position(|c| c.name == "setting")
        .map_or(all.len(), |i| i + 1);
    all.splice(at..at, MEASURES.iter().map(measure_command));
    for c in &mut all {
        if c.name == "query" {
            c.help = query_help();
        }
    }
    all
});

/// `measure NAME`: the sessions query `QUERY | measure NAME`.
fn measure_command(m: &'static Measure) -> Command {
    Command {
        name: format!("measure {}", m.name).leak(),
        reason: "A named measurement over the sessions a query selects",
        help: m.help,
        run: Run::Daemon(daemon::<H::Measure>()),
        ..CLI
    }
}

/// The `query` help, naming every unit and measurement.
fn query_help() -> &'static str {
    let units: Vec<&str> = polylogue_query::ast::Unit::all()
        .map(polylogue_query::ast::Unit::name)
        .collect();
    format!(
        "Run a query: rows of a unit ({}) with stages (`| sort by`, `| group by`, `| matches`). \
         `| measure WHAT` ends a sessions query with a measurement: {}; unknown figures are null, never 0. \
         Each answer's `view.cursor` continues `changes where after:CURSOR`",
        units.join(", "),
        crate::daemon::measure::names()
    )
    .leak()
}

const WRITTEN: &[Command] = &[
    Command {
        name: "schema infer",
        reason: "Observe stored wire structure that parsed query rows cannot reconstruct",
        latency: command_surface::LatencyClass::Local,
        help: "Observe stored inputs at idle priority; default: latest revisions of every origin",
        calling_process: Some(calling_process::schema::<calling_process::SchemaOutput>),
        run: Run::Daemon(daemon::<H::SchemaInfer>()),
        output: Some(crate::cli::schema::output),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "schema unread",
        reason: "Expose stored wire paths outside the parser coverage",
        latency: command_surface::LatencyClass::Read,
        help: "Show stored wire paths neither read nor kept, with drift and coverage",
        calling_process: Some(calling_process::schema::<calling_process::SchemaOutput>),
        run: Run::Daemon(daemon::<H::SchemaUnread>()),
        output: Some(crate::cli::schema::output),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "schema diff",
        help: "Compare explicit inferred schema files recursively, with sample differences labeled",
        calling_process: Some(calling_process::schema::<calling_process::SchemaOutput>),
        run: local_command::<local::arguments::SchemaDiff>(crate::cli::schema::diff),
        ..CLI
    },
    Command {
        name: "demo tour",
        help: "Prepare a separate Morrow archive and walk through search, tool outcomes, parents, costs and notes",
        run: configured_local::<local::arguments::Tour>(crate::demo::tour),
        ..CLI
    },
    Command {
        name: "explain",
        reason: "Describe the language and a query's plan without running it",
        help: "Explain a query without executing it, or page through current syntax, units, fields, operators and capabilities",
        run: Run::Daemon(daemon::<H::Explain>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "manual",
        help: "The agent card and the MCP operation reference, every argument named (available without a daemon)",
        run: local_command::<local::arguments::Empty>(local::manual),
        ..CLI
    },
    Command {
        name: "query",
        help: "Run a query",
        reason: "Execute a question expressed in the archive query language",
        latency: command_surface::LatencyClass::Search,
        failures: &[
            command_surface::FailureKind::InvalidQuery,
            command_surface::FailureKind::UnknownField,
            command_surface::FailureKind::BadValue,
            command_surface::FailureKind::NotAQuery,
            command_surface::FailureKind::BadRequest,
            command_surface::FailureKind::Ambiguous,
            command_surface::FailureKind::StaleCursor,
            command_surface::FailureKind::NotYetAvailable,
            command_surface::FailureKind::NotFound,
            command_surface::FailureKind::Conflict,
            command_surface::FailureKind::Refused,
            command_surface::FailureKind::NotRunning,
            command_surface::FailureKind::Io,
            command_surface::FailureKind::Timeout,
            command_surface::FailureKind::Cancelled,
        ],
        run: Run::Daemon(daemon::<H::Query>()),
        cli: Cli::NoSubcommand,
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "status",
        help: "What the archive holds, what the daemon is doing and which sessions are running, waiting first",
        reason: "Orient the caller with measured archive and daemon state",
        latency: command_surface::LatencyClass::Status,
        run: Run::Daemon(daemon::<H::Status>()),
        output: Some(crate::cli::status_output),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "daemon",
        help: "Run the daemon in the foreground",
        run: configured_local::<local::arguments::Daemon>(local::daemon),
        ..CLI
    },
    Command {
        name: "import",
        help: "Store explicit files, directories and bundles through the daemon, or with --from-archive an archive written before the version 1 baseline. The typed report lists imported and already stored paths, classifications and refusals, packages and authored restoration counts, and for an older archive what became of each class of row. Primary authored restores awaiting confirmation are pending",
        reason: "Capture explicitly selected source bytes and restore the notes and other user records a bundle carries",
        latency: command_surface::LatencyClass::Aggregate,
        run: Run::Daemon(daemon::<H::Import>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "verify",
        reason: "Check every stored byte against its recorded hash without changing anything",
        latency: command_surface::LatencyClass::Local,
        help: "Check stored bytes at idle priority: every blob against its name, every extent and revision against the bytes it cites, the kept counters and each database; reports, never repairs",
        run: Run::Daemon(daemon::<H::Verify>()),
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "list",
        help: "Sessions matching a query, newest first",
        reason: "List archive rows through the query language",
        latency: command_surface::LatencyClass::Read,
        run: Run::Daemon(daemon::<H::List>()),
        http: true,
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "read",
        reason: "Render a selected query or reference and transfer its records",
        latency: command_surface::LatencyClass::Read,
        help: "Read a query or reference; write a consistent bundle with --to DIR",
        calling_process: Some(calling_process::schema::<calling_process::Read>),
        run: Run::Delivered {
            registration: streams::<read::Read>(),
            deliver: crate::cli::write_out::run,
        },
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "attachment",
        reason: "Retrieve the checked stored bytes behind an attachment descriptor",
        latency: command_surface::LatencyClass::Read,
        help: "Read a bounded range of the stored bytes for a message attachment",
        calling_process: Some(calling_process::schema::<calling_process::Attachment>),
        run: Run::Daemon(daemon::<H::Attachment>()),
        output: Some(crate::cli::attachment::output),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "assert",
        help: "Record checked assertions; the accepting transport supplies authorship",
        reason: "Record an authored assertion through the writer",
        latency: command_surface::LatencyClass::Local,
        failures: &[
            command_surface::FailureKind::BadRequest,
            command_surface::FailureKind::InvalidQuery,
            command_surface::FailureKind::UnknownField,
            command_surface::FailureKind::BadValue,
            command_surface::FailureKind::NotAQuery,
            command_surface::FailureKind::Ambiguous,
            command_surface::FailureKind::NotFound,
            command_surface::FailureKind::Conflict,
            command_surface::FailureKind::NotYetAvailable,
            command_surface::FailureKind::Refused,
            command_surface::FailureKind::NotRunning,
            command_surface::FailureKind::Io,
            command_surface::FailureKind::Timeout,
            command_surface::FailureKind::Cancelled,
        ],
        run: Run::Daemon(daemon::<H::Assert>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "assert-query",
        help: "Record one checked assertion about each selected query row",
        reason: "Record an authored assertion through the writer",
        latency: command_surface::LatencyClass::Search,
        failures: &[
            command_surface::FailureKind::BadRequest,
            command_surface::FailureKind::InvalidQuery,
            command_surface::FailureKind::UnknownField,
            command_surface::FailureKind::BadValue,
            command_surface::FailureKind::NotAQuery,
            command_surface::FailureKind::Ambiguous,
            command_surface::FailureKind::NotFound,
            command_surface::FailureKind::Conflict,
            command_surface::FailureKind::NotYetAvailable,
            command_surface::FailureKind::Refused,
            command_surface::FailureKind::NotRunning,
            command_surface::FailureKind::Io,
            command_surface::FailureKind::Timeout,
            command_surface::FailureKind::Cancelled,
        ],
        run: Run::Daemon(daemon::<H::AssertQuery>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "revise",
        help: "Revise assertion heads with one checked patch",
        reason: "Record an authored assertion through the writer",
        latency: command_surface::LatencyClass::Local,
        failures: &[
            command_surface::FailureKind::BadRequest,
            command_surface::FailureKind::InvalidQuery,
            command_surface::FailureKind::UnknownField,
            command_surface::FailureKind::BadValue,
            command_surface::FailureKind::NotAQuery,
            command_surface::FailureKind::Ambiguous,
            command_surface::FailureKind::NotFound,
            command_surface::FailureKind::Conflict,
            command_surface::FailureKind::NotYetAvailable,
            command_surface::FailureKind::Refused,
            command_surface::FailureKind::NotRunning,
            command_surface::FailureKind::Io,
            command_surface::FailureKind::Timeout,
            command_surface::FailureKind::Cancelled,
        ],
        run: Run::Daemon(daemon::<H::Revise>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "withdraw",
        help: "Withdraw assertion or variant roots, exact revisions, or withdrawals",
        reason: "Record an authored assertion through the writer",
        latency: command_surface::LatencyClass::Local,
        failures: &[
            command_surface::FailureKind::BadRequest,
            command_surface::FailureKind::InvalidQuery,
            command_surface::FailureKind::UnknownField,
            command_surface::FailureKind::BadValue,
            command_surface::FailureKind::NotAQuery,
            command_surface::FailureKind::Ambiguous,
            command_surface::FailureKind::NotFound,
            command_surface::FailureKind::Conflict,
            command_surface::FailureKind::NotYetAvailable,
            command_surface::FailureKind::Refused,
            command_surface::FailureKind::NotRunning,
            command_surface::FailureKind::Io,
            command_surface::FailureKind::Timeout,
            command_surface::FailureKind::Cancelled,
        ],
        run: Run::Daemon(daemon::<H::Withdraw>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "brief",
        reason: "Prepare the named context with its declared selection and token budget",
        latency: command_surface::LatencyClass::Search,
        help: "The session-start brief for a directory or session: the operator's notes there, other agents working there now, Beads claimed and open, what the recent sessions did, checks failing with no later pass, and signal notices; within its request token budget",
        run: Run::Daemon(daemon::<H::Brief>()),
        tool: Some(Tool::Context),
        http: true,
        ..CLI
    },
    Command {
        name: "advisory",
        reason: "Offer one operator-judged file pointer without delaying the prompt",
        latency: command_surface::LatencyClass::Read,
        help: "One current operator-judged decision or correction for a file named in a prompt; silent when unavailable within 20 ms or on cooldown",
        run: Run::Daemon(daemon::<H::Advisory>()),
        tool: Some(Tool::Context),
        http: true,
        ..CLI
    },
    Command {
        name: "takeover",
        reason: "Prepare the named context with its declared selection and token budget",
        latency: command_surface::LatencyClass::Search,
        help: "Context for taking over another session: its prompts, notes, what it left open and its last answer",
        run: Run::Daemon(daemon::<H::Takeover>()),
        tool: Some(Tool::Context),
        http: true,
        ..CLI
    },
    Command {
        name: "continue",
        reason: "Resume a session through its own tool, which no query can start",
        help: "Resume a session through its tool's own command. Without a session, the sessions in this directory that left something open (an unanswered prompt, an unfinished tool call, a todo, a handoff)",
        calling_process: Some(calling_process::schema::<calling_process::Continue>),
        run: Run::Daemon(daemon::<H::Continue>()),
        output: Some(crate::cli::resume_output),
        tool: Some(Tool::Context),
        http: true,
        ..CLI
    },
    Command {
        name: "recall",
        reason: "Prepare the named context with its declared selection and token budget",
        latency: command_surface::LatencyClass::Search,
        help: "Search past work for something relevant now: messages holding every word (in prose or tool output) and, when vectors exist, messages similar to the text, ranked together; grouped by session, with notes on them and a summary of how it searched. A page ends with the cursor of the next",
        run: Run::Daemon(daemon::<H::Recall>()),
        tool: Some(Tool::Context),
        http: true,
        ..CLI
    },
    Command {
        name: "setting",
        help: "List the typed runtime settings catalog, read KEY, or write KEY VALUE. Includes reading.variants, browser.actions, embed.* and plan.*. Standalone attention settings are local; in primary mode Sinex holds attention settings",
        reason: "Inspect declared runtime settings and author changes at their declared scope",
        latency: command_surface::LatencyClass::Local,
        run: Run::Daemon(daemon::<H::Setting>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "live",
        help: "Sessions running now and what each is doing: the query `sessions where running:true | sort by live_rank, last_event_at desc`",
        reason: "Show the current running-session view",
        latency: command_surface::LatencyClass::Read,
        run: Run::Daemon(daemon::<H::Live>()),
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "waiting",
        help: "Sessions waiting on you (a permission, a question, a plan or input), the longest wait first: the query `sessions where waiting:true | sort by live_since`",
        reason: "Show what blocks an agent now",
        latency: command_surface::LatencyClass::Read,
        run: Run::Daemon(daemon::<H::Waiting>()),
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "changes",
        help: "A finite JSON-lines change feed; persist its new position only after applying the complete batch",
        run: Run::Feed {
            registration: feed::<feeds::Changes>(),
            deliver: crate::cli::changes::run,
        },
        http: true,
        ..CLI
    },
    Command {
        name: "follow",
        help: "Follow a session or a sessions query: one line per live-state change (`→ waiting: permission`) and per new message or tool call, until interrupted",
        reason: "A live stream records nothing and cannot be a query's page",
        calling_process: Some(calling_process::schema::<calling_process::Follow>),
        run: Run::Feed {
            registration: feed::<follow::Follow>(),
            deliver: crate::cli::follow::run,
        },
        ..CLI
    },
    Command {
        name: "ping",
        help: "Send an addressed message to a session, or to every session a quoted query selects now (`@[sessions repo:x running:true]`): a running session gets it at its next tool result or prompt, one that ended is resumed headlessly with it. Each recipient gets one delivery; there is no undo",
        reason: "An authored act that reaches a session through its harness",
        run: Run::Daemon(daemon::<acts::Ping>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "answer",
        help: "Approve or deny a session's pending permission request; --standing answers the same tool and input whenever the session asks next. The first answer wins; there is no undo",
        reason: "An authored act on a request a session waits on",
        run: Run::Daemon(daemon::<acts::AnswerRequest>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "interrupt",
        help: "Stop a session's current turn, through the control connection of a session Polylogue hosts; refused otherwise",
        reason: "An authored act that reaches a session through its harness",
        run: Run::Daemon(daemon::<acts::Interrupt>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "start",
        help: "Start a session Polylogue hosts on the harness's control protocol (Codex's app-server, Claude Code stream-json, Gemini CLI and Hermes ACP): its approvals, questions and turns can then be answered, steered and interrupted from anywhere, and it survives a daemon restart. With --role, an instance of that role: its harness, with its instructions",
        reason: "Starts a harness session; no query selects one that does not exist yet",
        run: Run::Daemon(daemon::<acts::Start>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "hand-off",
        help: "Hand a session's work to a new hosted session in any harness, briefed with the session's takeover pack and the message to continue from (--at), in its directory; the new session records what it was briefed from and is never a resume or fork. --stop-source then interrupts the source; --print shows the pack and starts nothing",
        reason: "Starts a harness session; no query selects one that does not exist yet",
        run: Run::Daemon(daemon::<acts::HandOffSession>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "notices",
        help: "Signal notices and what happened to each: the query `deliveries where sender:signal | sort by at desc`",
        reason: "Read recorded signal deliveries and their outcomes",
        latency: command_surface::LatencyClass::Read,
        run: Run::Daemon(daemon::<H::Notices>()),
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "save",
        reason: "Record a named query as the operator's authored definition",
        help: "Save a query under a name, to run as `query:NAME`, alone or among other conditions. A `$word` standing for a value is a parameter, given when it runs: `query:NAME(word=VALUE)`. Saving a name again revises it",
        run: Run::Daemon(daemon::<H::Save>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "watch",
        reason: "Record a signal rule that delivers notices as rows start matching",
        help: "Watch a query: each row that starts matching is a notice in `status`, the next briefs and `notices`. Rows matching when the watch starts are its baseline. Withdraw it like a note",
        run: Run::Daemon(daemon::<H::Watch>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "keep",
        reason: "Freeze the rows a query returned now as a result that later reads cite",
        help: "Run a query and keep the rows it returned, to read back later as `result:<id>` (or `result:<id>.matches`) exactly as they were",
        run: Run::Daemon(daemon::<H::Keep>()),
        tool: Some(Tool::Note),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "sql",
        help: "Read-only SQL. Programs read the views v1_sessions, v1_messages, v1_tool_calls, v1_links and v1_assertions, whose names and columns stay; the tables change with the index format",
        reason: "Run expert read-only queries over the stable archive views",
        latency: command_surface::LatencyClass::Aggregate,
        run: Run::Daemon(daemon::<H::Sql>()),
        http: true,
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "embed preview",
        reason: "Show what an embedding build would read and cost before spending",
        help: "What a build would embed and what it would cost",
        run: Run::Daemon(daemon::<H::Embed>()),
        ..CLI
    },
    Command {
        name: "embed build",
        reason: "Compute and store embedding vectors through the configured provider",
        help: "Embed the texts not yet embedded. A paid provider needs `--yes`",
        run: Run::Daemon(daemon::<H::Embed>()),
        writes: true,
        ..CLI
    },
    Command {
        name: "embed status",
        reason: "Report the stored embedding vectors and what a build would add",
        help: "Vectors per model, and the provider settings",
        run: Run::Daemon(daemon::<H::EmbedStatus>()),
        ..CLI
    },
    Command {
        name: "chat send",
        reason: "Queue a chat action the browser extension carries out in a web chat",
        help: "Ask the browser extension to draft a turn in a web chat, or with `--submit` to send it. Needs the setting `browser.actions on`; a submit into an existing conversation waits for approval in the extension's popup",
        calling_process: Some(calling_process::schema::<calling_process::ChatSend>),
        run: Run::Daemon(daemon::<browser::ChatSend>()),
        prepare: Some(attach_files),
        tool: Some(Tool::Chat),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "chat list",
        reason: "Show queued chat actions and their progress in the extension",
        help: "Chat actions and what happened to each",
        run: Run::Daemon(daemon::<browser::ChatList>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "chat approve",
        reason: "The operator approves a chat action waiting in the extension",
        help: "Approve a held submit into an existing conversation",
        run: Run::Daemon(daemon::<browser::ChatApprove>()),
        writes: true,
        ..CLI
    },
    Command {
        name: "chat decline",
        reason: "The operator declines a chat action waiting in the extension",
        help: "Decline a held submit; the action is cancelled",
        run: Run::Daemon(daemon::<browser::ChatDecline>()),
        writes: true,
        ..CLI
    },
    Command {
        name: "backfill start",
        reason: "Start a backfill job the browser extension carries out",
        help: "Ask the browser extension to capture a web chat's older conversations: every one the archive lacks or holds an older copy of. One job runs per provider; starting again shows it",
        run: Run::Daemon(daemon::<browser::BackfillStart>()),
        tool: Some(Tool::Chat),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "backfill list",
        reason: "Show backfill jobs and their progress in the extension",
        help: "Backfill jobs and how far each got",
        run: Run::Daemon(daemon::<browser::BackfillList>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "backfill cancel",
        reason: "Stop a backfill job the browser extension is running",
        help: "Stop a backfill job",
        run: Run::Daemon(daemon::<browser::BackfillCancel>()),
        tool: Some(Tool::Chat),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "web",
        help: "Open the web reader",
        run: configured_local::<local::arguments::Empty>(local::web),
        ..CLI
    },
    Command {
        name: "install-agent",
        help: "Register Polylogue with a browser: `chrome` writes the extension's native messaging host manifest, which allows only `--extension-id`",
        run: configured_local::<local::arguments::InstallAgent>(local::install_agent),
        ..CLI
    },
    Command {
        name: "browser-host",
        help: "The browser extension's native messaging host; Chrome starts it",
        run: configured_local::<local::arguments::Empty>(local::browser_host),
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "mcp",
        help: "Serve MCP over stdio",
        run: configured_local::<local::arguments::Empty>(local::mcp),
        ..CLI
    },
    Command {
        name: "hook",
        help: "Called by a harness hook; reads the event from stdin and answers with context when the event takes it: the brief at session start, takeover context for a pasted session ID or supported web link",
        run: configured_local::<local::arguments::Hook>(local::hook),
        ..CLI
    },
    Command {
        name: "host",
        help: "Started by the daemon for a session it hosts over stdio (Claude Code stream-json, Gemini CLI and Hermes ACP): holds the agent's stdin and stdout so a daemon restart never ends it, and relays them to the daemon's connection",
        run: local_command::<local::arguments::Host>(local::host),
        ..CLI
    },
    Command {
        name: "hooks install",
        help: "Merge Polylogue's hook entries into the settings file, keeping everything else. Installing twice changes nothing",
        run: configured_local::<local::arguments::Hooks>(local::hooks_install),
        ..CLI
    },
    Command {
        name: "hooks uninstall",
        help: "Remove Polylogue's hook entries, keeping everything else",
        run: configured_local::<local::arguments::Hooks>(local::hooks_uninstall),
        ..CLI
    },
    Command {
        name: "completions",
        help: "The shell code that registers completion: commands and options, and while you type a query, its units, fields, values seen and recent session IDs from the daemon (`source <(polylogue completions zsh)`)",
        run: local_command::<local::arguments::Completions>(local::completions),
        ..CLI
    },
    Command {
        name: "timeline",
        reason: "Lay recorded activity out by logical day, which needs the calendar",
        help: "Paged recorded activity, calendar counts and usage in the configured logical day",
        run: Run::Daemon(daemon::<H::Timeline>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "cost",
        reason: "Total usage and price of selected sessions with response details",
        help: "Usage of selected sessions, optionally during one logical day, and paged per-response records",
        run: Run::Daemon(daemon::<H::Cost>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "complete",
        reason: "Complete an unfinished query at its cursor position",
        latency: command_surface::LatencyClass::Completion,
        help: "Complete a query at a UTF-8 byte offset using its full scope and suffix",
        run: Run::Daemon(daemon::<H::Complete>()),
        cli: Cli::Hidden,
        http: true,
        tool: Some(Tool::Read),
        ..CLI
    },
    // Rows the hooks, the browser extension's host and the operator's own
    // pages ask; no person types them.
    Command {
        name: "judge",
        reason: "The operator accepts or rejects one revision of an agent's note",
        help: "Accept or reject one revision of an agent's note",
        run: Run::Daemon(daemon::<H::Judge>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "delivery-event",
        reason: "A hook reports what happened to a delivery it was handed",
        help: "Record what happened to a delivery after it was handed out",
        run: Run::Daemon(daemon::<H::DeliveryEvent>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "mail-claim",
        reason: "A session's hook takes the pings and notices waiting for it",
        help: "Take the pings and notices pending for a session",
        run: Run::Daemon(daemon::<acts::Claim>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "permission",
        reason: "The synchronous permission hook asks for a decision given ahead",
        help: "A decision given ahead for a session's pending tool request",
        run: Run::Daemon(daemon::<acts::Permission>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "chat claim",
        reason: "The extension takes the oldest queued chat action",
        help: "Take the oldest queued chat action for an extension instance",
        run: Run::Daemon(daemon::<browser::ChatClaim>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "chat report",
        reason: "The extension reports the progress or outcome of a chat action",
        help: "Report a held chat action's progress or outcome",
        run: Run::Daemon(daemon::<browser::ChatReport>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "chat decide",
        reason: "The extension's popup records the operator's approval",
        help: "Approve or decline a chat action from the extension",
        run: Run::Daemon(daemon::<browser::ChatDecision>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "chat attachment",
        reason: "The extension fetches the bytes of a file a chat action attaches",
        help: "The bytes of a chat action's attachment, base64",
        run: Run::Daemon(daemon::<browser::ChatAttachment>()),
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "backfill claim",
        reason: "The extension takes the backfill job it holds or the oldest waiting one",
        help: "Take a backfill job for an extension instance",
        run: Run::Daemon(daemon::<browser::BackfillClaim>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "backfill report",
        reason: "The extension reports a backfill job's progress",
        help: "Report a held backfill job's progress or outcome",
        run: Run::Daemon(daemon::<browser::BackfillReport>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
    Command {
        name: "conversations",
        reason: "The extension asks what the archive holds of the conversations it lists",
        help: "What the archive holds of some of a web chat's conversations",
        run: Run::Daemon(daemon::<browser::Conversations>()),
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "stored-files",
        reason: "The extension skips downloading files whose bytes the archive holds",
        help: "The files of a web chat's conversation whose bytes the archive holds",
        run: Run::Daemon(daemon::<browser::StoredFiles>()),
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "selection-note",
        reason: "A selection in a chat web page becomes the operator's note on its message",
        help: "Save a browser selection as a note on its message",
        run: Run::Daemon(daemon::<H::SelectionNote>()),
        cli: Cli::Hidden,
        writes: true,
        ..CLI
    },
];

/// The command called `name` (`assert-query`).
pub fn command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// `chat send`: the command line reads the `--attach` files in the
/// caller's process and sends their bytes.
fn attach_files(
    parsed: &command_surface::ParsedCommand,
    cwd: &std::path::Path,
    arguments: &mut Map<String, Value>,
) -> Result<(), String> {
    let mut files = Vec::new();
    for path in calling_process::options::<calling_process::ChatSend>(parsed)?.attach {
        let path = cwd.join(path);
        let bytes =
            std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| format!("{} names no file", path.display()))?;
        files.push(crate::browser::actions::ChatFile {
            media_type: media_type(&name).to_owned(),
            name,
            bytes,
        });
    }
    if !files.is_empty() {
        arguments.insert(
            "files".into(),
            serde_json::to_value(files).map_err(|e| e.to_string())?,
        );
    }
    Ok(())
}

/// The media type a file's extension names; the provider checks the bytes.
fn media_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match extension.as_deref() {
        Some("txt" | "md") => "text/plain",
        Some("json") => "application/json",
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("csv") => "text/csv",
        Some("html") => "text/html",
        _ => "application/octet-stream",
    }
}

pub const PRODUCT: &str = "polylogue";

/// The default read deadline of each latency class.
pub fn deadlines() -> command_surface::Deadlines {
    let mut deadlines = command_surface::Deadlines::default();
    for (class, milliseconds) in [
        (command_surface::LatencyClass::Status, 80),
        (command_surface::LatencyClass::Read, 250),
        (command_surface::LatencyClass::Search, 2000),
        (command_surface::LatencyClass::Aggregate, 400),
    ] {
        deadlines.set(class, Some(std::time::Duration::from_millis(milliseconds)));
    }
    deadlines
}

/// One row as the shared declarations describe it, its arguments read
/// from its request type `A` and its calling-process options.
pub(crate) fn specification<A: command_surface::JsonSchema>(
    command: &Command,
) -> command_surface::OperationSpec {
    let mut arguments = arguments::from_schema(&schemars::schema_for!(A), false);
    if let Some(calling_process) = command.calling_process {
        arguments.extend(arguments::from_schema(&calling_process(), true));
    }
    command_surface::OperationSpec {
        name: command.name.into(),
        command: command.cli_words(),
        summary: command.help.into(),
        reason: command.reason.into(),
        kind: command_surface::OperationKind::Request {
            tool: command.tool.map(|tool| {
                tool.name()
                    .strip_prefix("polylogue_")
                    .expect("product tool prefix")
                    .into()
            }),
        },
        http_route: command
            .http
            .then(|| format!("/api/v1{}", command.http_path())),
        latency: command.latency,
        failures: command.failures.to_vec(),
        access: if command.writes {
            command_surface::Access::Write
        } else {
            command_surface::Access::Read
        },
        arguments,
        listed: command.cli != Cli::Hidden,
        default_budget_tokens: None,
    }
}

/// The operation declarations: schemas, help, MCP tools and rendering,
/// without handlers. Clients use them without a daemon.
pub fn declarations() -> &'static std::sync::Arc<command_surface::OperationDeclarations> {
    static DECLARATIONS: LazyLock<std::sync::Arc<command_surface::OperationDeclarations>> =
        LazyLock::new(|| {
            let mut declarations =
                command_surface::OperationDeclarations::builder(PRODUCT, deadlines());
            declarations.accept(command_surface::RequestOption::Cursor);
            declarations.accept(command_surface::RequestOption::BudgetTokens);
            for code in polylogue_model::Code::ALL {
                declarations
                    .register_failure_code(code.as_str(), crate::protocol::kind(*code))
                    .expect("failure classification is unique");
            }
            for tool in Tool::ALL {
                if COMMANDS
                    .iter()
                    .any(|command| command.tool == Some(tool) && command.registration().is_some())
                {
                    declarations
                        .declare_tool(command_surface::ToolFamily {
                            name: tool
                                .name()
                                .strip_prefix("polylogue_")
                                .expect("product tool prefix")
                                .into(),
                            description: tool.description().into(),
                            open_world: tool.open_world(),
                            idempotent: tool.idempotent(),
                        })
                        .expect("declared tool is unique");
                }
            }
            for command in COMMANDS.iter() {
                match command.run {
                    Run::Local(_, declare, _) | Run::ConfiguredLocal(_, declare, _) => {
                        declare(&mut declarations, command)
                    }
                    Run::Feed { registration, .. } => {
                        registration.declare(&mut declarations, command)
                    }
                    _ => command
                        .registration()
                        .expect("request registration")
                        .declare(&mut declarations, command),
                }
            }
            declarations.finish()
        });
    &DECLARATIONS
}

/// Every daemon row's handler bound to its declaration.
pub fn table(
    daemon: std::sync::Arc<crate::daemon::service::Daemon>,
) -> command_surface::OperationTable {
    let mut table = command_surface::OperationTable::new(std::sync::Arc::clone(declarations()));
    for command in COMMANDS.iter() {
        if let Some(registration) = command.registration() {
            registration.bind(std::sync::Arc::clone(&daemon), &mut table, command);
        }
    }
    table
}

#[cfg(test)]
mod tests;
