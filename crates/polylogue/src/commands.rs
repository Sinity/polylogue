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
use crate::daemon::handler::{Row, daemon};
use crate::daemon::service::handlers as H;
use crate::daemon::service::{acts, browser, feeds, line};
use serde_json::Value;
use std::process::ExitCode;
use std::sync::LazyLock;

/// An operation's home tool. Larger tools also accept smaller tools' operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Read,
    Write,
    Act,
}

impl Tool {
    pub const ALL: [Tool; 3] = [Tool::Read, Tool::Write, Tool::Act];

    /// The tool called `name`.
    pub fn named(name: &str) -> Option<Tool> {
        Self::ALL.into_iter().find(|tool| tool.name() == name)
    }

    /// The tool whose access is exactly `access`.
    pub fn with_access(access: ToolAccess) -> Tool {
        match access {
            ToolAccess::Read => Tool::Read,
            ToolAccess::Write => Tool::Write,
            ToolAccess::Act => Tool::Act,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Read => "polylogue_read",
            Tool::Write => "polylogue_write",
            Tool::Act => "polylogue_act",
        }
    }

    /// The name without the `polylogue_` prefix.
    pub fn family(self) -> &'static str {
        self.access().as_str()
    }

    pub fn access(self) -> ToolAccess {
        match self {
            Self::Read => ToolAccess::Read,
            Self::Write => ToolAccess::Write,
            Self::Act => ToolAccess::Act,
        }
    }

    /// The tool's description: its fixed sentences around the lines
    /// generated from the declarations: the signature of each verb this
    /// tool adds, with the first sentence of the verb's help, and of each
    /// operation whose home it is.
    pub fn description(self, steps: &command_surface::StepDeclarations) -> String {
        let (opening, closing, example) = match self {
            Tool::Read => (
                "Read the archive. A call is {operation, arguments}; operation defaults to rows.\n\
                 selection is one query or reference: \"repo:polylogue since:7d\", \"rate limit\", \"tool_calls where outcome:error | group by tool | count\", session:codex:019d..., a chat URL, latest, self, query:NAME, result:ID. No steps lists the rows; each step acts on the same rows, in order. Steps:",
                "explain: true says how a call is read and what it would do, running nothing; dry_run: true resolves the rows and says what each step would do.",
                r#"{"arguments":{"selection":"tool_calls where outcome:error since:1d","steps":[{"verb":"read","around":2}]}}"#,
            ),
            Tool::Write => (
                "Everything polylogue_read does, plus annotations: notes, tags, judgments and saved queries, recorded with you as author. Steps add:",
                "A query matching several rows needs all: true. Each write that changes the archive returns the call that undoes it.",
                r#"{"arguments":{"selection":"self","steps":[{"verb":"note","text":"Catalog stays SQLite","as":"decision"}]}}"#,
            ),
            Tool::Act => (
                "Everything polylogue_write does, plus acts that reach live sessions, web chats and paid providers. Each act is recorded; a sent message cannot be taken back. Post into web chats only on the operator's word. Steps add:",
                "",
                r#"{"arguments":{"selection":"session:codex:019d0f2a","steps":[{"verb":"send","text":"rebase on master first"}]}}"#,
            ),
        };
        let mut text = format!("{opening}\n");
        let offered = crate::daemon::verbs::offered_over_mcp();
        for verb in steps.verbs() {
            if offered.contains(&(verb.verb(), self.access())) {
                text.push_str(&format!("  {} {}\n", verb.signature(), verb_sentence(verb)));
            }
        }
        let mut operations: Vec<String> = COMMANDS
            .iter()
            .filter(|command| command.tool == Some(self) && command.operation().is_some())
            .map(|command| signature(command, None))
            .collect();
        if self == Tool::Read
            && let Some(set) = command("set")
        {
            operations.push(signature(set, Some(SETTING_READ)));
        }
        text.push_str(&format!("Operations: {}\n", operations.join(" | ")));
        if !closing.is_empty() {
            text.push_str(&format!("{closing}\n"));
        }
        text.push_str(&format!("Example: {example}"));
        text
    }

    /// Writes with the same request ID are applied once.
    pub fn idempotent(self) -> bool {
        !matches!(self, Tool::Act)
    }

    /// Its operations act outside the archive (a web chat, a harness).
    pub fn open_world(self) -> bool {
        matches!(self, Tool::Act)
    }
}

/// The arguments of `set` that list settings or show one, which the read
/// tool offers; writing a value is the write tool's.
pub const SETTING_READ: &[&str] = &["key", "machine"];

/// `name(argument, optional?)`: the arguments of a row's request type that
/// the tool list offers, or only `only` of them.
fn signature(command: &Command, only: Option<&[&str]>) -> String {
    let Some(row) = command.operation() else {
        return format!("{}()", command.name);
    };
    let schema = row.arguments();
    let required = schema.as_value()["required"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let arguments: Vec<String> = arguments::from_schema(&schema, false)
        .into_iter()
        .filter(|argument| argument.listed)
        .filter(|argument| only.is_none_or(|only| only.contains(&argument.name.as_str())))
        .map(|argument| {
            let optional = !required.contains(&Value::from(argument.name.as_str()));
            format!("{}{}", argument.name, if optional { "?" } else { "" })
        })
        .collect();
    format!("{}({})", command.name, arguments.join(", "))
}

/// The first sentence of a verb's help: its step type's doc comment after
/// the `` `verb WORDS`: `` that names it.
pub(crate) fn verb_sentence(verb: &command_surface::StepDeclaration) -> String {
    let help = verb.input_schema()["description"]
        .as_str()
        .unwrap_or_default();
    let help = help.split_once("`: ").map_or(help, |(_, rest)| rest);
    let help = help.split("\n\n").next().unwrap_or(help).replace('\n', " ");
    match help.find(". ") {
        Some(end) => help[..=end].to_owned(),
        None => help,
    }
}

polylogue_model::stored_names! {
    /// The smallest MCP tool that may run a request: one that reads, one
    /// that also writes the archive, or one that also acts outside it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum ToolAccess {
        Read => "read",
        Write => "write",
        Act => "act",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cli {
    Subcommand,
    /// A subcommand left out of help: programs run it, people do not.
    Hidden,
    /// A selection and the verbs after it, read by the line parser
    /// (`cli/line.rs`); it is not a subcommand.
    Line,
}
/// The first word of an actual command, from the same table as CLI parsing.
pub(crate) fn is_cli_command(word: &str) -> bool {
    COMMANDS.iter().any(|row| {
        matches!(row.cli, Cli::Subcommand | Cli::Hidden) && row.name.split(' ').next() == Some(word)
    })
}

impl Cli {
    /// Left out of help and completion.
    pub fn hidden(self) -> bool {
        matches!(self, Cli::Hidden | Cli::Line)
    }
}

pub type OutputHandler = fn(
    &command_surface::ParsedCommand,
    &command_surface::Answer<Value>,
    &crate::config::Client,
    bool,
) -> ExitCode;

/// What the command line does in place of one request and its answer: it
/// writes a file or a directory, or follows a stream.
pub type Deliver = fn(
    &command_surface::ParsedCommand,
    &crate::config::Client,
    &command_surface::RunnableRequest,
) -> ExitCode;

/// What runs a command.
#[derive(Clone, Copy)]
pub enum Run {
    /// The daemon answers through the row's operation.
    Daemon(&'static dyn Row),
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
    /// Runs in the calling process with the whole deployment: the daemon,
    /// and the commands that run the engine or read the machine identity.
    ConfiguredLocal(
        fn(&Command, &command_surface::ParsedCommand, &crate::config::Startup, bool) -> ExitCode,
        fn(&mut command_surface::DeclarationsBuilder, &'static Command),
        fn(&Value) -> Result<(), String>,
    ),
    /// Runs in the calling process with the client's configuration alone.
    ClientLocal(
        fn(&Command, &command_surface::ParsedCommand, &crate::config::Client, bool) -> ExitCode,
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
const fn client_local<A: command_surface::JsonSchema + serde::de::DeserializeOwned>(
    run: fn(&Command, &command_surface::ParsedCommand, &crate::config::Client, bool) -> ExitCode,
) -> Run {
    Run::ClientLocal(run, declare_local::<A>, validate_local::<A>)
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
    /// The command-line words (`dev sql`), also the operation's name
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
    /// What the command line does with the answer in place of printing it
    /// (`dev schema` writes its result where `--to` names, `status` prints
    /// its own view).
    pub output: Option<OutputHandler>,
    /// What the command line does in place of one request and its printed
    /// answer: `dev feed` follows its events.
    pub deliver: Option<Deliver>,
}

impl Command {
    /// The operation of a row the daemon answers.
    pub(crate) fn operation(&self) -> Option<&'static dyn Row> {
        match self.run {
            Run::Daemon(operation) => Some(operation),
            Run::Local(..) | Run::ConfiguredLocal(..) | Run::ClientLocal(..) => None,
        }
    }

    pub fn cli_words(&self) -> Vec<String> {
        self.name.split(' ').map(str::to_owned).collect()
    }

    pub fn http_path(&self) -> String {
        format!("/{}", self.name.replace(' ', "/"))
    }
}

/// A word that groups commands on the command line (`dev`, `measure`).
pub struct Group {
    pub name: &'static str,
    pub help: &'static str,
}

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
    run: local_command::<local::arguments::Empty>(local::no_action),
    cli: Cli::Subcommand,
    tool: None,
    http: false,
    writes: false,
    output: None,
    deliver: None,
};

pub const GROUPS: &[Group] = &[
    Group {
        name: "dev",
        help: "Developer queries, schema descriptions and change feed",
    },
    Group {
        name: "dev schema",
        help: "Structure, counts and unread paths from stored revisions",
    },
];

/// Every command, in help order.
pub static COMMANDS: &[Command] = &[
    Command {
        name: "help",
        examples: &["polylogue help units"],
        help: "Read a topic or verb's help, or the full reference with --all",
        reason: "Describe the command and query language",
        run: Run::Daemon(daemon::<H::Help>()),
        tool: Some(Tool::Read),
        http: true,
        ..CLI
    },
    Command {
        name: "embed",
        examples: &[
            "polylogue embed",
            "polylogue embed --query 'sessions where physical:true | limit 5'",
        ],
        reason: "Preview embedding work and build it only after confirmation",
        help: "Preview selected texts and cost; --query selects sessions (including | limit N), --yes builds their embedding vectors",
        run: Run::Daemon(daemon::<H::Embed>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "install",
        help: "Install harness hooks, Chrome or shell integration, or authorize Google Drive; --dry-run previews the installation",
        run: configured_local::<local::arguments::Install>(local::install),
        examples: &["polylogue install zsh --dry-run"],
        ..CLI
    },
    Command {
        name: "rows",
        reason: "Acts on selected rows with verbs, in order, on every surface",
        latency: command_surface::LatencyClass::Search,
        help: "A selection and the verbs that act on its rows, in order: the line every surface carries",
        run: Run::Daemon(daemon::<line::Line>()),
        cli: Cli::Line,
        tool: Some(Tool::Read),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "dev schema infer",
        examples: &["polylogue dev schema infer codex"],
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
        name: "dev schema unread",
        examples: &["polylogue dev schema unread codex"],
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
        name: "dev schema diff",
        examples: &["polylogue dev schema diff /home/op/before.json /home/op/after.json"],
        help: "Compare explicit inferred schema files recursively, with sample differences labeled",
        calling_process: Some(calling_process::schema::<calling_process::SchemaOutput>),
        run: local_command::<local::arguments::SchemaDiff>(crate::cli::schema::diff),
        ..CLI
    },
    Command {
        name: "demo",
        examples: &["polylogue demo /home/op/morrow-demo"],
        help: "Prepare a separate Morrow archive and walk through search, tool outcomes, parents, costs and notes",
        run: configured_local::<local::arguments::Tour>(crate::demo::tour),
        ..CLI
    },
    Command {
        name: "status",
        examples: &[
            "polylogue status",
            "polylogue status --vectors",
            "polylogue status --verify --manifest /home/op/archive.tsv",
        ],
        help: "What the archive holds and what the daemon is doing; --vectors counts stored vectors, --verify checks every stored byte at idle priority (CLI only)",
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
        examples: &[
            "polylogue import /home/op/morrow/export.json --origin chatgpt",
            "polylogue import chatgpt --since 2026-01-01",
        ],
        help: "Store explicit files, directories and bundles through the daemon. The typed report lists imported and already stored paths, classifications and refusals, packages and authored restoration counts, and for a Python Polylogue vector store what became of each vector. Primary authored restores awaiting confirmation are pending. One web chat's name (chatgpt, claude-ai, grok) has the browser extension import its history instead: one job per provider, --since, --cancel, --wait; a file named like one is ./chatgpt",
        reason: "Capture explicitly selected source bytes and restore the notes and other user records a bundle carries",
        latency: command_surface::LatencyClass::Aggregate,
        run: Run::Daemon(daemon::<H::Import>()),
        tool: Some(Tool::Write),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "internal advisory",
        reason: "Offer one operator-judged file pointer without delaying the prompt",
        latency: command_surface::LatencyClass::Read,
        help: "One current operator-judged decision or correction for a file named in a prompt; silent when unavailable within 20 ms or on cooldown",
        run: Run::Daemon(daemon::<H::Advisory>()),
        tool: None,
        http: true,
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "set",
        examples: &["polylogue set browser.actions"],
        help: "List the typed runtime settings catalog, read KEY, or write KEY VALUE. Includes reading.alternates, browser.actions, embed.* and plan.*. Standalone attention settings are local; in primary mode Sinex holds attention settings",
        reason: "Inspect declared runtime settings and author changes at their declared scope",
        latency: command_surface::LatencyClass::Local,
        run: Run::Daemon(daemon::<H::Setting>()),
        tool: Some(Tool::Write),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "dev feed",
        reason: "Give a consumer every change since its last applied position, once",
        examples: &["polylogue dev feed"],
        help: "A finite JSON-lines change feed; persist its new position only after applying the complete batch",
        run: Run::Daemon(daemon::<feeds::Changes>()),
        deliver: Some(crate::cli::changes::run),
        http: true,
        ..CLI
    },
    Command {
        name: "start",
        examples: &[
            "polylogue start 'Summarize the current work' --in codex --cwd /home/op/morrow",
            "polylogue start 'Summarize the attached notes' --in chatgpt --attach /home/op/morrow/notes.md",
        ],
        help: "Start a session Polylogue hosts on the harness's control protocol (Codex's app-server, Claude Code stream-json, Gemini CLI and Hermes ACP): its approvals, questions and turns can then be answered, steered and interrupted from anywhere, and it survives a daemon restart. With --role, an instance of that role: its harness, with its instructions. With --in chatgpt, the browser extension drafts a new web conversation (--model, --effort, --project, --attach); --submit sends it, held for the operator's answer when anyone else asks",
        reason: "Starts a harness session; no query selects one that does not exist yet",
        run: Run::Daemon(daemon::<acts::StartSession>()),
        tool: Some(Tool::Act),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "dev sql",
        examples: &["polylogue dev sql 'SELECT id FROM v1_sessions'"],
        help: "Read-only SQL. Programs read the views v1_sessions, v1_messages, v1_tool_calls, v1_links and v1_annotations, whose names and columns stay; the tables change with the index format",
        reason: "Run expert read-only queries over the stable archive views",
        latency: command_surface::LatencyClass::Aggregate,
        run: Run::Daemon(daemon::<H::Sql>()),
        http: true,
        tool: Some(Tool::Read),
        ..CLI
    },
    Command {
        name: "web",
        help: "Open the web reader",
        run: client_local::<local::arguments::Empty>(local::web),
        ..CLI
    },
    Command {
        name: "internal browser-host",
        help: "The browser extension's native messaging host; Chrome starts it",
        run: configured_local::<local::arguments::Empty>(local::browser_host),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "mcp",
        help: "Serve MCP over stdio",
        run: client_local::<local::arguments::Empty>(local::mcp),
        ..CLI
    },
    Command {
        name: "internal hook",
        help: "Called by a harness hook; reads the event from stdin and answers with context when the event takes it: the brief at session start, the brief of a session whose ID or supported web link is pasted",
        run: configured_local::<local::arguments::Hook>(local::hook),
        tool: None,
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "internal host",
        help: "Started by the daemon for a session it hosts over stdio (Claude Code stream-json, Gemini CLI and Hermes ACP): holds the agent's stdin and stdout so a daemon restart never ends it, and relays them to the daemon's connection",
        run: local_command::<local::arguments::Host>(local::host),
        tool: None,
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "internal complete",
        reason: "Complete an unfinished query at its cursor position",
        latency: command_surface::LatencyClass::Completion,
        help: "Complete a query at a UTF-8 byte offset using its full scope and suffix",
        run: Run::Daemon(daemon::<H::Complete>()),
        cli: Cli::Hidden,
        http: true,
        tool: None,
        ..CLI
    },
    // Rows the hooks, the browser extension's host and the operator's own
    // pages ask; no person types them.
    Command {
        name: "internal delivery-event",
        reason: "A hook reports what happened to a delivery it was handed",
        help: "Record what happened to a delivery after it was handed out",
        run: Run::Daemon(daemon::<H::DeliveryEvent>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal mail-claim",
        reason: "A session's hook takes the messages and notices waiting for it",
        help: "Take the messages and notices pending for a session",
        run: Run::Daemon(daemon::<acts::MailClaim>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal permission",
        reason: "The synchronous permission hook asks for a decision given ahead",
        help: "A decision given ahead for a session's pending tool request",
        run: Run::Daemon(daemon::<acts::StandingPermission>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal chat claim",
        reason: "The extension takes the oldest queued chat action",
        help: "Take the oldest queued chat action for an extension instance",
        run: Run::Daemon(daemon::<browser::ChatClaim>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal chat report",
        reason: "The extension reports the progress or outcome of a chat action",
        help: "Report a held chat action's progress or outcome",
        run: Run::Daemon(daemon::<browser::ChatReport>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal chat answer",
        reason: "The extension's popup gives the operator's answer to a held turn",
        help: "Allow or deny a held chat action, as `answer allow|deny` does",
        run: Run::Daemon(daemon::<browser::HeldTurnAnswer>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal chat attachment",
        reason: "The extension fetches the bytes of a file a chat action attaches",
        help: "The bytes of a chat action's attachment, base64",
        run: Run::Daemon(daemon::<browser::ChatAttachment>()),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal import claim",
        reason: "The extension takes the import job it holds or the oldest waiting one",
        help: "Take an import job for an extension instance",
        run: Run::Daemon(daemon::<browser::ImportJobClaim>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal import report",
        reason: "The extension reports an import job's progress",
        help: "Report a held import job's progress or outcome",
        run: Run::Daemon(daemon::<browser::ImportJobReport>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal conversations",
        reason: "The extension asks what the archive holds of the conversations it lists",
        help: "What the archive holds of some of a web chat's conversations",
        run: Run::Daemon(daemon::<browser::Conversations>()),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal reader-url",
        reason: "The extension opens the running reader",
        help: "The running reader's authenticated address",
        run: Run::Daemon(daemon::<browser::ReaderUrl>()),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal capture-files",
        reason: "The extension reads attachment names with the capture parser",
        help: "Read the conversation and attachment names in a capture",
        run: Run::Daemon(daemon::<browser::CaptureFiles>()),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal stored-files",
        reason: "The extension skips downloading files whose bytes the archive holds",
        help: "The files of a web chat's conversation whose bytes the archive holds",
        run: Run::Daemon(daemon::<browser::StoredFiles>()),
        cli: Cli::Hidden,
        tool: None,
        ..CLI
    },
    Command {
        name: "internal selection-note",
        reason: "A selection in a chat web page becomes the operator's note on its message",
        help: "Save a browser selection as a note on its message",
        run: Run::Daemon(daemon::<H::SelectionNote>()),
        cli: Cli::Hidden,
        writes: true,
        tool: None,
        ..CLI
    },
];

/// The command called `name` (`dev sql`).
pub fn command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
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
            tool: command.tool.map(|tool| tool.family().into()),
        },
        http_route: command
            .http
            .then(|| format!("/api/v1{}", command.http_path())),
        latency: command.latency,
        failures: command.failures.to_vec(),
        access: if command
            .tool
            .is_some_and(|tool| tool.access() == ToolAccess::Read)
        {
            command_surface::Access::Read
        } else if command.writes {
            command_surface::Access::Write
        } else {
            command_surface::Access::Read
        },
        arguments,
        listed: !command.cli.hidden(),
        default_budget_tokens: None,
    }
}

/// The operation declarations: schemas, help, MCP tools and rendering,
/// without handlers. Clients use them without a daemon.
pub fn declarations() -> &'static std::sync::Arc<command_surface::OperationDeclarations> {
    static DECLARATIONS: LazyLock<std::sync::Arc<command_surface::OperationDeclarations>> =
        LazyLock::new(|| {
            let steps = crate::daemon::verbs::declarations();
            let mut declarations =
                command_surface::OperationDeclarations::builder(PRODUCT, deadlines());
            declarations.accept(command_surface::RequestOption::Cursor);
            declarations.accept(command_surface::RequestOption::BudgetTokens);
            declarations.accept(command_surface::RequestOption::Format);
            for code in polylogue_model::Code::ALL {
                declarations
                    .register_failure_code(code.as_str(), crate::protocol::kind(*code))
                    .expect("failure classification is unique");
            }
            for tool in Tool::ALL {
                if COMMANDS
                    .iter()
                    .any(|command| command.tool == Some(tool) && command.operation().is_some())
                {
                    declarations
                        .declare_tool(command_surface::ToolFamily {
                            name: tool.family().into(),
                            description: tool.description(&steps),
                            open_world: tool.open_world(),
                            idempotent: tool.idempotent(),
                        })
                        .expect("declared tool is unique");
                }
            }
            for command in COMMANDS.iter() {
                match command.run {
                    Run::Local(_, declare, _)
                    | Run::ConfiguredLocal(_, declare, _)
                    | Run::ClientLocal(_, declare, _) => declare(&mut declarations, command),
                    Run::Daemon(operation) => operation.declare(&mut declarations, command),
                }
                if command.cli == Cli::Line {
                    declarations
                        .declare_steps(command.name, "steps", steps.clone())
                        .expect("the line's steps are declared once");
                }
            }
            declarations
                .set_tool_default("read", "rows")
                .expect("rows belongs to the read tool");
            declarations.finish()
        });
    &DECLARATIONS
}

#[cfg(test)]
mod tests;
