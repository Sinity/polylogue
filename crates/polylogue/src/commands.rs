//! The command table: every operation once, with its help, its arguments
//! and the surfaces that offer it. The CLI and its completions, the MCP
//! tools and the HTTP API under `/api/v1/` are built from it, and its
//! command names are the words a query may not start with.
//!
//! An argument has one name everywhere: `--name` on the command line (with
//! `-` for `_`), the property of an MCP tool's input, the field of an HTTP
//! request. Values from every surface become one JSON object ([`Values`]),
//! checked against the table, and a command's function turns that object
//! into the daemon request.

mod requests;

use crate::cli::{local, print};
use crate::daemon::Request;
use crate::daemon::measure::{MEASURES, Measure};
use polylogue_model::SessionId;
use serde_json::{Map, Value};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::LazyLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    Integer,
    Bool,
    /// Words given by position, joined by spaces.
    Words,
    /// Words given by position that take the rest of the command line,
    /// flags included.
    Rest,
    /// Several values: by position, or a repeated or comma-separated option.
    List,
    /// One of these values.
    Choice(&'static [&'static str]),
    /// A flag that may carry a value (`--agent` or `--agent SESSION`):
    /// `true` or the text.
    FlagOrText,
}

#[derive(Clone, Copy, Debug)]
pub struct Arg {
    pub name: &'static str,
    pub kind: Kind,
    pub help: &'static str,
    pub required: bool,
    /// Given by position on the command line rather than as `--name`.
    pub positional: bool,
    pub default: Option<&'static str>,
    pub short: Option<char>,
    /// Taken before or after any command on the command line.
    pub global: bool,
    /// Acts on the calling machine (`--out`, `--exec`): command line only.
    pub cli_only: bool,
}

const fn arg(name: &'static str, kind: Kind, help: &'static str) -> Arg {
    Arg {
        name,
        kind,
        help,
        required: false,
        positional: false,
        default: None,
        short: None,
        global: false,
        cli_only: false,
    }
}

/// A required positional argument.
const fn pos(name: &'static str, kind: Kind, help: &'static str) -> Arg {
    Arg {
        required: true,
        positional: true,
        ..arg(name, kind, help)
    }
}

/// A positional argument that may be left out.
const fn pos_opt(name: &'static str, kind: Kind, help: &'static str) -> Arg {
    Arg {
        positional: true,
        ..arg(name, kind, help)
    }
}

/// A flag that may be left out.
const fn opt(name: &'static str, kind: Kind, help: &'static str) -> Arg {
    arg(name, kind, help)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cli {
    /// Not offered on the command line.
    Absent,
    Subcommand,
    /// A subcommand left out of help: programs run it, people do not.
    Hidden,
    /// What `polylogue` runs when no subcommand is given.
    NoSubcommand,
}

#[derive(Clone, Copy)]
pub enum Action {
    /// Sent to the daemon as the request this builds.
    Request(fn(&Command, &Values, &Caller) -> Result<Request, String>),
    /// An HTTP-only structured body, decoded by its typed request owner.
    JsonRequest(fn(Value) -> Result<Request, String>),
    /// An HTTP-only binary body, passed directly to the daemon in process.
    BytesRequest(fn(Vec<u8>) -> Request),
    /// Runs in the calling process (the CLI only), with `--json`.
    Local(fn(&Command, &Values, bool) -> ExitCode),
}

#[derive(Clone, Copy)]
pub struct Command {
    /// The command-line words (`note add`); the MCP tool joins them with
    /// `_`, the HTTP route with `/`.
    pub name: &'static str,
    pub help: &'static str,
    pub args: &'static [Arg],
    pub action: Action,
    pub cli: Cli,
    pub mcp: bool,
    pub http: bool,
    /// Changes the archive: POST over HTTP, where the origin is checked,
    /// and MCP and HTTP callers may pass a `request_id` so a resent call
    /// applies once.
    pub writes: bool,
    /// The answer as text for a person on the command line; `--json`
    /// prints the answer itself.
    pub print: fn(&Value) -> String,
    /// What the command line does with the answer in place of printing it
    /// (`export` writes a file, `continue --exec` runs the plan).
    pub output: Option<fn(&Values, &Value, bool) -> ExitCode>,
}

impl Command {
    pub fn mcp_name(&self) -> String {
        self.name.replace(' ', "_")
    }

    pub fn http_path(&self) -> String {
        format!("/{}", self.name.replace(' ', "/"))
    }

    /// The arguments a surface offers: the command line has them all, MCP
    /// and HTTP those that do not act on the calling machine.
    pub fn args_for(&self, surface: Surface) -> impl Iterator<Item = &'static Arg> {
        let args: &'static [Arg] = self.args;
        args.iter()
            .filter(move |a| surface == Surface::Cli || !a.cli_only)
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
}

/// Who is calling, for what a command's arguments leave open.
#[derive(Clone, Debug)]
pub struct Caller {
    pub surface: Surface,
    /// The caller's working directory, which `cwd` defaults to; HTTP
    /// callers have none and must name one.
    pub cwd: Option<PathBuf>,
    /// Wants JSON: `--json` on the command line, always over MCP and HTTP.
    pub json: bool,
}

impl Caller {
    /// The sender a prepared context is recorded under: MCP hands it to a
    /// model; the command line and the reader show it to a person.
    pub fn sender(&self) -> Option<String> {
        (self.surface == Surface::Mcp).then(|| "mcp".to_owned())
    }
}

/// The live view as a query: what `polylogue live` runs.
pub const LIVE: &str = "sessions where -live:ended | sort by live_rank, last_event_at desc";
/// Signal notices as a query: what `polylogue notices` runs.
pub const NOTICES: &str = "deliveries where sender:signal | sort by at desc";

const WHY: Arg = Arg {
    global: true,
    ..opt(
        "why",
        Kind::Bool,
        "Explain an empty result: how many rows each condition keeps",
    )
};
const CURSOR: Arg = Arg {
    global: true,
    ..opt(
        "cursor",
        Kind::Text,
        "Continue from a previous page's `next`",
    )
};
const FIELDS: Arg = Arg {
    global: true,
    ..opt(
        "fields",
        Kind::List,
        "The columns to show instead of the unit's defaults, comma-separated (`--fields id,cost,repo`)",
    )
};
const LIMIT: Arg = opt("limit", Kind::Integer, "Rows per page");
const QUERY_WORDS: Arg = pos_opt("query", Kind::Words, "A query in the query language");
const CWD: Arg = opt(
    "cwd",
    Kind::Text,
    "The directory (default: the caller's working directory)",
);
const FORMATS: &[&str] = polylogue_render::Format::NAMES;
const VIEWS: &[&str] = polylogue_render::View::NAMES;
const SHELLS: &[&str] = &["bash", "elvish", "fish", "powershell", "zsh"];

/// A query over a unit with the page, cursor, explanation and columns.
const QUERY_ARGS: &[Arg] = &[QUERY_WORDS, LIMIT, WHY, CURSOR, FIELDS];
/// A measurement over the sessions a query selects.
const MEASURE_ARGS: &[Arg] = &[
    pos_opt(
        "query",
        Kind::Words,
        "A sessions query (default: every session)",
    ),
    WHY,
    CURSOR,
    FIELDS,
];
const MEASURE_ALL: &[Arg] = &[WHY, CURSOR, FIELDS];
const EMBED_ARGS: &[Arg] = &[
    pos_opt(
        "query",
        Kind::Words,
        "A sessions query limiting the texts (default: every prose text)",
    ),
    opt("limit", Kind::Integer, "At most this many texts"),
    opt("yes", Kind::Bool, "Confirm a build with a paid provider"),
];

/// A command on the command line only, sent to the daemon unless its
/// action is local.
const CLI: Command = Command {
    name: "",
    help: "",
    args: &[],
    action: Action::Local(local::not_built),
    cli: Cli::Subcommand,
    mcp: false,
    http: false,
    writes: false,
    print: print::json,
    output: None,
};

pub const GROUPS: &[Group] = &[
    Group {
        name: "note",
        help: "Notes and tags on sessions, messages and tool calls",
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
        name: "sinex",
        help: "The Sinex backend: status and the one-time move to primary",
    },
    Group {
        name: "hooks",
        help: "Add or remove Polylogue's hooks in Claude Code's settings",
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

/// `measure NAME`: a sessions query (none for a whole-archive measure),
/// and the series for one that takes it.
fn measure_command(m: &'static Measure) -> Command {
    let mut args: Vec<Arg> = Vec::new();
    if !m.metrics.is_empty() {
        args.push(pos("metric", Kind::Choice(m.metrics), "The series"));
    }
    args.extend_from_slice(if m.whole_archive {
        MEASURE_ALL
    } else {
        MEASURE_ARGS
    });
    Command {
        name: format!("measure {}", m.name).leak(),
        help: m.help,
        args: args.leak(),
        action: Action::Request(requests::measure),
        print: print::query,
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
        name: "query",
        help: "Run a query",
        args: &[
            pos_opt("query", Kind::Rest, "A query, when no subcommand is given"),
            LIMIT,
            WHY,
            CURSOR,
            FIELDS,
        ],
        action: Action::Request(requests::query),
        print: print::query,
        cli: Cli::NoSubcommand,
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "status",
        help: "What the archive holds and what the daemon is doing",
        action: Action::Request(requests::status),
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "daemon",
        help: "Run the daemon in the foreground",
        action: Action::Local(local::daemon),
        ..CLI
    },
    Command {
        name: "import",
        help: "Bring in an export (ChatGPT, claude.ai, Gemini, a Polylogue export)",
        args: &[pos_opt("paths", Kind::List, "Files or directories")],
        action: Action::Request(requests::import),
        writes: true,
        ..CLI
    },
    Command {
        name: "list",
        help: "Sessions matching a query, newest first",
        args: QUERY_ARGS,
        action: Action::Request(requests::list),
        print: print::query,
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "read",
        help: "Read one session",
        args: &[
            pos(
                "session",
                Kind::Text,
                "`origin:native_id`, as `list` prints it",
            ),
            Arg {
                default: Some("full"),
                ..opt("view", Kind::Choice(VIEWS), "What to show")
            },
            opt(
                "format",
                Kind::Choice(FORMATS),
                "Default: JSON messages for programs (`--json`, MCP, HTTP), else terminal text",
            ),
            opt(
                "budget",
                Kind::Integer,
                "Pages sized for a model, at most this many tokens each",
            ),
            CURSOR,
            opt(
                "logical",
                Kind::Bool,
                "Include messages inherited from a parent session",
            ),
        ],
        action: Action::Request(requests::read),
        print: print::read,
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "export",
        help: "Write sessions out as files",
        args: &[
            pos_opt("sessions", Kind::List, "The sessions to write"),
            Arg {
                default: Some("json"),
                ..opt("format", Kind::Choice(FORMATS), "The file format")
            },
            Arg {
                cli_only: true,
                ..opt(
                    "out",
                    Kind::Text,
                    "The file to write (default: standard output)",
                )
            },
        ],
        action: Action::Request(requests::export),
        output: Some(crate::cli::write_export),
        ..CLI
    },
    Command {
        name: "note add",
        help: "Add a note to one or more targets: session:<id>, message:<id>, tool:<session>#<call>, repo:<name>, cwd:<path>, work:<n>. Over MCP the author is the calling agent",
        args: &[
            pos("targets", Kind::List, "What the note is about"),
            Arg {
                required: true,
                short: Some('m'),
                ..opt("body", Kind::Text, "The note's text")
            },
            Arg {
                default: Some("note"),
                ..opt(
                    "kind",
                    Kind::Text,
                    "note, lesson, handoff, decision, blocker, goal, question, finding, outcome, correction, judgment, title, claim, problem, forecast",
                )
            },
            opt(
                "revises",
                Kind::Text,
                "Replace an earlier note: the new one joins its revision family",
            ),
            opt(
                "agent",
                Kind::FlagOrText,
                "Written by an agent (by its session when given)",
            ),
        ],
        action: Action::Request(requests::note_add),
        mcp: true,
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "note tag",
        help: "Tag targets: `key` or `key:value`",
        args: &[
            pos("targets", Kind::List, "What to tag"),
            Arg {
                required: true,
                ..opt("tag", Kind::Text, "The tag")
            },
            opt(
                "value",
                Kind::Text,
                "Confidence or affinity, as JSON (`{\"confidence\": 0.8}`)",
            ),
        ],
        action: Action::Request(requests::note_tag),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "note withdraw",
        help: "Withdraw a note and every revision of it",
        args: &[
            pos("note", Kind::Text, "The note's ID"),
            opt("reason", Kind::Text, "Why"),
        ],
        action: Action::Request(requests::note_withdraw),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "note list",
        help: "Notes, current revisions only unless --history; agent-written notes are marked by their author",
        args: &[
            pos_opt(
                "targets",
                Kind::List,
                "References: `session:<id>`, `repo:<name>`, `cwd:<path>`, ...",
            ),
            opt(
                "kind",
                Kind::List,
                "Note kinds: lesson, handoff, decision, ...",
            ),
            opt("agents", Kind::Bool, "Only notes agents wrote"),
            opt("history", Kind::Bool, "Earlier revisions too"),
            opt("withdrawn", Kind::Bool, "Withdrawn notes too"),
            opt(
                "cwd",
                Kind::Text,
                "Notes on this directory and its repository",
            ),
            opt("limit", Kind::Integer, "At most this many notes"),
        ],
        action: Action::Request(requests::note_list),
        print: print::notes,
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "brief",
        help: "The session-start brief for a directory or session: the operator's lessons, handoffs and decisions there, and recent sessions",
        args: &[
            opt(
                "session",
                Kind::Text,
                "The session asking, when known (`claude-code:<id>`)",
            ),
            CWD,
            Arg {
                default: Some("startup"),
                ..opt(
                    "source",
                    Kind::Choice(&["startup", "resume", "clear", "compact"]),
                    "Why the session starts",
                )
            },
        ],
        action: Action::Request(requests::brief),
        print: print::context,
        mcp: true,
        ..CLI
    },
    Command {
        name: "takeover",
        help: "Context for taking over another session: its prompts, notes, what it left open and its last answer",
        args: &[pos("session", Kind::Text, "`origin:native_id`")],
        action: Action::Request(requests::takeover),
        print: print::context,
        mcp: true,
        ..CLI
    },
    Command {
        name: "continue",
        help: "Resume a session through its tool's own command. Without a session, the sessions in this directory that left something open (an unanswered prompt, an unfinished tool call, a todo, a handoff)",
        args: &[
            pos_opt("session", Kind::Text, "`origin:native_id`"),
            Arg {
                cli_only: true,
                ..opt(
                    "exec",
                    Kind::Bool,
                    "Run the native resume command instead of printing it",
                )
            },
            CWD,
            opt("limit", Kind::Integer, "At most this many sessions"),
        ],
        action: Action::Request(requests::resume),
        print: print::resume,
        output: Some(crate::cli::resume_output),
        mcp: true,
        http: true,
        ..CLI
    },
    Command {
        name: "recall",
        help: "Search past work for something relevant now: matching messages grouped by session, notes on them, and a receipt of how it searched",
        args: &[
            pos_opt(
                "query",
                Kind::Words,
                "What to look for; every word must match",
            ),
            CWD,
        ],
        action: Action::Request(requests::recall),
        print: print::context,
        mcp: true,
        ..CLI
    },
    Command {
        name: "setting",
        help: "Read or set a setting (`briefing.session_start off`)",
        args: &[
            pos("key", Kind::Text, "The setting"),
            pos_opt(
                "value",
                Kind::Text,
                "The new value; without it, the current one",
            ),
            opt("machine", Kind::Bool, "Only on this machine"),
        ],
        action: Action::Request(requests::setting),
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "work",
        help: "The work graph: goals, attempts, verifications and artifacts, by a query over the `work` unit (`polylogue work kind:attempt state:open`)",
        args: QUERY_ARGS,
        action: Action::Request(requests::work),
        print: print::query,
        ..CLI
    },
    Command {
        name: "live",
        help: "Sessions running now and what each is doing: the query `sessions where -live:ended | sort by live_rank, last_event_at desc`",
        args: &[WHY, CURSOR, FIELDS],
        action: Action::Request(requests::live),
        print: print::query,
        ..CLI
    },
    Command {
        name: "changes",
        help: "What changed since a cursor: the query `changes where after:CURSOR`",
        args: &[
            opt("since", Kind::Text, "A previous answer's `view.cursor`"),
            WHY,
            CURSOR,
            FIELDS,
        ],
        action: Action::Request(requests::changes),
        print: print::query,
        ..CLI
    },
    Command {
        name: "notices",
        help: "Signal notices and what happened to each: the query `deliveries where sender:signal | sort by at desc`",
        args: &[
            Arg {
                default: Some("50"),
                ..LIMIT
            },
            WHY,
            CURSOR,
            FIELDS,
        ],
        action: Action::Request(requests::notices),
        print: print::query,
        ..CLI
    },
    Command {
        name: "keep",
        help: "Run a query and keep the rows it returned, to read back later as `result:<id>` (or `result:<id>.matches`) exactly as they were",
        args: &[
            opt("name", Kind::Text, "A name for the kept result"),
            Arg {
                required: true,
                ..pos("query", Kind::Rest, "The query")
            },
        ],
        action: Action::Request(requests::keep),
        writes: true,
        ..CLI
    },
    Command {
        name: "sql",
        help: "Read-only SQL over the stable views",
        args: &[pos("statement", Kind::Text, "One SQL statement")],
        action: Action::Request(requests::sql),
        http: true,
        ..CLI
    },
    Command {
        name: "excise",
        help: "Remove content permanently from every file and derived product: `session:<id>`, `input:<id>` or `text:<string>`. Without `--yes`, shows what would go and changes nothing",
        args: &[
            pos("target", Kind::Text, "What to remove"),
            opt("yes", Kind::Bool, "Remove it"),
            opt("reason", Kind::Text, "Why"),
        ],
        action: Action::Request(requests::excise),
        writes: true,
        ..CLI
    },
    Command {
        name: "embed preview",
        help: "What a build would embed and what it would cost",
        args: EMBED_ARGS,
        action: Action::Request(requests::embed),
        ..CLI
    },
    Command {
        name: "embed build",
        help: "Embed the texts not yet embedded. A paid provider needs `--yes`",
        args: EMBED_ARGS,
        action: Action::Request(requests::embed),
        writes: true,
        ..CLI
    },
    Command {
        name: "embed status",
        help: "Vectors per model, and the provider settings",
        action: Action::Request(requests::embed),
        ..CLI
    },
    Command {
        name: "sinex status",
        help: "The Sinex backend's state (not built yet)",
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "sinex cutover",
        help: "Make Sinex the primary store, once (not built yet)",
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "browser-actions",
        help: "Queue one operator chat action while browser.actions is enabled",
        action: Action::JsonRequest(requests::browser_action),
        cli: Cli::Absent,
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "browser-actions attachments",
        help: "Store binary attachment bytes for a later operator chat action",
        action: Action::BytesRequest(requests::browser_attachment),
        cli: Cli::Absent,
        http: true,
        writes: true,
        ..CLI
    },
    Command {
        name: "web",
        help: "Open the web reader",
        action: Action::Local(local::web),
        ..CLI
    },
    Command {
        name: "install-agent",
        help: "Register Polylogue with a browser: `chrome` writes the extension's native messaging host manifest, admitting only `--extension-id`",
        args: &[
            pos("harness", Kind::Text, "The browser"),
            opt(
                "extension_id",
                Kind::Text,
                "The ID chrome://extensions shows",
            ),
            opt("remove", Kind::Bool, "Delete the manifest instead"),
        ],
        action: Action::Local(local::install_agent),
        ..CLI
    },
    Command {
        name: "browser-host",
        help: "The browser extension's native messaging host; Chrome starts it",
        action: Action::Local(local::browser_host),
        cli: Cli::Hidden,
        ..CLI
    },
    Command {
        name: "mcp",
        help: "Serve MCP over stdio",
        action: Action::Local(local::mcp),
        ..CLI
    },
    Command {
        name: "hook",
        help: "Called by a harness hook; reads the event from stdin",
        args: &[pos("event", Kind::Text, "The hook event's name")],
        action: Action::Local(local::hook),
        ..CLI
    },
    Command {
        name: "hooks install",
        help: "Merge Polylogue's hook entries into the settings file, keeping everything else. Installing twice changes nothing",
        args: &[opt(
            "settings",
            Kind::Text,
            "Default: `$CLAUDE_CONFIG_DIR/settings.json` or `~/.claude/settings.json`",
        )],
        action: Action::Local(local::hooks_install),
        ..CLI
    },
    Command {
        name: "hooks uninstall",
        help: "Remove Polylogue's hook entries, keeping everything else",
        args: &[opt(
            "settings",
            Kind::Text,
            "Default: `$CLAUDE_CONFIG_DIR/settings.json` or `~/.claude/settings.json`",
        )],
        action: Action::Local(local::hooks_uninstall),
        ..CLI
    },
    Command {
        name: "completions",
        help: "Shell completions",
        args: &[pos("shell", Kind::Choice(SHELLS), "The shell")],
        action: Action::Local(local::completions),
        ..CLI
    },
];

/// The command called `name` (`note add`).
pub fn command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// The first word of every command the command line has: a query starting
/// with one of these is refused rather than searched for.
pub fn command_words() -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for c in COMMANDS.iter() {
        if matches!(c.cli, Cli::Subcommand | Cli::Hidden) {
            let first = c.name.split(' ').next().unwrap_or(c.name);
            if !words.iter().any(|w| w == first) {
                words.push(first.to_owned());
            }
        }
    }
    words
}

/// A command's argument values as one JSON object, keyed by argument name.
#[derive(Clone, Debug, Default)]
pub struct Values(pub Map<String, Value>);

impl Values {
    pub fn text(&self, name: &str) -> Option<String> {
        self.0.get(name).and_then(Value::as_str).map(str::to_owned)
    }

    pub fn flag(&self, name: &str) -> bool {
        self.0.get(name).and_then(Value::as_bool).unwrap_or(false)
    }

    pub fn integer(&self, name: &str) -> Option<u64> {
        self.0.get(name).and_then(Value::as_u64)
    }

    pub fn list(&self, name: &str) -> Vec<String> {
        self.0
            .get(name)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect()
    }

    pub fn session(&self, name: &str) -> Result<Option<SessionId>, String> {
        self.text(name)
            .map(|s| SessionId::parse(&s).map_err(|e| e.to_string()))
            .transpose()
    }

    /// The caller's request ID when it gave one, else a new one.
    pub fn request_id(&self) -> String {
        self.text("request_id")
            .unwrap_or_else(polylogue_store::user_writes::new_request_id)
    }

    /// Values from a JSON object (an MCP call, an HTTP body), checked
    /// against the command: unknown names, wrong types, missing required
    /// arguments and values outside a choice are refused. Declared defaults
    /// fill what is left out.
    pub fn from_json(
        command: &Command,
        surface: Surface,
        object: Map<String, Value>,
    ) -> Result<Values, String> {
        let name = command.name;
        for (key, value) in &object {
            if key == "request_id" && command.writes {
                if !value.is_string() {
                    return Err(format!("`{name}`: `request_id` is text"));
                }
                continue;
            }
            let Some(arg) = command.args_for(surface).find(|a| a.name == key) else {
                let known: Vec<&str> = command.args_for(surface).map(|a| a.name).collect();
                return Err(format!(
                    "`{name}` has no argument `{key}`; it takes: {}",
                    known.join(", ")
                ));
            };
            if !value.is_null() && !fits(arg.kind, value) {
                return Err(format!(
                    "`{name}`: `{key}` should be {}",
                    describe(arg.kind)
                ));
            }
        }
        let mut values = Values(object.into_iter().filter(|(_, v)| !v.is_null()).collect());
        values.complete(command, surface)?;
        Ok(values)
    }

    /// Values from URL query pairs (an HTTP GET): each converted to its
    /// argument's kind, then checked as [`Values::from_json`] checks.
    pub fn from_pairs(
        command: &Command,
        surface: Surface,
        pairs: Vec<(String, String)>,
    ) -> Result<Values, String> {
        let mut object = Map::new();
        for (key, text) in pairs {
            let kind = command
                .args_for(surface)
                .find(|a| a.name == key)
                .map(|a| a.kind);
            let value = match kind {
                Some(Kind::Integer) => text
                    .parse::<u64>()
                    .map(Value::from)
                    .map_err(|_| format!("`{}`: `{key}` should be a number", command.name))?,
                Some(Kind::Bool) => Value::Bool(matches!(text.as_str(), "" | "true" | "1")),
                Some(Kind::FlagOrText) if text.is_empty() || text == "true" => Value::Bool(true),
                Some(Kind::List) => {
                    let mut list = object
                        .remove(&key)
                        .and_then(|v| match v {
                            Value::Array(a) => Some(a),
                            _ => None,
                        })
                        .unwrap_or_default();
                    list.push(Value::String(text));
                    Value::Array(list)
                }
                _ => Value::String(text),
            };
            object.insert(key, value);
        }
        Self::from_json(command, surface, object)
    }

    /// Fills declared defaults and checks that required arguments are there.
    pub fn complete(&mut self, command: &Command, surface: Surface) -> Result<(), String> {
        for arg in command.args_for(surface) {
            if self.0.contains_key(arg.name) {
                continue;
            }
            if let Some(d) = arg.default {
                let value = match arg.kind {
                    Kind::Integer => d.parse::<u64>().map_or(Value::Null, Value::from),
                    _ => Value::String(d.to_owned()),
                };
                self.0.insert(arg.name.to_owned(), value);
            } else if arg.required {
                return Err(format!(
                    "`{}` needs `{}`: {}",
                    command.name, arg.name, arg.help
                ));
            }
        }
        Ok(())
    }
}

fn fits(kind: Kind, value: &Value) -> bool {
    match kind {
        Kind::Text | Kind::Words | Kind::Rest => value.is_string(),
        Kind::Integer => value.is_u64(),
        Kind::Bool => value.is_boolean(),
        Kind::List => value
            .as_array()
            .is_some_and(|a| a.iter().all(Value::is_string)),
        Kind::Choice(choices) => value.as_str().is_some_and(|v| choices.contains(&v)),
        Kind::FlagOrText => value.is_string() || value.as_bool() == Some(true),
    }
}

fn describe(kind: Kind) -> String {
    match kind {
        Kind::Text | Kind::Words | Kind::Rest => "text".into(),
        Kind::Integer => "a whole number".into(),
        Kind::Bool => "true or false".into(),
        Kind::List => "a list of text".into(),
        Kind::Choice(choices) => format!("one of {}", choices.join(", ")),
        Kind::FlagOrText => "true or text".into(),
    }
}

/// The JSON Schema of a command's arguments, for an MCP tool.
pub fn schema(command: &Command) -> Map<String, Value> {
    use serde_json::json;
    let mut properties = Map::new();
    for a in command.args_for(Surface::Mcp) {
        let mut p = match a.kind {
            Kind::Integer => json!({ "type": "integer", "minimum": 0 }),
            Kind::Bool => json!({ "type": "boolean" }),
            Kind::List => json!({ "type": "array", "items": { "type": "string" } }),
            Kind::Choice(choices) => json!({ "type": "string", "enum": choices }),
            Kind::FlagOrText => json!({ "type": ["string", "boolean"] }),
            Kind::Text | Kind::Words | Kind::Rest => json!({ "type": "string" }),
        };
        p["description"] = json!(a.help);
        if let Some(d) = a.default {
            p["default"] = match a.kind {
                Kind::Integer => d.parse::<u64>().map_or(Value::Null, Value::from),
                _ => json!(d),
            };
        }
        properties.insert(a.name.to_owned(), p);
    }
    if command.writes {
        properties.insert(
            "request_id".into(),
            json!({ "type": "string", "description": "Makes a resent call apply once" }),
        );
    }
    let required: Vec<&str> = command
        .args_for(Surface::Mcp)
        .filter(|a| a.required && a.default.is_none())
        .map(|a| a.name)
        .collect();
    let mut s = Map::new();
    s.insert("type".into(), json!("object"));
    s.insert("properties".into(), Value::Object(properties));
    s.insert("required".into(), json!(required));
    s.insert("additionalProperties".into(), json!(false));
    s
}

#[cfg(test)]
mod tests;
