//! The origin table: one declaration per origin, naming where its files
//! live, how a file is recognized and which parser reads it.

use crate::parse::Parse;
use polylogue_model::QuotaLimit;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Parsed into sessions and messages.
    Parsed,
    /// Collected as raw bytes or committed row dumps, with parse state
    /// `no_parser_yet` until the origin has a parser.
    NoParser,
    /// Bytes kept with no records in them: the images, audio and HTML an
    /// export carries beside its conversations. Messages refer to them.
    Kept,
}

/// How an input reaches Polylogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrival {
    /// A file the tool keeps writing at a fixed location.
    Live,
    /// An export the operator brings in with `polylogue import`.
    Import,
    /// A spool Polylogue's own hook, process watcher or browser extension
    /// writes.
    Spool,
}

/// The directory an origin's locations are relative to: the tool's own
/// override variable when it is set, otherwise a directory under home.
/// Exactly where the tool itself writes: Claude Code
/// honours `CLAUDE_CONFIG_DIR`, Codex `CODEX_HOME`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Root {
    pub env: Option<&'static str>,
    /// Relative to home; empty for home itself.
    pub default: &'static str,
}

impl Root {
    pub const HOME: Root = Root {
        env: None,
        default: "",
    };
    /// The root of Polylogue's own state directory ([`STATE_DIR`] under
    /// it): the spools its hook, browser host and collectors write.
    pub const STATE: Root = Root {
        env: Some("XDG_STATE_HOME"),
        default: ".local/state",
    };
}

/// Polylogue's own directory under [`Root::STATE`]. Every spool origin's
/// locations lie under it.
pub const STATE_DIR: &str = "polylogue";

/// What is stored for a file found at an origin's location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    /// The file's bytes.
    Bytes,
    /// A tool's SQLite database: a typed row dump of its rows,
    /// written by the store and parsed in place of the database, never
    /// page images.
    RowDump,
}

impl Arrival {
    pub const fn rank(self) -> u8 {
        match self {
            Self::Live => 0,
            Self::Spool => 1,
            Self::Import => 2,
        }
    }
}

/// A tool's resume command line (program first) from a session's native
/// ID, or `None` when that session cannot be resumed on its own.
pub type Resume = fn(&str) -> Option<Vec<String>>;

pub struct Database {
    pub is_input: fn(&Path) -> bool,
}

/// One module contributing to an origin's reading.
#[derive(Debug, Clone, Copy)]
pub struct ParserModule {
    pub name: &'static str,
    pub version: u32,
}

// Every parser runs through the shared framing, accounting and value readers.
macro_rules! modules {
    ($($module:path),* $(,)?) => { &[
        ParserModule{name:"parse",version:crate::parse::VERSION},
        ParserModule{name:"lines",version:crate::lines::VERSION},
        ParserModule{name:"records",version:crate::records::VERSION},
        ParserModule{name:"fields",version:crate::fields::VERSION},
        ParserModule{name:"document",version:crate::document::VERSION},
        ParserModule{name:"time",version:crate::time::VERSION},
        ParserModule{name:"command",version:polylogue_model::command::VERSION},
        $(ParserModule{name:stringify!($module),version:$module}),*
    ] };
}

pub struct OriginSpec {
    pub modules: &'static [ParserModule],
    pub name: &'static str,
    /// Text prefixes written by this harness in user-role messages.
    pub harness_prefixes: &'static [&'static str],
    pub support: Support,
    pub arrival: Arrival,
    pub root: Root,
    /// Glob patterns relative to `root`, exactly the layout the tool
    /// writes. Empty for imports.
    pub locations: &'static [&'static str],
    pub capture: Capture,
    /// Which inputs are row dumps.
    pub database: Option<Database>,
    /// Whether a file is this origin's, from its path and first bytes. Runs
    /// before the origin is decided; a suffix alone never decides it.
    pub recognize: fn(&Path, &[u8]) -> bool,
    /// The parser for one input, which also says whether the input is read
    /// line by line or whole; `None` when the path is not one this origin
    /// parses.
    pub open: fn(&Path) -> Option<Parse>,
    /// The tool's own command line that reopens a session, from its
    /// native ID; `None` inside when that session cannot be resumed on its
    /// own.
    pub resume: Option<Resume>,
    /// Whether a bare UUID the operator pastes can name one of this
    /// origin's sessions.
    pub bare_uuid_ids: bool,
    /// When the tool runs under a subscription quota: the limits it always
    /// has, whether or not a response reported them (others come with the
    /// readings its parser stores).
    pub quota: Option<&'static [QuotaLimit]>,
    /// Environment variables the tool sets in the commands its agent runs.
    /// Any of them in a process's environment means an agent started it.
    pub agent_env: &'static [&'static str],
    /// The web chat whose conversation pages hold this origin's sessions.
    pub web: Option<WebChat>,
    /// How files beside this origin's export name the assets its messages
    /// attach.
    pub assets: Option<ExportAssets>,
    /// What fetches this origin's files into its spool.
    pub collector: Option<Collector>,
}

/// Fetches an origin's files into its spool before the loop reads it.
pub struct Collector {
    /// None runs every pass; Some runs only when this setting is `on`.
    pub setting: Option<&'static str>,
    /// Run outside the loop thread when requests can wait on another process.
    pub background: bool,
    /// Writes new inputs into the spool and returns the errors for the pass.
    /// Background collectors check the stop flag between files.
    pub sync: fn(&CollectorPaths<'_>, &std::sync::atomic::AtomicBool) -> Vec<String>,
}

/// Configured directories a collector reads from and writes to.
pub struct CollectorPaths<'a> {
    pub home: &'a Path,
    pub state_dir: &'a Path,
    pub spool: &'a Path,
}

/// An export's asset files: the member files that hold attachment bytes,
/// named by the provider's asset ID, and the sidecars that describe them.
pub struct ExportAssets {
    /// The asset ID a member file holds, from its path in the export.
    pub member_id: fn(&str) -> Option<&str>,
    /// The kind of sidecar a member is, from its path in the export.
    pub sidecar_kind: fn(&str) -> Option<&str>,
    /// Reads a sidecar, one asset's metadata at a time.
    pub metadata: ReadSidecar,
}

/// Reads one sidecar member, handing each asset's metadata to the callback.
pub type ReadSidecar = fn(
    &str,
    &mut dyn std::io::Read,
    &mut dyn FnMut(AssetMetadata) -> Result<(), String>,
) -> Result<(), crate::export::ScanError>;

/// What a sidecar says about one asset.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetMetadata {
    pub source_id: String,
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub size: Option<u64>,
}

/// A web chat: the page that shows one conversation is
/// `https://<host>/<path>/<conversation id>`, with an optional declared prefix.
pub struct WebChat {
    /// The names the browser extension gives this provider.
    pub extension_names: &'static [&'static str],
    /// The hosts that serve conversation pages, the current one first.
    pub hosts: &'static [&'static str],
    pub path: &'static str,
    /// Optional two-segment prefixes before the conversation path.
    pub prefixes: &'static [LinkPrefix],
    /// Whether the extension can draft and submit turns on its pages.
    pub actions: bool,
    /// Whether the extension can page through the provider's conversation
    /// list to capture older conversations.
    pub backfill: bool,
}

/// A provider's optional account or project prefix on a conversation link.
pub struct LinkPrefix {
    pub path: &'static str,
    pub numeric: bool,
}

impl OriginSpec {
    /// Stable, ordered tuple of module names and versions.
    pub fn reading_version(&self) -> String {
        self.modules
            .iter()
            .map(|m| format!("{}={}", m.name, m.version))
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn is_row_dump(&self, path: &Path) -> bool {
        self.database
            .as_ref()
            .is_some_and(|database| (database.is_input)(path))
    }

    /// Where Polylogue's own writer puts a spool origin's files, relative
    /// to its root: the directory every location lies under, up to the
    /// first wildcard (`polylogue/hooks`). `None` for other arrivals.
    pub fn spool_dir(&self) -> Option<PathBuf> {
        if self.arrival != Arrival::Spool {
            return None;
        }
        fn fixed(l: &'static str) -> Vec<&'static str> {
            let segs: Vec<&str> = l.split('/').collect();
            // The last segment names files, never the directory.
            segs[..segs.len() - 1]
                .iter()
                .take_while(|s| !s.contains('*'))
                .copied()
                .collect()
        }
        let mut shared = fixed(self.locations.first()?);
        for l in &self.locations[1..] {
            let mine = fixed(l);
            let n = shared.iter().zip(&mine).take_while(|(a, b)| a == b).count();
            shared.truncate(n);
        }
        Some(shared.iter().collect())
    }
}

fn never(_: &Path, _: &[u8]) -> bool {
    false
}

fn none(_: &Path) -> Option<Parse> {
    None
}

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
        harness_prefixes: &[
            "<local-command-stdout>",
            "<local-command-stderr>",
            "<local-command-caveat>",
            "<system-reminder>",
            "<task-notification>",
            "[Request interrupted by user",
            "Caveat: The messages below were generated by the user while running local commands",
            "Caveat: The messages below are were generated by the user while running local commands",
            "[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.",
            "This session is being continued from a previous session that ran out of context.",
            "Base directory for this skill:",
        ],
        modules: modules!(crate::claude_code::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::claude_code::recognize,
        open: crate::claude_code::open,
        resume: Some(crate::claude_code::resume),
        bare_uuid_ids: true,
        quota: Some(crate::claude_code::QUOTA_LIMITS),
        agent_env: &["CLAUDECODE", "CLAUDE_CODE_SESSION_ID"],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "claude-code-side",
        harness_prefixes: &[],
        modules: modules!(
            crate::claude_code::VERSION,
            crate::claude_code::side::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::claude_code::side::SIDE_ROOT,
        locations: crate::claude_code::side::SIDE_LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::claude_code::side::recognize_side,
        open: crate::claude_code::side::open_side,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "claude-code-history",
        harness_prefixes: &[],
        modules: modules!(
            crate::claude_code::VERSION,
            crate::claude_code::side::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::side::HISTORY_LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::claude_code::side::recognize_history,
        open: crate::claude_code::side::open_history,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "codex",
        harness_prefixes: &[
            "<environment_context>",
            "# AGENTS.md instructions",
            "<user_instructions>",
            "<subagent_notification>",
            "<recommended_plugins>",
            "<codex_internal_context",
            "<turn_aborted>",
            "<skill>",
            "<user_action>",
            "<permissions instructions>",
        ],
        modules: modules!(crate::codex::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::codex::ROOT,
        locations: crate::codex::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::codex::recognize,
        open: crate::codex::open,
        resume: Some(crate::codex::resume),
        bare_uuid_ids: true,
        quota: Some(&[]),
        agent_env: &["CODEX_THREAD_ID", "CODEX_SESSION_ID"],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "codex-database",
        harness_prefixes: &[],
        modules: modules!(crate::codex::VERSION, crate::codex::database::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::codex::database::ROOT,
        locations: crate::codex::database::LOCATIONS,
        capture: Capture::RowDump,
        database: Some(Database {
            is_input: crate::codex::database::is_row_dump,
        }),
        recognize: crate::codex::database::recognize,
        open: crate::codex::database::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "gemini-cli",
        harness_prefixes: &[],
        modules: modules!(
            crate::chatgpt::VERSION,
            crate::export::VERSION,
            crate::gemini_cli::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: crate::gemini_cli::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::gemini_cli::recognize,
        open: crate::gemini_cli::open,
        resume: Some(crate::gemini_cli::resume),
        bare_uuid_ids: false,
        quota: None,
        agent_env: &["GEMINI_CLI"],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "antigravity",
        modules: modules!(crate::antigravity::VERSION),
        harness_prefixes: &[],
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: crate::antigravity::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::antigravity::recognize,
        open: crate::antigravity::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "antigravity-exports",
        modules: modules!(crate::antigravity::VERSION),
        harness_prefixes: &[],
        support: Support::Parsed,
        arrival: Arrival::Spool,
        root: Root::STATE,
        locations: &["polylogue/antigravity/*.json"],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::antigravity::recognize,
        open: crate::antigravity::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: Some(crate::antigravity::server::COLLECTOR),
    },
    OriginSpec {
        name: "hermes",
        harness_prefixes: &[],
        modules: modules!(crate::export::VERSION, crate::hermes::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: crate::locations::HERMES_ROOT,
        locations: &["state.db", "verification_evidence.db"],
        capture: Capture::RowDump,
        database: Some(Database {
            is_input: crate::hermes::is_database,
        }),
        recognize: crate::hermes::recognize,
        open: crate::hermes::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "grok-cli",
        harness_prefixes: &[],
        modules: modules!(crate::grok_cli::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: crate::grok_cli::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::grok_cli::recognize,
        open: crate::grok_cli::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "polylogue-repo-lookup",
        harness_prefixes: &[],
        modules: modules!(crate::repo_lookup::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        root: Root::STATE,
        locations: crate::repo_lookup::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::repo_lookup::recognize,
        open: crate::repo_lookup::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::hook_spool::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(crate::hook_spool::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        root: Root::STATE,
        locations: crate::hook_spool::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::hook_spool::recognize,
        open: crate::hook_spool::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::browser_capture::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::browser_capture::VERSION,
            crate::chatgpt::VERSION,
            crate::claude_ai::VERSION,
            crate::codex_cloud::VERSION,
            crate::grok::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        root: Root::STATE,
        locations: crate::browser_capture::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::browser_capture::recognize,
        open: crate::browser_capture::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "pi",
        harness_prefixes: &[],
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        root: Root {
            env: None,
            default: ".pi/agent",
        },
        locations: &["sessions/**/*.jsonl"],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "llm",
        harness_prefixes: &[],
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        root: Root {
            env: None,
            default: ".config/io.datasette.llm",
        },
        locations: &["logs.db"],
        capture: Capture::RowDump,
        database: Some(Database { is_input: |_| true }),
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "tool-context",
        harness_prefixes: &[],
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "tool-database",
        harness_prefixes: &[],
        modules: &[],
        support: Support::NoParser,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::RowDump,
        database: Some(Database { is_input: |_| true }),
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "codex-local-file",
        harness_prefixes: &[],
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "codex-compressed",
        harness_prefixes: &[],
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "codex-cloud",
        harness_prefixes: &[],
        modules: modules!(crate::export::VERSION, crate::codex_cloud::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::codex_cloud::recognize,
        open: crate::codex_cloud::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "aider",
        harness_prefixes: &[],
        modules: modules!(crate::export::VERSION, crate::aider::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::aider::recognize,
        open: crate::aider::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "export-file",
        harness_prefixes: &[],
        modules: &[],
        support: Support::Kept,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: never,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::polylogue_export::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::export::VERSION,
            crate::polylogue_export::VERSION,
            polylogue_model::export::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::polylogue_export::recognize,
        open: crate::polylogue_export::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "chatgpt",
        harness_prefixes: &[],
        modules: modules!(crate::chatgpt::VERSION, crate::export::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::chatgpt::recognize,
        open: crate::chatgpt::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: Some(WebChat {
            extension_names: &["chatgpt"],
            hosts: &["chatgpt.com", "chat.openai.com"],
            path: "c",
            prefixes: &[LinkPrefix {
                path: "g",
                numeric: false,
            }],
            actions: true,
            backfill: true,
        }),
        assets: Some(crate::chatgpt::assets::RULES),
        collector: None,
    },
    OriginSpec {
        name: "claude-ai",
        harness_prefixes: &[],
        modules: modules!(crate::chatgpt::VERSION, crate::claude_ai::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::claude_ai::recognize,
        open: crate::claude_ai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: Some(WebChat {
            extension_names: &["claude-ai", "claude"],
            hosts: &["claude.ai"],
            path: "chat",
            prefixes: &[],
            actions: false,
            backfill: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "claude-design",
        harness_prefixes: &[],
        modules: modules!(crate::chatgpt::VERSION, crate::claude_ai::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::claude_ai::recognize_design,
        open: crate::claude_ai::open_design,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::ai_studio::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::ai_studio::VERSION,
            crate::chatgpt::VERSION,
            crate::export::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Spool,
        root: Root::STATE,
        locations: crate::ai_studio::LOCATIONS,
        capture: Capture::Bytes,
        database: None,
        recognize: crate::ai_studio::recognize,
        open: crate::ai_studio::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: Some(WebChat {
            extension_names: &[],
            hosts: &["aistudio.google.com"],
            path: "prompts",
            prefixes: &[LinkPrefix {
                path: "u",
                numeric: true,
            }],
            actions: false,
            backfill: false,
        }),
        assets: None,
        collector: Some(crate::ai_studio::drive::COLLECTOR),
    },
    OriginSpec {
        name: crate::gemini_app::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(crate::export::VERSION, crate::gemini_app::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::gemini_app::recognize,
        open: crate::gemini_app::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: Some(WebChat {
            extension_names: &["gemini"],
            hosts: &["gemini.google.com"],
            path: "app",
            prefixes: &[LinkPrefix {
                path: "u",
                numeric: true,
            }],
            actions: false,
            backfill: false,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "grok",
        harness_prefixes: &[],
        modules: modules!(crate::export::VERSION, crate::grok::VERSION),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::grok::recognize,
        open: crate::grok::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: Some(WebChat {
            extension_names: &["grok"],
            hosts: &["grok.com"],
            path: "c",
            prefixes: &[],
            actions: false,
            backfill: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::otel_genai::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::chatgpt::VERSION,
            crate::export::VERSION,
            crate::otel_genai::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::otel_genai::recognize,
        open: crate::otel_genai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::openrouter::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::chatgpt::VERSION,
            crate::export::VERSION,
            crate::openrouter::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::openrouter::recognize,
        open: crate::openrouter::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
    // Last of the imports: its shapes are the most generic, so a specific
    // origin that recognizes the same file is tried first.
    OriginSpec {
        name: crate::openai_chat::ORIGIN,
        harness_prefixes: &[],
        modules: modules!(
            crate::chatgpt::VERSION,
            crate::export::VERSION,
            crate::openai_chat::VERSION
        ),
        support: Support::Parsed,
        arrival: Arrival::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: crate::openai_chat::recognize,
        open: crate::openai_chat::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    },
];

/// Environment variables that mean an agent started this process: every
/// origin's `agent_env`, and the job runners agents use.
pub fn agent_env() -> impl Iterator<Item = &'static str> {
    ORIGINS
        .iter()
        .flat_map(|origin| origin.agent_env.iter().copied())
        .chain(["AGENTCTL_JOB_ID", "AGENTCTL_RUN_ID", "POLYLOGUE_JOB_ID"])
}

pub fn by_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| o.name == name)
}

/// Every origin's export asset rules.
pub fn export_assets() -> impl Iterator<Item = &'static ExportAssets> {
    ORIGINS.iter().filter_map(|o| o.assets.as_ref())
}

/// The origin whose web chat the browser extension calls `name`.
pub fn by_extension_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| {
        o.web
            .as_ref()
            .is_some_and(|w| w.extension_names.contains(&name))
    })
}

#[cfg(test)]
mod tests;
