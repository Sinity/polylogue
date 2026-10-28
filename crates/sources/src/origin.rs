//! The origin table: one declaration per origin, naming where its files
//! live, how a file is recognized and which parser reads it.

use crate::parse::Parse;
use polylogue_model::QuotaLimit;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Parsed into sessions and messages.
    Parsed,
    /// Declared only: where the tool writes is known; nothing is collected
    /// or parsed yet.
    Declared,
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

/// A tool's resume command line (program first) from a session's native
/// ID, or `None` when that session cannot be resumed on its own.
pub type Resume = fn(&str) -> Option<Vec<String>>;

pub struct Database {
    pub is_input: fn(&Path) -> bool,
}

pub struct OriginSpec {
    pub name: &'static str,
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

/// Fetches an origin's files into its spool, each pass while its setting
/// is `on`. The files it writes change a watched location, so the loop
/// collects them like any other.
pub struct Collector {
    pub setting: &'static str,
    /// Writes what is new into the spool (second argument); the state
    /// directory (first) holds the credentials the operator gave it.
    pub sync: fn(&Path, &Path) -> Result<(), String>,
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
/// `https://<host>/<path>/<conversation id>`.
pub struct WebChat {
    /// The names the browser extension gives this provider.
    pub extension_names: &'static [&'static str],
    /// The hosts that serve conversation pages, the current one first.
    pub hosts: &'static [&'static str],
    pub path: &'static str,
    /// Whether the extension can draft and submit turns on its pages.
    pub actions: bool,
    /// Whether the extension can page through the provider's conversation
    /// list to capture older conversations.
    pub backfill: bool,
}

impl OriginSpec {
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

macro_rules! declared {
    ($name:literal, $arrival:ident, [$($loc:literal),*]) => {
        declared!($name, $arrival, Root::HOME, [$($loc),*])
    };
    ($name:literal, $arrival:ident, $root:expr, [$($loc:literal),*]) => {
        OriginSpec {
            name: $name,
            support: Support::Declared,
            arrival: Arrival::$arrival,
            root: $root,
            locations: &[$($loc),*],
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
        }
    };
}

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
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
    declared!("antigravity", Live, [".gemini/antigravity/**"]),
    OriginSpec {
        name: "hermes",
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: crate::hermes::LOCATIONS,
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
        name: "export-file",
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
            actions: true,
            backfill: true,
        }),
        assets: Some(crate::chatgpt::assets::RULES),
        collector: None,
    },
    OriginSpec {
        name: "claude-ai",
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
            actions: false,
            backfill: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "claude-design",
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
            actions: false,
            backfill: false,
        }),
        assets: None,
        collector: Some(crate::ai_studio::drive::COLLECTOR),
    },
    OriginSpec {
        name: crate::gemini_app::ORIGIN,
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
            actions: false,
            backfill: false,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: "grok",
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
            actions: false,
            backfill: true,
        }),
        assets: None,
        collector: None,
    },
    OriginSpec {
        name: crate::otel_genai::ORIGIN,
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
