//! The origin table: one declaration per origin, naming where its files
//! live, how a file is recognized and which parser reads it.

use crate::parse::Parse;
use polylogue_model::QuotaLimit;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Parsed into sessions and messages.
    Parsed,
    /// Declared and collected (bytes are stored), not parsed yet. Shown as
    /// such in `status`, never as empty.
    Reserved,
    /// Bytes kept with no records in them: the images, audio and HTML an
    /// export carries beside its conversations. Messages refer to them.
    Kept,
}

/// How an input reaches Polylogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
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
/// Exactly where the tool itself writes (architecture 5.5b): Claude Code
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
    /// A tool's SQLite database: a typed logical export of its rows,
    /// written by the store and parsed in place of the database, never
    /// page images (architecture 4.2).
    LogicalExport,
}

/// A tool's resume command line (program first) from a session's native
/// ID, or `None` when that session cannot be resumed on its own.
pub type Resume = fn(&str) -> Option<Vec<String>>;

pub struct OriginSpec {
    pub name: &'static str,
    /// Bumped when the parser's output for the same bytes changes; the
    /// loop then parses the origin's inputs again. Kept apart from the
    /// input's format so a parser change never looks like a format change.
    pub parser_version: u32,
    pub support: Support,
    pub route: Route,
    pub root: Root,
    /// Glob patterns relative to `root`, exactly the layout the tool
    /// writes. Empty for imports.
    pub locations: &'static [&'static str],
    pub capture: Capture,
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
}

impl OriginSpec {
    /// Where Polylogue's own writer puts a spool origin's files, relative
    /// to its root: the directory every location lies under, up to the
    /// first wildcard (`polylogue/hooks`). `None` for other routes.
    pub fn spool_dir(&self) -> Option<PathBuf> {
        if self.route != Route::Spool {
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

macro_rules! reserved {
    ($name:literal, $route:ident, [$($loc:literal),*]) => {
        reserved!($name, $route, Root::HOME, [$($loc),*])
    };
    ($name:literal, $route:ident, $root:expr, [$($loc:literal),*]) => {
        OriginSpec {
            name: $name,
            parser_version: 0,
            support: Support::Reserved,
            route: Route::$route,
            root: $root,
            locations: &[$($loc),*],
            capture: Capture::Bytes,
            recognize: never,
            open: none,
            resume: None,
            bare_uuid_ids: false,
            quota: None,
        }
    };
}

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
        parser_version: 8,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::recognize,
        open: crate::claude_code::open,
        resume: Some(crate::claude_code::resume),
        bare_uuid_ids: true,
        quota: Some(crate::claude_code::QUOTA_LIMITS),
    },
    OriginSpec {
        name: "claude-code-side",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::claude_code::side::SIDE_ROOT,
        locations: crate::claude_code::side::SIDE_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_side,
        open: crate::claude_code::side::open_side,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "claude-code-history",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::side::HISTORY_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_history,
        open: crate::claude_code::side::open_history,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "codex",
        parser_version: 6,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::codex::ROOT,
        locations: crate::codex::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::codex::recognize,
        open: crate::codex::open,
        resume: Some(crate::codex::resume),
        bare_uuid_ids: true,
        quota: Some(&[]),
    },
    OriginSpec {
        name: "codex-database",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::codex::database::ROOT,
        locations: crate::codex::database::LOCATIONS,
        capture: Capture::LogicalExport,
        recognize: crate::codex::database::recognize,
        open: crate::codex::database::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "gemini-cli",
        parser_version: 3,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::gemini_cli::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::gemini_cli::recognize,
        open: crate::gemini_cli::open,
        resume: Some(crate::gemini_cli::resume),
        bare_uuid_ids: false,
        quota: None,
    },
    reserved!("antigravity", Live, [".gemini/antigravity/**"]),
    OriginSpec {
        name: "hermes",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::hermes::LOCATIONS,
        capture: Capture::LogicalExport,
        recognize: crate::hermes::recognize,
        open: crate::hermes::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "grok-cli",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::grok_cli::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::grok_cli::recognize,
        open: crate::grok_cli::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "polylogue-repo-lookup",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Spool,
        root: Root::STATE,
        locations: crate::repo_lookup::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::repo_lookup::recognize,
        open: crate::repo_lookup::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::hook_spool::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Spool,
        root: Root::STATE,
        locations: crate::hook_spool::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::hook_spool::recognize,
        open: crate::hook_spool::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::browser_capture::ORIGIN,
        parser_version: 5,
        support: Support::Parsed,
        route: Route::Spool,
        root: Root::STATE,
        locations: crate::browser_capture::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::browser_capture::recognize,
        open: crate::browser_capture::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "export-file",
        parser_version: 0,
        support: Support::Kept,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: never,
        open: none,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::polylogue_export::ORIGIN,
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::polylogue_export::recognize,
        open: crate::polylogue_export::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "chatgpt",
        parser_version: 4,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::chatgpt::recognize,
        open: crate::chatgpt::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "claude-ai",
        parser_version: 4,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize,
        open: crate::claude_ai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: "claude-design",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize_design,
        open: crate::claude_ai::open_design,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::ai_studio::ORIGIN,
        parser_version: 5,
        support: Support::Parsed,
        route: Route::Spool,
        root: Root::STATE,
        locations: crate::ai_studio::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::ai_studio::recognize,
        open: crate::ai_studio::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::gemini_app::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::gemini_app::recognize,
        open: crate::gemini_app::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    reserved!("grok", Import, []),
    OriginSpec {
        name: crate::otel_genai::ORIGIN,
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::otel_genai::recognize,
        open: crate::otel_genai::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    OriginSpec {
        name: crate::openrouter::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openrouter::recognize,
        open: crate::openrouter::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
    // Last of the imports: its shapes are the most generic, so a specific
    // origin that recognizes the same file is tried first.
    OriginSpec {
        name: crate::openai_chat::ORIGIN,
        parser_version: 3,
        support: Support::Parsed,
        route: Route::Import,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openai_chat::recognize,
        open: crate::openai_chat::open,
        resume: None,
        bare_uuid_ids: false,
        quota: None,
    },
];

pub fn by_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| o.name == name)
}

/// The export file name of a database found at a [`Capture::LogicalExport`]
/// origin's location: `<db file name>.jsonl`.
pub fn export_name(db: &Path) -> Option<String> {
    Some(format!("{}.jsonl", db.file_name()?.to_str()?))
}

#[cfg(test)]
mod tests;
