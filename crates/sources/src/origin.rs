//! The origin table: one declaration per origin, naming where its files
//! live, how a file is recognized and which parser reads it.

use crate::parse::StepParser;
use std::path::Path;

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

/// How an input's bytes divide into records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// One record per line; appends are parsed from where the last step
    /// stopped.
    Lines,
    /// One JSON document (or archive) read whole: an export, a capture the
    /// extension rewrites. Any change parses the whole input again.
    Document,
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
}

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

pub struct OriginSpec {
    pub name: &'static str,
    /// Bumped when the parser's output for the same bytes changes; the
    /// loop then parses the origin's inputs again. Kept apart from the
    /// input's format so a parser change never looks like a format change.
    pub parser_version: u32,
    pub support: Support,
    pub route: Route,
    pub framing: Framing,
    pub root: Root,
    /// Glob patterns relative to `root`, exactly the layout the tool
    /// writes. Empty for imports.
    pub locations: &'static [&'static str],
    pub capture: Capture,
    /// Whether a file is this origin's, from its path and first bytes. Runs
    /// before the origin is decided; a suffix alone never decides it.
    pub recognize: fn(&Path, &[u8]) -> bool,
    /// The parser for one input, or `None` when the path is not one this
    /// origin parses.
    pub open: fn(&Path) -> Option<Box<dyn StepParser>>,
}

fn never(_: &Path, _: &[u8]) -> bool {
    false
}

fn none(_: &Path) -> Option<Box<dyn StepParser>> {
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
            framing: Framing::Lines,
            root: $root,
            locations: &[$($loc),*],
            capture: Capture::Bytes,
            recognize: never,
            open: none,
        }
    };
}

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
        parser_version: 5,
        support: Support::Parsed,
        route: Route::Live,
        framing: Framing::Lines,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::recognize,
        open: crate::claude_code::open,
    },
    OriginSpec {
        name: "claude-code-side",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        framing: Framing::Document,
        root: crate::claude_code::side::SIDE_ROOT,
        locations: crate::claude_code::side::SIDE_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_side,
        open: crate::claude_code::side::open_side,
    },
    OriginSpec {
        name: "claude-code-history",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        framing: Framing::Lines,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::side::HISTORY_LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::claude_code::side::recognize_history,
        open: crate::claude_code::side::open_history,
    },
    OriginSpec {
        name: "codex",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::codex::ROOT,
        locations: crate::codex::LOCATIONS,
        capture: Capture::Bytes,
        framing: Framing::Lines,
        recognize: crate::codex::recognize,
        open: crate::codex::open,
    },
    OriginSpec {
        name: "codex-database",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: crate::codex::database::ROOT,
        locations: crate::codex::database::LOCATIONS,
        capture: Capture::LogicalExport,
        framing: Framing::Lines,
        recognize: crate::codex::database::recognize,
        open: crate::codex::database::open,
    },
    OriginSpec {
        name: "gemini-cli",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::gemini_cli::LOCATIONS,
        capture: Capture::Bytes,
        framing: Framing::Document,
        recognize: crate::gemini_cli::recognize,
        open: crate::gemini_cli::open,
    },
    reserved!("antigravity", Live, [".gemini/antigravity/**"]),
    OriginSpec {
        name: "hermes",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::kept_rows::HERMES_LOCATIONS,
        capture: Capture::LogicalExport,
        framing: Framing::Lines,
        recognize: crate::kept_rows::recognize_hermes,
        open: crate::kept_rows::open,
    },
    OriginSpec {
        name: "grok-cli",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        root: Root::HOME,
        locations: crate::grok_cli::LOCATIONS,
        capture: Capture::Bytes,
        framing: Framing::Lines,
        recognize: crate::grok_cli::recognize,
        open: crate::grok_cli::open,
    },
    OriginSpec {
        name: "polylogue-repo-lookup",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Spool,
        root: crate::repo_lookup::ROOT,
        locations: crate::repo_lookup::LOCATIONS,
        capture: Capture::Bytes,
        framing: Framing::Document,
        recognize: crate::repo_lookup::recognize,
        open: crate::repo_lookup::open,
    },
    OriginSpec {
        name: "hook-spool",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Spool,
        root: crate::hook_spool::ROOT,
        locations: crate::hook_spool::LOCATIONS,
        capture: Capture::Bytes,
        framing: Framing::Lines,
        recognize: crate::hook_spool::recognize,
        open: crate::hook_spool::open,
    },
    OriginSpec {
        name: "browser-capture",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Spool,
        framing: Framing::Document,
        root: crate::browser_capture::ROOT,
        locations: crate::browser_capture::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::browser_capture::recognize,
        open: crate::browser_capture::open,
    },
    OriginSpec {
        name: "export-file",
        parser_version: 0,
        support: Support::Kept,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: never,
        open: none,
    },
    OriginSpec {
        name: crate::polylogue_export::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::polylogue_export::recognize,
        open: crate::polylogue_export::open,
    },
    OriginSpec {
        name: "chatgpt",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::chatgpt::recognize,
        open: crate::chatgpt::open,
    },
    OriginSpec {
        name: "claude-ai",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize,
        open: crate::claude_ai::open,
    },
    OriginSpec {
        name: "claude-design",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::claude_ai::recognize_design,
        open: crate::claude_ai::open_design,
    },
    OriginSpec {
        name: crate::ai_studio::ORIGIN,
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Spool,
        framing: Framing::Document,
        root: crate::ai_studio::ROOT,
        locations: crate::ai_studio::LOCATIONS,
        capture: Capture::Bytes,
        recognize: crate::ai_studio::recognize,
        open: crate::ai_studio::open,
    },
    OriginSpec {
        name: crate::gemini_app::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::gemini_app::recognize,
        open: crate::gemini_app::open,
    },
    reserved!("grok", Import, []),
    OriginSpec {
        name: crate::otel_genai::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::otel_genai::recognize,
        open: crate::otel_genai::open,
    },
    OriginSpec {
        name: crate::openrouter::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openrouter::recognize,
        open: crate::openrouter::open,
    },
    // Last of the imports: its shapes are the most generic, so a specific
    // origin that recognizes the same file is tried first.
    OriginSpec {
        name: crate::openai_chat::ORIGIN,
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        recognize: crate::openai_chat::recognize,
        open: crate::openai_chat::open,
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
