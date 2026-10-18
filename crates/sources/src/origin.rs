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
            recognize: never,
            open: none,
        }
    };
}

const CODEX_HOME: Root = Root {
    env: Some("CODEX_HOME"),
    default: ".codex",
};

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
        parser_version: 2,
        support: Support::Parsed,
        route: Route::Live,
        framing: Framing::Lines,
        root: crate::claude_code::ROOT,
        locations: crate::claude_code::LOCATIONS,
        recognize: crate::claude_code::recognize,
        open: crate::claude_code::open,
    },
    reserved!(
        "codex",
        Live,
        CODEX_HOME,
        ["sessions/**/*.jsonl", "history.jsonl"]
    ),
    reserved!("codex-database", Live, CODEX_HOME, ["*.sqlite"]),
    reserved!("gemini-cli", Live, [".gemini/tmp/*/chats/*.json"]),
    reserved!("antigravity", Live, [".gemini/antigravity/**"]),
    reserved!("hermes", Live, [".hermes/state.db"]),
    reserved!("grok-cli", Live, [".grok/**"]),
    reserved!(
        "hook-spool",
        Spool,
        [".local/state/polylogue/hooks/*.ndjson"]
    ),
    reserved!(
        "browser-capture",
        Spool,
        [".local/state/polylogue/browser-capture/**"]
    ),
    OriginSpec {
        name: "export-file",
        parser_version: 0,
        support: Support::Kept,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
        recognize: never,
        open: none,
    },
    OriginSpec {
        name: "chatgpt",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Import,
        framing: Framing::Document,
        root: Root::HOME,
        locations: &[],
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
        recognize: crate::claude_ai::recognize_design,
        open: crate::claude_ai::open_design,
    },
    reserved!("ai-studio", Import, []),
    reserved!("gemini-app", Import, []),
    reserved!("grok", Import, []),
    reserved!("otel-genai", Import, []),
    reserved!("openai-chat", Import, []),
    reserved!("polylogue-export", Import, []),
];

pub fn by_name(name: &str) -> Option<&'static OriginSpec> {
    ORIGINS.iter().find(|o| o.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_valid_origins() {
        let mut names: Vec<_> = ORIGINS.iter().map(|o| o.name).collect();
        for n in &names {
            polylogue_model::Origin::new(n).unwrap();
        }
        names.sort();
        names.dedup();
        assert_eq!(names.len(), ORIGINS.len());
    }

    #[test]
    fn imports_have_no_live_locations() {
        for o in ORIGINS.iter().filter(|o| o.route == Route::Import) {
            assert!(o.locations.is_empty(), "{}", o.name);
        }
    }
}
