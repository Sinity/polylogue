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

pub struct OriginSpec {
    pub name: &'static str,
    /// Bumped when the parser's output for the same bytes changes; the
    /// loop then parses the origin's inputs again. Kept apart from the
    /// input's format so a parser change never looks like a format change.
    pub parser_version: u32,
    pub support: Support,
    pub route: Route,
    /// Glob patterns relative to the home directory, exactly the layout
    /// the tool writes. Empty for imports.
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
        OriginSpec {
            name: $name,
            parser_version: 0,
            support: Support::Reserved,
            route: Route::$route,
            locations: &[$($loc),*],
            recognize: never,
            open: none,
        }
    };
}

pub static ORIGINS: &[OriginSpec] = &[
    OriginSpec {
        name: "claude-code",
        parser_version: 1,
        support: Support::Parsed,
        route: Route::Live,
        locations: crate::claude_code::LOCATIONS,
        recognize: crate::claude_code::recognize,
        open: crate::claude_code::open,
    },
    reserved!(
        "codex",
        Live,
        [".codex/sessions/**/*.jsonl", ".codex/history.jsonl"]
    ),
    reserved!("codex-database", Live, [".codex/*.sqlite"]),
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
    reserved!("chatgpt", Import, []),
    reserved!("claude-ai", Import, []),
    reserved!("claude-design", Import, []),
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
