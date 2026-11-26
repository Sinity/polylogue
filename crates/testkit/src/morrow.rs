//! The Morrow example archive (`examples/morrow/`, described by its
//! `FACTS.md`): a week of synthetic work in Claude Code, Codex, ChatGPT and
//! claude.ai. "Morrow" is only the example's name. [`Home::morrow`](crate::Home::morrow)
//! lays its files out in a temporary home exactly where each tool writes
//! them; storing the home runs the engine's loop over it and imports the
//! exports.

use polylogue_model::SessionId;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// `examples/morrow/` in this repository.
pub fn morrow_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/morrow")
}

/// Which of the example's sets an archive holds (`FACTS.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Set {
    /// Sessions #1–#8: Claude Code, Codex with its state database, the
    /// hook spool, and the ChatGPT and claude.ai exports.
    Main,
    /// The live set V1–V3 with its hook spool files.
    Live,
    /// The main set plus the annotation session.
    Annotation,
}

/// Where each part of the example goes, relative to the example directory
/// and to the home.
const MAIN: &[(&str, &str)] = &[
    ("claude-code", ".claude"),
    ("codex/sessions", ".codex/sessions"),
    ("codex/state_5.sqlite", ".codex/state_5.sqlite"),
    ("hooks", ".local/state/polylogue/hooks"),
];
const LIVE: &[(&str, &str)] = &[
    ("live/claude-code", ".claude"),
    ("live/codex", ".codex"),
    ("live/hooks", ".local/state/polylogue/hooks"),
];
const ANNOTATION: &[(&str, &str)] = &[("annotation/claude-code", ".claude")];
/// Exports reach the archive by `import`, never by collection.
const EXPORTS: &[&str] = &[
    "chatgpt/export/conversations.json",
    "claude-ai/export/conversations.json",
];

impl crate::Home {
    /// A home holding one of the example's sets, laid out where each tool
    /// writes it, with the set's exports to import; nothing stored yet.
    /// `Home::morrow(set).stored()` is the built example.
    pub fn morrow(set: Set) -> Self {
        let mut home = Self::empty();
        let src = morrow_dir();
        let parts: Vec<&(&str, &str)> = match set {
            Set::Main => MAIN.iter().collect(),
            Set::Live => LIVE.iter().collect(),
            Set::Annotation => MAIN.iter().chain(ANNOTATION).collect(),
        };
        for (from, to) in parts {
            copy(&src.join(from), &home.home.join(to)).expect("lay out the example");
        }
        if matches!(set, Set::Main | Set::Annotation) {
            home.exports = EXPORTS.iter().map(|p| src.join(p)).collect();
        }
        home
    }
}

/// The ID of session `#n` of `FACTS.md` (1–8).
pub fn session(n: u8) -> SessionId {
    let id = match n {
        1 => "claude-code:7f3a2c10-0001-4000-8000-000000000001",
        2 => "claude-code:7f3a2c10-0002-4000-8000-000000000002",
        3 => "claude-code:agent-a7f3a2c1000000003",
        4 => "claude-code:7f3a2c10-0004-4000-8000-000000000004",
        5 => "codex:01a20000-0005-7000-8000-000000000005",
        6 => "codex:01a20000-0006-7000-8000-000000000006",
        7 => "chatgpt:68f0a1b2-0007-8000-8000-000000000007",
        8 => "claude-ai:0b9d4e6c-0008-4000-8000-000000000008",
        _ => panic!("Morrow has sessions #1–#8, not #{n}"),
    };
    SessionId::parse(id).expect("a valid session ID")
}

/// `#n` for a Morrow session ID, or the ID itself for any other.
pub fn short(id: &str) -> String {
    (1..=8)
        .find(|n| session(*n).to_string() == id)
        .map_or_else(|| id.to_owned(), |n| format!("#{n}"))
}

fn copy(from: &Path, to: &Path) -> io::Result<()> {
    if from.is_dir() {
        fs::create_dir_all(to)?;
        let mut entries: Vec<_> = fs::read_dir(from)?.collect::<Result<_, _>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            copy(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(from, to).map(drop)
    }
}

#[cfg(test)]
mod tests;
