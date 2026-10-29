//! The Morrow example archive (`examples/morrow/`, described by its
//! `FACTS.md`): a week of synthetic work in Claude Code, Codex, ChatGPT and
//! claude.ai. "Morrow" is only the example's name. A [`Morrow`] lays its
//! files out in a temporary home exactly where each tool writes them, runs
//! the engine's loop over that home, and imports the exports.

use polylogue_engine::{Config, Engine, ImportReport, Pass};
use polylogue_model::SessionId;
use polylogue_store::{Archive, ArchiveLock, ReadPool};
use std::collections::BTreeMap;
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
    /// The Claude-Code-only home: #1–#4 and the parents set L1–L8.
    ClaudeCodeOnly,
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
const CLAUDE_CODE_ONLY: &[(&str, &str)] = &[("claude-code-only/claude-code", ".claude")];
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

/// A temporary home holding one of the example's sets and an archive
/// beside it.
pub struct Morrow {
    pub dir: tempfile::TempDir,
    pub home: PathBuf,
    pub root: PathBuf,
    pub set: Set,
}

impl Morrow {
    /// The set's files laid out in a fresh home; nothing run yet.
    pub fn lay_out(set: Set) -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let home = dir.path().join("home");
        let root = dir.path().join("archive");
        let src = morrow_dir();
        let parts: Vec<&(&str, &str)> = match set {
            Set::Main => MAIN.iter().collect(),
            Set::ClaudeCodeOnly => CLAUDE_CODE_ONLY.iter().collect(),
            Set::Live => LIVE.iter().collect(),
            Set::Annotation => MAIN.iter().chain(ANNOTATION).collect(),
        };
        for (from, to) in parts {
            copy(&src.join(from), &home.join(to)).expect("lay out the example");
        }
        Self {
            dir,
            home,
            root,
            set,
        }
    }

    /// The set run, with its exports imported.
    pub fn build(set: Set) -> Self {
        let morrow = Self::lay_out(set);
        morrow.run_pass();
        morrow
    }

    /// Settings that read only this archive's home.
    pub fn config(&self) -> Config {
        Config {
            clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
            home: self.home.clone(),
            tool_dirs: Default::default(),
            workers: 3,
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 64 << 20,
            machine: "morrow-machine".into(),
            host: "morrow".into(),
            chunk: 64 << 20,
            rescan_every: std::time::Duration::from_secs(60),
            live_every: std::time::Duration::from_secs(2),
        }
    }

    pub fn engine(&self) -> (Engine, ArchiveLock) {
        self.engine_with(self.config())
    }

    pub fn engine_with(&self, config: Config) -> (Engine, ArchiveLock) {
        let (archive, lock) = Archive::open_or_create(&self.root).expect("open the archive");
        (
            Engine::new(config, archive).expect("start the engine"),
            lock,
        )
    }

    /// One pass of the loop, then the set's exports imported.
    pub fn run_pass(&self) -> (Pass, Option<ImportReport>) {
        let (mut engine, _lock) = self.engine();
        let pass = engine.run_pass().expect("a pass");
        let report = self.import_exports(&mut engine);
        (pass, report)
    }

    /// Imports the set's exports (the main set has two).
    pub fn import_exports(&self, engine: &mut Engine) -> Option<ImportReport> {
        if !matches!(self.set, Set::Main | Set::Annotation) {
            return None;
        }
        let paths: Vec<PathBuf> = EXPORTS.iter().map(|p| morrow_dir().join(p)).collect();
        Some(engine.import(&paths).expect("import the exports"))
    }

    pub fn reads(&self) -> ReadPool {
        ReadPool::open(&Archive::at(&self.root)).expect("open readers")
    }

    /// Every session as the export format holds it, by ID: what tests
    /// compare when they say "the view is unchanged".
    pub fn view(&self) -> BTreeMap<String, serde_json::Value> {
        let pool = self.reads();
        let r = pool.reader().expect("a reader");
        r.query("SELECT id FROM session", &[])
            .expect("list sessions")
            .into_iter()
            .filter_map(|row| {
                let id = SessionId::parse(row["id"].as_str()?).ok()?;
                let doc = r.session_export(&id).expect("export a session")?;
                Some((
                    id.to_string(),
                    serde_json::to_value(doc).expect("serialize"),
                ))
            })
            .collect()
    }

    /// The bytes stored for the file at `path`'s current revision, read
    /// back through the blob store. A missing or corrupt blob is an error.
    pub fn stored_bytes(&self, path: &Path) -> Result<Vec<u8>, String> {
        let pool = self.reads();
        let r = pool.reader().map_err(|e| e.to_string())?;
        let known = r
            .known_watched_inputs(&self.config().machine)
            .map_err(|e| e.to_string())?;
        let input = known
            .iter()
            .find(|k| Path::new(&k.path) == path)
            .ok_or_else(|| format!("{} was never observed", path.display()))?;
        let chain = input.chain.ok_or("no stored revision")?;
        let stored = r
            .stored_chain(&input.input_id, chain, input.stored)
            .map_err(|e| e.to_string())?;
        let store = polylogue_store::Archive::at(&self.root)
            .blobs()
            .map_err(|e| e.to_string())?;
        stored
            .read_range(&store, 0, input.stored)
            .map_err(|e| e.to_string())
    }

    /// A source file of this archive, by its path under the home.
    pub fn file(&self, rel: &str) -> PathBuf {
        self.home.join(rel)
    }

    /// Every source file collection reads in this archive, in path order.
    pub fn provider_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        files_under(&self.home, &mut out).expect("list the home");
        out.sort();
        out
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

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files_under(&path, out)?;
        } else {
            out.push(path);
        }
    }
    Ok(())
}
