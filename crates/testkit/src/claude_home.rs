//! A temporary home with one Claude Code project directory (`-proj`) and an
//! archive beside it, for tests that write their own session lines.

use polylogue_engine::{Config, Engine};
use polylogue_model::{Message, SessionId};
use polylogue_store::{Archive, ArchiveLock, ReadPool};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct ClaudeHome {
    pub dir: tempfile::TempDir,
    pub home: PathBuf,
    pub root: PathBuf,
}

impl Default for ClaudeHome {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeHome {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let home = dir.path().join("home");
        fs::create_dir_all(home.join(".claude/projects/-proj")).expect("project directory");
        let root = dir.path().join("archive");
        Self { dir, home, root }
    }

    /// Settings that read only this home; `chunk` is the most one job reads.
    pub fn config(&self, chunk: u64) -> Config {
        Config {
            clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
            home: self.home.clone(),
            tool_dirs: Default::default(),
            workers: 3,
            memory_budget: Default::default(),
            reader_memory_bytes: 128 << 20,
            bytes_in_flight: 1 << 20,
            machine: "test-machine".into(),
            host: "test".into(),
            chunk,
            live_every: std::time::Duration::from_secs(2),
            reads_sources: true,
        }
    }

    pub fn engine(&self, chunk: u64) -> (Engine, ArchiveLock) {
        let (archive, lock) = Archive::open_or_create(&self.root).expect("open the archive");
        (
            Engine::new(self.config(chunk), archive).expect("start the engine"),
            lock,
        )
    }

    /// A session file in the project directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.home.join(".claude/projects/-proj").join(name)
    }

    pub fn append(&self, name: &str, text: &str) {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.file(name))
            .expect("open the session file");
        f.write_all(text.as_bytes()).expect("append");
    }

    pub fn reads(&self) -> ReadPool {
        ReadPool::open(&Archive::at(&self.root)).expect("open readers")
    }

    /// Runs `read` on a new reader.
    pub fn read<T>(
        &self,
        read: impl Fn(&polylogue_store::Reader<'_>) -> polylogue_store::Result<T>,
    ) -> T {
        self.reads()
            .reader()
            .and_then(|reader| read(&reader))
            .expect("read the archive")
    }

    /// The index file as the archive holds it now: the private build file
    /// while a build runs, otherwise `index.db`.
    pub fn index_db(&self) -> PathBuf {
        Archive::at(&self.root).index_db()
    }

    /// The texts of `claude-code:<session>`'s own messages, in order.
    pub fn texts(&self, session: &str) -> Vec<String> {
        let id = SessionId::parse(&format!("claude-code:{session}")).expect("session ID");
        self.read(|reader| reader.messages(&id, false))
            .iter()
            .map(Message::text)
            .collect()
    }

    /// The `n` column of the first row `sql` returns.
    pub fn count(&self, sql: &str) -> i64 {
        self.read(|reader| reader.query(sql, &[]))[0]["n"]
            .as_i64()
            .expect("an integer `n`")
    }

    /// A session's current bundle records, read from the stored rows.
    pub fn export(&self, session: &str) -> Vec<polylogue_model::bundle::Record> {
        self.read(|reader| {
            crate::session_bundle(reader, &SessionId::parse(session).expect("session ID"))
        })
    }
}

/// Installs a new archive's empty index the way a daemon started without
/// sources does: its first pass reads no source and installs the empty
/// build (docs/installing.md, replacing an index). Files in the home are
/// left for the caller's own passes, which then meet an installed index.
pub fn install_empty_index(engine: &mut Engine) {
    let reads_sources = engine.config.reads_sources;
    engine.config.reads_sources = false;
    let pass = engine.run_pass().expect("the first pass");
    engine.config.reads_sources = reads_sources;
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        engine.archive.index_build().is_none(),
        "the index is installed"
    );
}

/// Deletes `index.db` (and its WAL files) of a closed archive.
pub fn delete_index(root: &Path) {
    for suffix in ["", "-wal", "-shm"] {
        let _ = fs::remove_file(root.join(format!("index.db{suffix}")));
    }
}

/// A Claude Code user record holding `text`.
pub fn claude_user(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "message": {"role": "user", "content": text}
        })
    )
}

/// A Claude Code assistant record holding `text`, one second after the
/// user record's time.
pub fn claude_assistant(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "assistant", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": format!("msg_{uuid}"), "role": "assistant", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": text}]}
        })
    )
}
