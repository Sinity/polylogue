//! A temporary home with one Claude Code project directory (`-proj`) and an
//! archive beside it, for tests that write their own session lines.

use polylogue_engine::{Config, Engine};
use polylogue_model::export::SessionExport;
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
            home: self.home.clone(),
            tool_dirs: Default::default(),
            workers: 3,
            bytes_in_flight: 1 << 20,
            machine: "test-machine".into(),
            host: "test".into(),
            chunk,
            blob_roots: Vec::new(),
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

    /// The texts of `claude-code:<session>`'s own messages, in order.
    pub fn texts(&self, session: &str) -> Vec<String> {
        let id = SessionId::parse(&format!("claude-code:{session}")).expect("session ID");
        self.reads()
            .reader()
            .expect("a reader")
            .messages(&id, false)
            .expect("messages")
            .iter()
            .map(Message::text)
            .collect()
    }

    /// The `n` column of the first row `sql` returns.
    pub fn count(&self, sql: &str) -> i64 {
        self.reads()
            .reader()
            .expect("a reader")
            .query(sql, &[])
            .expect("query")[0]["n"]
            .as_i64()
            .expect("an integer `n`")
    }

    /// A session as the export format holds it.
    pub fn export(&self, session: &str) -> SessionExport {
        self.reads()
            .reader()
            .expect("a reader")
            .session_export(&SessionId::parse(session).expect("session ID"))
            .expect("export")
            .expect("the session exists")
    }
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
