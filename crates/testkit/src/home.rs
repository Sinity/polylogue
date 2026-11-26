//! A synthetic home with an archive beside it: the one way the testkit
//! builds an archive. Files go into the home where each tool writes them
//! (laid out from Morrow, written from an [`Activity`](crate::Activity), or
//! appended line by line), and [`Home::store`] stores them through the real
//! engine and import path.

use polylogue_engine::{Config, Engine, Pass};
use polylogue_model::{Message, SessionId};
use polylogue_store::{Archive, ArchiveLock, ReadPool};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A chunk larger than any test file: the engine reads each file in one job.
pub const WHOLE_FILES: u64 = 64 << 20;

/// The machine name every test archive records its inputs under.
pub const MACHINE: &str = "test-machine";

/// A temporary directory holding `home/` and `archive/`, and the exports
/// that [`Home::store`] imports after its pass.
pub struct Home {
    pub dir: tempfile::TempDir,
    pub home: PathBuf,
    pub root: PathBuf,
    pub exports: Vec<PathBuf>,
}

impl Default for Home {
    fn default() -> Self {
        Self::new()
    }
}

impl Home {
    /// An empty home with one Claude Code project directory (`-proj`, see
    /// [`Home::file`]) and no archive yet.
    pub fn new() -> Self {
        let home = Self::empty();
        fs::create_dir_all(home.home.join(".claude/projects/-proj")).expect("project directory");
        home
    }

    /// A home with nothing in it.
    pub fn empty() -> Self {
        let dir = tempfile::tempdir().expect("temporary directory");
        let home = dir.path().join("home");
        fs::create_dir_all(&home).expect("home directory");
        let root = dir.path().join("archive");
        Self {
            dir,
            home,
            root,
            exports: Vec::new(),
        }
    }

    /// The home with its files stored, for a caller that builds and reads.
    pub fn stored(self) -> Self {
        self.store();
        self
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
            machine: MACHINE.into(),
            host: "test".into(),
            chunk,
            live_every: std::time::Duration::from_secs(2),
            reads_sources: true,
        }
    }

    pub fn engine(&self, chunk: u64) -> (Engine, ArchiveLock) {
        self.engine_with(self.config(chunk))
    }

    /// An engine over this home's archive, for a test that drives passes
    /// itself (stopping between steps, arming faults). The only place the
    /// testkit starts one.
    pub fn engine_with(&self, config: Config) -> (Engine, ArchiveLock) {
        let (archive, lock) = Archive::open_or_create(&self.root).expect("open the archive");
        (
            Engine::new(config, archive, crate::stopped_attempt()).expect("start the engine"),
            lock,
        )
    }

    /// Stores what is in the home: starts an engine, runs it until nothing
    /// is due and imports the home's exports. Any error fails the test.
    pub fn store(&self) -> Pass {
        let (mut engine, _lock) = self.engine(WHOLE_FILES);
        store_with(&mut engine, &self.exports)
    }

    /// A session file in the `-proj` Claude Code project directory.
    pub fn file(&self, name: &str) -> PathBuf {
        self.home.join(".claude/projects/-proj").join(name)
    }

    /// A file of this home, by its path under the home.
    pub fn path(&self, rel: &str) -> PathBuf {
        self.home.join(rel)
    }

    /// Appends `text` to the session file `name` in `-proj`.
    pub fn append(&self, name: &str, text: &str) {
        self.append_bytes(&self.file(name), text.as_bytes());
    }

    /// Appends `bytes` to the file at `path`, creating it and its directory.
    pub fn append_bytes(&self, path: &Path, bytes: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("the file's directory");
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open the file");
        f.write_all(bytes).expect("append");
    }

    pub fn reads(&self) -> ReadPool {
        ReadPool::open(&Archive::at(&self.root)).expect("open readers")
    }

    /// Runs `read` on a new reader, again when the index was replaced under
    /// it, as a build's install does (`StoreError::ReadIndexReplaced`).
    pub fn read<T>(
        &self,
        read: impl Fn(&polylogue_store::Reader<'_>) -> polylogue_store::Result<T>,
    ) -> T {
        self.reads().read_current(read).expect("read the archive")
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

    /// Every session as current bundle records, by ID: what tests
    /// compare when they say "the view is unchanged".
    pub fn view(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        let pool = self.reads();
        let r = pool.reader().expect("a reader");
        r.query("SELECT id FROM session", &[])
            .expect("list sessions")
            .into_iter()
            .filter_map(|row| {
                let id = SessionId::parse(row["id"].as_str()?).ok()?;
                let doc = crate::session_bundle(&r, &id).expect("read session bundle");
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
        let known = r.known_watched_inputs(MACHINE).map_err(|e| e.to_string())?;
        let input = known
            .iter()
            .find(|k| Path::new(&k.path) == path)
            .ok_or_else(|| format!("{} was never observed", path.display()))?;
        let chain = input.chain.ok_or("no stored revision")?;
        let stored = r
            .stored_chain(&input.input_id, chain, input.stored)
            .map_err(|e| e.to_string())?;
        let store = Archive::at(&self.root).blobs().map_err(|e| e.to_string())?;
        stored
            .read_range(&store, 0, input.stored)
            .map_err(|e| e.to_string())
    }

    /// Every file in the home, in path order.
    pub fn provider_files(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        files_under(&self.home, &mut out).expect("list the home");
        out.sort();
        out
    }
}

/// Runs `engine` until nothing is due and imports `exports`; any error
/// fails the test. With [`Home::engine_with`], the testkit's only engine
/// calls.
pub fn store_with(engine: &mut Engine, exports: &[PathBuf]) -> Pass {
    let pass = engine.run_pass().expect("a pass");
    assert_eq!(pass.errors, Vec::new(), "the pass failed");
    if !exports.is_empty() {
        let report = engine.import(exports).expect("import the exports");
        assert_eq!(report.pass.errors, Vec::new(), "the import failed");
        assert!(report.refused.is_empty(), "refused: {:?}", report.refused);
    }
    pass
}

/// Installs a new archive's empty index the way a daemon started without
/// sources does: its first pass reads no source and installs the empty
/// build (docs/installing.md, replacing an index). Files in the home are
/// left for the caller's own passes, which then meet an installed index.
pub fn install_empty_index(engine: &mut Engine) {
    let reads_sources = engine.config.reads_sources;
    engine.config.reads_sources = false;
    store_with(engine, &[]);
    engine.config.reads_sources = reads_sources;
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

fn files_under(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
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
