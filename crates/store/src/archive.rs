//! The archive directory, its single-writer lock and connection settings.

pub mod build_progress;
pub(crate) mod generation;
pub mod index_build;
mod metadata;
pub mod writer;
pub use metadata::Mode;

use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::read::scope::InitialInvalidation;
use crate::{Result, StoreError, schema};
use rusqlite::{Connection, OpenFlags, Transaction};
use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Increments the database's write counter; one value per transaction.
pub(crate) fn bump_counter(tx: &Transaction<'_>) -> Result<i64> {
    let next: i64 = crate::write_counter(tx)?
        .checked_add(1)
        .or_internal("write counter exhausted")?;
    crate::meta_set(tx, "write_counter", next.to_string())?;
    Ok(next)
}

/// One process allowance, split between the index pipeline's resident caches
/// and its work. The allowance is the memory the process may use (its
/// control group's limit, else the memory the system has available when it
/// starts); the budget is three quarters of it, leaving room for the
/// program, the allocator's slack and the kernel. Nothing is capped at a
/// fixed size: a machine with more memory gives every share more.
#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    total: u64,
}
impl Default for MemoryBudget {
    fn default() -> Self {
        Self::from_allowance(4 << 30)
    }
}

/// A parse worker's own memory, at least: the cache of its working index
/// and its sort buffer, from the third of the budget workers share.
const WORKER_MEMORY: u64 = 128 << 20;

impl MemoryBudget {
    pub fn from_allowance(bytes: u64) -> Self {
        Self {
            total: bytes.saturating_mul(3).saturating_div(4),
        }
    }
    pub fn total(self) -> u64 {
        self.total
    }
    pub fn writer_cache_mib(self) -> i64 {
        ((self.total / 8) >> 20).max(1) as i64
    }
    /// The build file's cache while the workers prepare: the writer then
    /// only applies what upkeep writes meanwhile. Each later step of the
    /// writer's build sets it to what that step leaves ([`Self::build_cache_beside`]).
    pub fn build_cache_mib(self) -> i64 {
        ((self.total / 8) >> 20).max(1) as i64
    }
    /// The build file's cache beside `held` bytes the process holds or will
    /// hold outside it in the writer's build step (what is resident beside
    /// the cache, and the step's run readers, sort buffers and composition
    /// read-ahead): the budget less that and a reserve for the work itself,
    /// at most a quarter of the budget. Composing a session holds its rows'
    /// item entries, values and the like in proportion to the session, which
    /// no share bounds, so the cache leaves it that room.
    pub fn build_cache_beside(self, held: u64) -> i64 {
        let others = held.saturating_add(self.total / 8);
        (self.total.saturating_sub(others).min(self.total / 4) >> 20).max(32) as i64
    }
    pub fn archive_cache_mib(self) -> i64 {
        ((self.total / 32) >> 20).max(1) as i64
    }
    /// What the merge of a build's runs may read them with: the share the
    /// workers' prepared readings held, free once they are done.
    pub(crate) fn run_merge_memory(self) -> usize {
        (self.total / 3) as usize
    }
    /// What each smaller merge or sort beside it takes.
    pub(crate) fn side_merge_memory(self) -> usize {
        (self.total / 16) as usize
    }
    /// What a build's composition holds besides the build file's cache: the
    /// stream decoded ahead of the composers, the composers' own indexes and
    /// the rows they composed that wait for the writer.
    pub(crate) fn composition_memory(self) -> usize {
        (self.total / 8) as usize
    }
    /// What a build's full-text fill holds of texts read ahead of its
    /// files' threads.
    pub(crate) fn search_fill_memory(self) -> usize {
        (self.total / 16) as usize
    }
    pub fn search_cache_mib(self) -> i64 {
        ((self.total / 64) >> 20).max(1) as i64
    }
    pub fn readers(self) -> u64 {
        self.total / 16
    }
    pub fn in_flight(self) -> u64 {
        self.total / 16
    }
    /// The parse workers' prepared readings in flight, at most: a worker
    /// waits before it starts a chain until the chain's share fits.
    pub fn prepared_in_flight(self) -> u64 {
        self.total / 3
    }
    /// Each of `workers` build workers' own memory: its sort buffer and the
    /// cache of its working index, together a third of the budget.
    pub fn build_worker_memory(self, workers: usize) -> usize {
        (self.total / 3 / workers.max(1) as u64) as usize
    }
    /// Parse workers: one per core the process may use, as long as each
    /// has at least [`WORKER_MEMORY`] of its own ([`Self::build_worker_memory`]).
    /// The count does not trade against the caches: memory a worker's
    /// prepared readings need is admitted per chain
    /// ([`Self::prepared_in_flight`]).
    pub fn worker_count(self, cores: usize, requested: usize) -> usize {
        cores
            .min(requested)
            .min((self.total / 3 / WORKER_MEMORY) as usize)
            .max(1)
    }
}

/// An opened archive directory (`$XDG_DATA_HOME/polylogue`). A directory
/// Polylogue creates has copy-on-write turned off on btrfs; an existing one
/// keeps its own setting. Polylogue checks its own data with BLAKE3 for blobs. SQLite's checksum VFS is not wired in yet: it
/// needs the `cksumvfs` extension, which the bundled build does not ship.
#[derive(Debug, Clone)]
pub struct Archive {
    memory: MemoryBudget,
    pub root: PathBuf,
    pub(crate) read_scope: std::sync::Arc<crate::read::scope::ReadScopeState>,
    /// The stop request of the process using this archive. Every reader the
    /// store hands out, the writer's long work and the engine's loop observe
    /// it; clones share it.
    stop: crate::cancellation::Cancellation,
    /// The index files' state and a rebuild's progress, shared by every
    /// value for this root ([`index_build`]).
    pub(crate) index_build: std::sync::Arc<index_build::State>,
    /// The summary worker readers of this root ask, shared like `index_build`.
    pub(crate) summaries: std::sync::Arc<crate::summary::ask::Asking>,
}

/// Held by the daemon for as long as it runs. Only the holder writes.
#[derive(Debug)]
pub struct ArchiveLock {
    _file: File,
}

impl Archive {
    pub fn memory_budget(&self) -> MemoryBudget {
        self.memory
    }
    pub fn with_memory_budget(mut self, budget: MemoryBudget) -> Self {
        self.memory = budget;
        self
    }

    /// The stop request of the process using this archive: cancelling it
    /// ends every read, wait and reconstructible write made through it with
    /// [`StoreError::Stopped`](crate::StoreError::Stopped). Requests derive
    /// their own cancellation from it with [`Cancellation::child`](crate::cancellation::Cancellation::child).
    pub fn stop(&self) -> &crate::cancellation::Cancellation {
        &self.stop
    }

    /// This archive as a new process would open it, with a stop request of
    /// its own: a test's restart after a stop.
    #[cfg(any(test, feature = "fixture"))]
    pub fn reopened(&self) -> Self {
        Self {
            stop: crate::cancellation::Cancellation::stop_request(),
            ..self.clone()
        }
    }

    /// Resolve existing ancestors even before this archive's directory exists.
    pub fn resolved_root(root: &Path) -> PathBuf {
        crate::read::scope::root_key(root)
    }

    /// Register while holding the writer lock of a live shared archive. Startup
    /// uses the open-with-invalidator constructors to take that lock first.
    /// The callback runs immediately and at each reset, under the publication fence;
    /// it must invalidate its pointer on disk without acquiring archive readers
    /// or re-entering the scope fence.
    pub fn register_scope_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<crate::read::scope::ScopeInvalidation> {
        crate::read::scope::state(root).register(invalidate, InitialInvalidation::Invalidate)
    }

    /// A transition guard without SQL. Currentness still comes from a reader's
    /// confirmed checkpoint; this token alone cannot authorize cache activation.
    pub fn scope_token(&self) -> Result<crate::ReadScopeToken> {
        self.read_scope.token()
    }

    /// Start a new index generation in synthetic transport tests.
    /// This changes cursor identity only.
    #[cfg(any(test, feature = "fixture"))]
    pub fn advance_generation_for_test(&self) -> Result<()> {
        let mut index = open_write(&self.index_db(), cache::SETUP)?;
        index.pragma_update(None, "synchronous", 2)?;
        let mut reset = self.read_scope.begin_reset()?;
        reset.invalidate()?;
        drop(reset);
        let tx = index.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        generation::rotate(&tx)?;
        tx.commit()?;
        self.read_scope.finish_reset();
        Ok(())
    }

    /// Where the archive's record is kept: inputs, revisions, extents,
    /// observations, blobs and authored records.
    pub fn archive_db(&self) -> PathBuf {
        self.root.join("archive.db")
    }

    /// Resolve content hashes in Polylogue's committed catalog and read the
    /// immutable segment extents with the shared reader.
    pub fn blobs(&self) -> Result<crate::blobs::Blobs> {
        self.blob_reader()
    }

    pub fn blob_reader(&self) -> Result<crate::blobs::Blobs> {
        Ok(crate::blobs::Blobs::new(
            self.root.clone(),
            self.archive_db(),
        ))
    }

    pub fn embeddings_db(&self) -> PathBuf {
        self.root.join("embeddings.db")
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("segments")
    }

    /// The daemon's socket. Clients find the daemon of an archive here.
    pub fn socket(&self) -> PathBuf {
        self.root.join("daemon.sock")
    }

    /// The archive at `root` without opening it, for clients and readers.
    pub fn at(root: &Path) -> Self {
        Self {
            memory: MemoryBudget::default(),
            root: root.to_path_buf(),
            read_scope: crate::read::scope::state(root),
            stop: crate::cancellation::Cancellation::stop_request(),
            index_build: index_build::state(root),
            summaries: crate::summary::ask::state(root),
        }
    }

    /// Whether a daemon holds this archive's writer lock. A daemon takes
    /// the lock before it opens the archive and binds its socket, so a held
    /// lock with no socket answering means a daemon is still starting.
    pub fn lock_held(&self) -> bool {
        let Ok(file) = OpenOptions::new().read(true).open(self.root.join("lock")) else {
            return false;
        };
        match file.try_lock_shared() {
            Ok(()) => false,
            Err(std::fs::TryLockError::WouldBlock) => true,
            Err(std::fs::TryLockError::Error(_)) => false,
        }
    }

    /// Creates the directory with copy-on-write off and the three databases
    /// at their current versions, or opens an existing one. Takes the writer
    /// lock; a second daemon on the same archive is refused.
    pub fn open_or_create(root: &Path) -> Result<(Self, ArchiveLock)> {
        let (archive, lock, ()) = Self::open_configured(Self::at(root), |_| Ok(()))?;
        Ok((archive, lock))
    }

    /// Take the writer lock before registering cache invalidation and opening
    /// the standalone archive. Keep the returned registration with the lock.
    pub fn open_or_create_with_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        Self::open_configured(Self::at(root), |archive| {
            archive
                .read_scope
                .register(invalidate, InitialInvalidation::Keep)
        })
    }

    fn open_configured<T>(
        archive: Self,
        setup: impl FnOnce(&Self) -> Result<T>,
    ) -> Result<(Self, ArchiveLock, T)> {
        metadata::check(&archive)?;
        let lock = Self::lock(&archive.root)?;
        metadata::check(&archive)?;
        metadata::create(&archive)?;
        let setup = setup(&archive)?;
        let opened = (|| {
            fs::create_dir_all(archive.blobs_dir())?;
            let source = open_write(&archive.archive_db(), cache::SETUP)?;
            let schema_lock = crate::write_lock::hold(&source)?;
            schema::ensure_archive(&source)?;
            drop(schema_lock);
            archive.finish_open()
        })();
        if let Err(error) = opened {
            // A failed open cannot leave the previous cache authorized. Keep
            // the writer lock and registration until its pointer is revoked.
            let mut reset = archive.read_scope.begin_reset()?;
            reset.invalidate()?;
            drop(reset);
            archive.read_scope.finish_reset();
            return Err(error);
        }
        Ok((archive, lock, setup))
    }

    /// The archive's recorded mode, read without opening or creating a database.
    pub fn recorded_mode(root: &Path) -> Result<Option<Mode>> {
        metadata::read(root)
    }

    /// Broken deployment settings pause capture and writes while the recorded
    /// archive remains readable. They never choose another storage mode: an
    /// archive recorded as primary is refused, as every open refuses it.
    pub fn open_blocked_with_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        let archive = match Self::recorded_mode(root)? {
            Some(_) => Self::at(root),
            None => {
                return Err(polylogue_model::Reason::ArchiveModeMissing {
                    archive: root.display().to_string(),
                }
                .into());
            }
        };
        Self::open_configured(archive, |archive| {
            archive
                .read_scope
                .register(invalidate, InitialInvalidation::Invalidate)
        })
    }

    fn lock(root: &Path) -> Result<ArchiveLock> {
        let fresh = !root.exists();
        fs::create_dir_all(root)?;
        if fresh {
            no_cow(root);
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("lock"))?;
        match file.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                let holder = fs::read_to_string(root.join("lock"))
                    .ok()
                    .and_then(|text| text.trim().parse().ok());
                return Err(StoreError::Locked {
                    path: root.to_path_buf(),
                    holder,
                });
            }
            Err(fs::TryLockError::Error(error)) => return Err(error.into()),
        }
        // The holder's pid, so a refused second start can name it.
        file.set_len(0)?;
        file.write_all(format!("{}\n", std::process::id()).as_bytes())?;
        Ok(ArchiveLock { _file: file })
    }

    /// Private prepared output, cleared after the archive lock is taken on open.
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join("staging")
    }

    fn finish_open(&self) -> Result<()> {
        fs::create_dir_all(self.staging_dir())?;
        clear_directory(&self.staging_dir())?;
        let e = open_write(&self.embeddings_db(), cache::SETUP)?;
        schema::ensure_embeddings(&e)?;
        drop(e);
        self.prepare_index()?;
        crate::search::files::sweep(&self.root)
    }

    /// The caller holds the archive lock. An index is rebuilt in a private
    /// file and installed when its rebuild ends ([`index_build`]).
    pub(crate) fn prepare_index(&self) -> Result<()> {
        index_build::prepare(self)
    }
    /// Creates the index schema in an empty file or opens the one there.
    /// `building`: the file is a build, which leaves out the
    /// indexes it builds when it ends.
    pub(crate) fn ensure_index(&self, index: &Connection, building: bool) -> Result<()> {
        let fresh = schema::version(index)? == 0;
        if fresh {
            let mut reset = self.read_scope.begin_reset()?;
            reset.invalidate()?;
        }
        let archive = open_read(&self.archive_db())?;
        let _write = crate::write_lock::hold(&archive)?;
        if building {
            schema::ensure_build_index(index)?;
        } else {
            schema::ensure_index(index)?;
        }
        if fresh {
            self.read_scope.finish_reset();
        }
        Ok(())
    }
}

/// Deletes every entry in `dir`. Blob files are written there before they
/// are moved under their content names; only the lock holder writes there,
/// so when the archive opens anything in it is left from a stop.
fn clear_directory(dir: &Path) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let entry = entry?;
        let removed = if entry.file_type()?.is_dir() {
            fs::remove_dir_all(entry.path())
        } else {
            fs::remove_file(entry.path())
        };
        match removed {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// Turns copy-on-write off for a new, empty directory on btrfs, so files
/// created in it inherit the flag (SQLite and blob files fragment badly
/// under copy-on-write). Best effort: other filesystems ignore it.
pub(crate) fn no_cow(dir: &Path) {
    let _ = std::process::Command::new("chattr")
        .arg("+C")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// How long a connection waits for a lock another connection holds.
pub(crate) const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn tune(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    conn.pragma_update(None, "journal_size_limit", WAL_FILE_LIMIT)?;
    conn.pragma_update(None, "temp_store", "file")?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    enforce_references(conn)
}

/// Turns on the references and cascades the schemas declare. SQLite leaves
/// them off unless its build says otherwise, so each connection that writes
/// asks for them instead of relying on how the library was compiled.
pub(crate) fn enforce_references(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

/// Page-cache targets of write connections, in MiB. Process memory also includes
/// parsed values, queued batches, SQLite temporary work and allocator overhead.
pub mod cache {
    /// Default for callers without an archive allowance. Pipeline writers
    /// take their cache from the process's memory budget.
    pub const INDEX_WRITER: i64 = 384;
    /// Schema setup and one-off maintenance.
    pub const SETUP: i64 = 16;
}

/// Bytes of write-ahead log file kept after a checkpoint has copied the log
/// into the database: a restarted log is truncated to this, not left at the
/// size it once grew to.
const WAL_FILE_LIMIT: i64 = 64 << 20;

/// Pages of write-ahead log after which a write connection copies the log
/// into the database file: about 1 GB rather than 4 MB, so a page written by
/// many transactions reaches the database file once.
pub(crate) const WAL_AUTOCHECKPOINT: i64 = 262_144;

/// Starts the write-ahead log of `conn`'s database `schema` (at `path`) over
/// once it holds more than `limit` pages, waiting at most `wait` for readers
/// still reading the log. Returns whether the log was started over.
///
/// The automatic checkpoint copies the log into the database but leaves the
/// log in place while any reader uses it, and a steady stream of reads
/// (each commit wakes the readers that follow it) always has one: the log
/// then grows without bound, and every commit copies its pages again. This
/// checkpoint copies what is left, waits for the readers that started before
/// the copy finished (later readers read the database file), and truncates
/// the log. The writer calls it between its transactions; a log still in use
/// stays and is tried again after the next one.
pub(crate) fn restart_full_log(
    conn: &Connection,
    schema: &str,
    path: &Path,
    limit: i64,
    wait: std::time::Duration,
) -> Result<bool> {
    let mut log = path.as_os_str().to_owned();
    log.push("-wal");
    let size = match std::fs::metadata(&log) {
        Ok(metadata) => metadata.len(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let page: i64 = conn.scalar(&format!("PRAGMA {schema}.page_size"), [])?;
    if size <= limit.saturating_mul(page) as u64 {
        return Ok(false);
    }
    conn.busy_timeout(wait)?;
    let busy = conn.scalar::<i64>(&format!("PRAGMA {schema}.wal_checkpoint(TRUNCATE)"), []);
    conn.busy_timeout(BUSY_TIMEOUT)?;
    Ok(busy? == 0)
}

/// Prepared statements a write connection keeps.
const WRITER_STATEMENTS: usize = 512;

/// A connection the writer (or setup) uses. Each connection belongs to one
/// thread, so SQLite's own mutexes are off (they cost about 3% of the
/// writer's time).
pub fn open_write(path: &Path, cache_mib: i64) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    tune(&conn)?;
    crate::activity::trace_statements(&conn, crate::activity::Connection::Writer);
    crate::sql_functions::register_source_alias(&conn)?;
    writer_settings(&conn, cache_mib)?;
    conn.pragma_update(None, "wal_autocheckpoint", WAL_AUTOCHECKPOINT)?;
    crate::search::files::attach(&conn)?;
    Ok(conn)
}

/// The statement cache and page cache of a write connection.
pub(crate) fn writer_settings(conn: &Connection, cache_mib: i64) -> Result<()> {
    // The writer runs a few hundred distinct statements per batch; rusqlite
    // keeps 16 by default, so most were parsed again for every batch.
    conn.set_prepared_statement_cache_capacity(WRITER_STATEMENTS);
    conn.pragma_update(None, "cache_size", -(cache_mib * 1024))?;
    // Mapped pages count as resident memory and speed up reads only; the
    // writer's reads go through its page cache.
    conn.pragma_update(None, "mmap_size", 0)?;
    Ok(())
}

/// Pipeline writers checkpoint between turns. Closing them must not start a
/// potentially large WAL copy during shutdown; committed frames remain readable.
/// Setup connections keep their usual checkpoint-on-close behavior.
pub(crate) fn open_runtime_writer(path: &Path, cache_mib: i64) -> Result<Connection> {
    let conn = open_write(path, cache_mib)?;
    conn.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        true,
    )?;
    Ok(conn)
}

/// The writer's `archive.db` connection; its temporary tables spill to disk.
/// Its commits do not sync: every one goes through
/// [`crate::write_lock::commit`], which syncs the log after the commit is
/// visible, outside the readers' lock.
pub(crate) fn open_archive_writer(archive: &Archive) -> Result<Connection> {
    let conn = open_runtime_writer(
        &archive.archive_db(),
        archive.memory_budget().archive_cache_mib(),
    )?;
    conn.pragma_update(None, "temp_store", "file")?;
    conn.pragma_update(Some("temp"), "cache_size", -1024)?;
    crate::activity::trace_statements(&conn, crate::activity::Connection::ArchiveWriter);
    Ok(conn)
}

/// A read-only connection.
pub fn open_read(path: &Path) -> Result<Connection> {
    open_read_connection(path, false)
}

/// Private build readers may perform SQLite's journal recovery. Application
/// writes remain disabled by their caller; preserved databases use open_read.
fn open_read_connection(path: &Path, recover: bool) -> Result<Connection> {
    let access = if recover {
        OpenFlags::SQLITE_OPEN_READ_WRITE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };
    let conn = Connection::open_with_flags(
        path,
        access | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        true,
    )?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    crate::activity::trace_statements(&conn, crate::activity::Connection::Reader);
    if let Some(limit) = crate::read::limit::ReadLimit::current() {
        limit.install(&conn)?;
    }
    // One-off readers use the setup allocation; pools override both databases
    // from their aggregate budget after attaching archive.db.
    conn.pragma_update(None, "cache_size", -(cache::SETUP * 1024))?;
    conn.pragma_update(None, "mmap_size", 0)?;
    conn.pragma_update(None, "temp_store", "file")?;
    conn.pragma_update(Some("temp"), "cache_size", -1024)?;
    crate::sql_functions::register(&conn)?;
    crate::search::files::attach(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests;
