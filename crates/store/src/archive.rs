//! The archive directory, its single-writer lock and connection settings.

pub(crate) mod generation;
pub mod index_rebuild;
mod metadata;
pub use metadata::Mode;

use crate::{Result, StoreError, schema};
use rusqlite::{Connection, OpenFlags, OptionalExtension, Transaction};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

/// Increments the database's write counter; one value per transaction.
pub(crate) fn bump_counter(tx: &Transaction<'_>) -> Result<i64> {
    let next: i64 = tx
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'write_counter'",
            [],
            |r| r.get(0),
        )
        .optional()?
        .unwrap_or(0_i64)
        .checked_add(1)
        .ok_or_else(|| StoreError::Invalid("write counter exhausted".into()))?;
    tx.execute(
        "INSERT INTO meta(key, value) VALUES ('write_counter', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        [next.to_string()],
    )?;
    Ok(next)
}

/// An opened archive directory (`$XDG_DATA_HOME/polylogue`). The directory
/// has copy-on-write turned off on btrfs; Polylogue checks its own data
/// with BLAKE3 for blobs. SQLite's checksum VFS is not wired in yet: it
/// needs the `cksumvfs` extension, which the bundled build does not ship.
#[derive(Debug, Clone)]
pub struct Archive {
    pub root: PathBuf,
    /// In primary mode, Sinex's store root: its content store holds the
    /// inputs' bytes, and the register is a cache of what Sinex holds.
    sinex_store: Option<PathBuf>,
    primary: bool,
    pub(crate) read_scope: std::sync::Arc<crate::read::scope::ReadScopeState>,
}

/// Held by the daemon for as long as it runs. Only the holder writes.
#[derive(Debug)]
pub struct ArchiveLock {
    _file: File,
}

impl Archive {
    /// Resolve existing ancestors even before this archive's directory exists.
    pub fn resolved_root(root: &Path) -> PathBuf {
        crate::read::scope::root_key(root)
    }

    /// Register while holding writer custody of a live shared archive. Startup
    /// uses the open-with-invalidator constructors to acquire that custody first.
    /// The callback runs immediately and at each reset, under the publication fence;
    /// it must durably invalidate its pointer without acquiring archive readers
    /// or re-entering the scope fence.
    pub fn register_scope_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<crate::read::scope::ScopeInvalidation> {
        crate::read::scope::state(root).register(invalidate)
    }

    /// A transition guard without SQL. Currentness still comes from a reader's
    /// confirmed checkpoint; this token alone cannot authorize cache activation.
    pub fn scope_token(&self) -> Result<crate::ReadScopeToken> {
        self.read_scope.token()
    }

    /// Exercise the native generation transition in synthetic transport tests.
    /// This changes cursor identity only; primary grant resets use confirmed facts.
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

    /// Where source rows are read: the standalone ledger or the primary index cache.
    pub fn archive_db(&self) -> PathBuf {
        if self.is_primary() {
            self.index_db()
        } else {
            self.root.join("archive.db")
        }
    }

    /// Whether Sinex holds the archive.
    pub fn is_primary(&self) -> bool {
        self.primary
    }

    /// Resolve content hashes in Polylogue's committed catalog and read the
    /// immutable segment extents with the shared reader.
    pub fn blobs(&self) -> Result<crate::blobs::Blobs> {
        self.blob_reader()
    }

    pub fn blob_reader(&self) -> Result<crate::blobs::Blobs> {
        let root = self.byte_root()?;
        let (catalog, layout) = if self.is_primary() {
            (self.index_db(), crate::blobs::BlobCatalog::Projection)
        } else {
            (self.archive_db(), crate::blobs::BlobCatalog::Archive)
        };
        Ok(crate::blobs::Blobs::new(
            root.to_path_buf(),
            catalog,
            layout,
        ))
    }

    pub fn index_db(&self) -> PathBuf {
        self.root.join("index.db")
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
            root: root.to_path_buf(),
            sinex_store: None,
            primary: false,
            read_scope: crate::read::scope::state(root),
        }
    }

    /// The archive at `root` held by the Sinex whose store root is `store`,
    /// without opening it.
    pub fn on_sinex(root: &Path, store: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            sinex_store: Some(store.to_path_buf()),
            primary: true,
            read_scope: crate::read::scope::state(root),
        }
    }

    /// The primary cache remains readable when configuration cannot name its byte owner.
    pub fn open_primary_cache(root: &Path) -> Result<(Self, ArchiveLock)> {
        let (archive, lock, ()) = Self::open_configured(Self::primary_cache(root), |_| Ok(()))?;
        Ok((archive, lock))
    }

    pub(crate) fn byte_root(&self) -> Result<&Path> {
        if self.is_primary() {
            self.sinex_store
                .as_deref()
                .ok_or_else(|| StoreError::Sinex("the primary byte owner is unavailable".into()))
        } else {
            Ok(&self.root)
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

    /// Opens the single primary cache. Only the confirmed consumer fills it.
    pub fn open_primary(root: &Path, store: &Path) -> Result<(Self, ArchiveLock)> {
        let (archive, lock, ()) = Self::open_configured(Self::on_sinex(root, store), |_| Ok(()))?;
        Ok((archive, lock))
    }

    /// Acquire writer custody before registering cache invalidation and opening
    /// the standalone archive. Retain the returned registration with the lock.
    pub fn open_or_create_with_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        Self::open_configured(Self::at(root), |archive| {
            archive.read_scope.register(invalidate)
        })
    }

    /// Open a primary cache with invalidation installed before generation setup.
    pub fn open_primary_with_invalidator(
        root: &Path,
        store: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        Self::open_configured(Self::on_sinex(root, store), |archive| {
            archive.read_scope.register(invalidate)
        })
    }

    /// Install invalidation while opening a cache whose byte owner is unavailable.
    pub fn open_primary_cache_with_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        Self::open_configured(Self::primary_cache(root), |archive| {
            archive.read_scope.register(invalidate)
        })
    }

    fn primary_cache(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            sinex_store: None,
            primary: true,
            read_scope: crate::read::scope::state(root),
        }
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
        if !archive.is_primary() {
            fs::create_dir_all(archive.blobs_dir())?;
            let source = open_write(&archive.archive_db(), cache::SETUP)?;
            let schema_lock = crate::write_lock::hold(&source)?;
            schema::ensure_archive(&source)?;
            drop(schema_lock);
        }
        archive.finish_open()?;
        Ok((archive, lock, setup))
    }

    /// The archive's durable identity, read without opening or creating a database.
    pub fn recorded_mode(root: &Path) -> Result<Option<Mode>> {
        metadata::read(root)
    }

    /// Broken deployment settings pause capture and writes while the recorded
    /// archive remains readable. They never choose another storage mode.
    pub fn open_blocked_with_invalidator(
        root: &Path,
        invalidate: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<(Self, ArchiveLock, crate::ScopeInvalidation)> {
        let archive = match Self::recorded_mode(root)? {
            Some(Mode::Standalone) => Self::at(root),
            Some(Mode::Primary) => Self::primary_cache(root),
            None => {
                return Err(StoreError::Invalid(format!(
                    "archive {} has no recorded mode; fix its configuration before creating it",
                    root.display()
                )));
            }
        };
        Self::open_configured(archive, |archive| archive.read_scope.register(invalidate))
    }

    /// Where input bytes wait, in primary mode, until the writer sends them
    /// to Sinex; emptied when the archive opens.
    pub fn parts_dir(&self) -> Result<PathBuf> {
        if self.is_primary() {
            Ok(self.byte_root()?.join("polylogue-prepared").join(
                blake3::hash(self.root.as_os_str().as_encoded_bytes())
                    .to_hex()
                    .to_string(),
            ))
        } else {
            Ok(self.root.join("parts"))
        }
    }

    fn lock(root: &Path) -> Result<ArchiveLock> {
        let fresh = !root.exists();
        fs::create_dir_all(root)?;
        if fresh {
            no_cow(root);
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join("lock"))?;
        if file.try_lock().is_err() {
            return Err(StoreError::Locked {
                path: root.to_path_buf(),
            });
        }
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
        generation::open(self)
    }

    /// The caller holds the archive lock. Parse positions in the index
    /// resume an interrupted build. Unknown current shapes refuse.
    /// An index from another format
    /// is set aside under its schema identity and a new one is built from the
    /// archive's stored bytes; the set-aside file is removed once that first
    /// build ends ([`index_rebuild`]).
    pub(crate) fn prepare_index(&self) -> Result<()> {
        let path = self.index_db();
        let mut index = open_runtime_writer(&path, cache::SETUP)?;
        if let Err(error) = self.ensure_index(&index) {
            let Some(reason) = index_rebuild::reason(&error) else {
                return Err(error);
            };
            let previous = schema::index_identity(&index)?;
            drop(index);
            index_rebuild::set_aside(&path, &previous)?;
            crate::fault::check("index.after_set_aside", &path.to_string_lossy())?;
            index = open_write(&path, cache::SETUP)?;
            self.ensure_index(&index)?;
            index_rebuild::record(
                &index,
                reason,
                previous,
                schema::build_index_identity(self.is_primary())?,
            )?;
        }
        index_rebuild::remove_when_built(&index, &path, self.is_primary())?;
        crate::fault::check("index.after_creation", "")?;
        Ok(())
    }

    /// Creates the index schema in an empty file or opens the one there.
    fn ensure_index(&self, index: &Connection) -> Result<()> {
        let fresh = schema::version(index)? == 0;
        if fresh {
            let mut reset = self.read_scope.begin_reset()?;
            reset.invalidate()?;
        }
        if self.is_primary() {
            schema::ensure_primary_index(index)?;
        } else {
            let archive = open_read(&self.archive_db())?;
            let _write = crate::write_lock::hold(&archive)?;
            schema::ensure_index(index)?;
        }
        if fresh {
            self.read_scope.finish_reset();
        }
        Ok(())
    }
}

/// Deletes every file in `dir`. Blob files are written there before they
/// are moved under their content names; only the lock holder writes there,
/// so when the archive opens anything in it is left from a stop.
fn clear_directory(dir: &Path) -> Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        match fs::remove_file(entry?.path()) {
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
fn no_cow(dir: &Path) {
    let _ = std::process::Command::new("chattr")
        .arg("+C")
        .arg(dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// How long a connection waits for a lock another connection holds.
const BUSY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

fn tune(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    conn.pragma_update(None, "temp_store", "memory")?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    Ok(())
}

/// Page-cache targets of write connections, in MiB. Process memory also includes
/// parsed values, queued batches, SQLite temporary work and allocator overhead.
pub mod cache {
    /// The writer's `index.db` connection. It holds the indexes probed on
    /// every insert (text hashes, message IDs) and the search segments it
    /// merges; at 256 MB a first build read back from disk more than twice
    /// its input.
    pub const INDEX_WRITER: i64 = 768;
    /// The search thread's `index.db` connection (writer.rs, `Search`).
    pub const SEARCH: i64 = 64;
    /// The derive thread's `index.db` connection (writer.rs, `DeriveThread`):
    /// it reads one session at a time.
    pub const DERIVE: i64 = 64;
    /// The writer's `archive.db` connection: inputs, revisions, extents.
    pub const ARCHIVE_WRITER: i64 = 256;
    /// Schema setup and one-off maintenance.
    pub const SETUP: i64 = 16;
}

/// Pages of write-ahead log after which a write connection copies the log
/// into the database file: about 1 GB rather than 4 MB, so a page written by
/// many transactions reaches the database file once.
pub(crate) const WAL_AUTOCHECKPOINT: i64 = 262_144;

/// Starts the write-ahead log of `conn`'s main database (at `path`) over once
/// it holds more than `limit` pages, waiting at most `wait` for readers still
/// reading the log. Returns whether the log was started over.
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
    let page: i64 = conn.query_row("PRAGMA main.page_size", [], |r| r.get(0))?;
    if size <= limit.saturating_mul(page) as u64 {
        return Ok(false);
    }
    conn.busy_timeout(wait)?;
    let busy = conn.query_row("PRAGMA main.wal_checkpoint(TRUNCATE)", [], |r| {
        r.get::<_, i64>(0)
    });
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
    // The writer runs a few hundred distinct statements per batch; rusqlite
    // keeps 16 by default, so most were parsed again for every batch.
    conn.set_prepared_statement_cache_capacity(WRITER_STATEMENTS);
    conn.pragma_update(None, "cache_size", -(cache_mib * 1024))?;
    // Mapped pages count as resident memory and speed up reads only; the
    // writer's reads go through its page cache.
    conn.pragma_update(None, "mmap_size", 0)?;
    conn.pragma_update(None, "wal_autocheckpoint", WAL_AUTOCHECKPOINT)?;
    Ok(conn)
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
    let conn = open_runtime_writer(&archive.archive_db(), cache::ARCHIVE_WRITER)?;
    conn.pragma_update(None, "temp_store", "file")?;
    conn.pragma_update(Some("temp"), "cache_size", -1024)?;
    Ok(conn)
}

/// A read-only connection.
pub fn open_read(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        true,
    )?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    if let Some(limit) = crate::read::limit::ReadLimit::current() {
        limit.install(&conn)?;
    }
    // One-off readers use the setup allocation; pools override both databases
    // from their aggregate budget after attaching archive.db.
    conn.pragma_update(None, "cache_size", -(cache::SETUP * 1024))?;
    conn.pragma_update(None, "mmap_size", 0)?;
    crate::sql_functions::register(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests;
