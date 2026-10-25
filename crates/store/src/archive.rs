//! The archive directory, its single-writer lock and connection settings.

pub(crate) mod generation;

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
}

/// Held by the daemon for as long as it runs. Only the holder writes.
#[derive(Debug)]
pub struct ArchiveLock {
    _file: File,
}

impl Archive {
    /// The register of inputs and of what people and agents wrote. In
    /// standalone mode it is `archive.db`, the archive's record; in primary
    /// mode it is `sinex.db`, a copy of what Sinex holds, written only after
    /// Sinex acknowledged it and rebuilt from Sinex when missing.
    pub fn archive_db(&self) -> PathBuf {
        match self.sinex_store {
            None => self.root.join("archive.db"),
            Some(_) => self.root.join("sinex.db"),
        }
    }

    /// Whether Sinex holds the archive.
    pub fn is_primary(&self) -> bool {
        self.sinex_store.is_some()
    }

    /// The content store the inputs' bytes are read from: this directory's
    /// `blobs/` (standalone), or Sinex's content store (primary), which
    /// Polylogue only reads.
    pub fn blobs(&self) -> Result<hashed_blobs::Store> {
        Ok(match &self.sinex_store {
            None => hashed_blobs::Store::open(self.root.clone(), vec![], 0o600)?,
            Some(store) => hashed_blobs::Store::open(store.clone(), vec![], 0o600)?,
        })
    }

    pub fn index_db(&self) -> PathBuf {
        self.root.join("index.db")
    }

    pub fn embeddings_db(&self) -> PathBuf {
        self.root.join("embeddings.db")
    }

    pub fn blobs_dir(&self) -> PathBuf {
        self.root.join("blobs")
    }

    /// Row dumps of tool databases: one
    /// append-only file per database, stored as an ordinary input.
    pub fn exports_dir(&self) -> PathBuf {
        self.root.join("exports")
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
        }
    }

    /// The archive at `root` held by the Sinex whose store root is `store`,
    /// without opening it.
    pub fn on_sinex(root: &Path, store: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            sinex_store: Some(store.to_path_buf()),
        }
    }

    /// The archive location: `POLYLOGUE_ARCHIVE` when set (tests and
    /// measurements), otherwise `$XDG_DATA_HOME/polylogue`.
    pub fn default_root() -> PathBuf {
        if let Some(dir) = std::env::var_os("POLYLOGUE_ARCHIVE") {
            return PathBuf::from(dir);
        }
        let data = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
            .unwrap_or_else(|| PathBuf::from("."));
        data.join("polylogue")
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
        let lock = Self::lock(root)?;
        let archive = Self::at(root);
        fs::create_dir_all(archive.blobs_dir())?;
        let a = open_write(&archive.archive_db(), cache::SETUP)?;
        let schema_lock = crate::write_lock::hold(&a)?;
        schema::ensure_archive(&a)?;
        drop(schema_lock);
        // Whatever a stop left of blob writes: files not yet moved under
        // their content names, rows still `writing` or `deleting`.
        let store = archive.blobs()?;
        let blob_lock = crate::write_lock::hold(&a)?;
        clear_directory(&archive.blobs_dir().join("tmp"))?;
        crate::blobs::settle(&a, &store)?;
        drop(blob_lock);
        archive.finish_open()?;
        Ok((archive, lock))
    }

    /// Opens the archive at `root` held by Sinex (primary mode), whose
    /// store root is `store`. A missing register is rebuilt from what Sinex
    /// holds, and the index with it; there is no local blob store.
    pub fn open_primary(
        root: &Path,
        store: &Path,
        sinex: &dyn crate::primary::Sinex,
    ) -> Result<(Self, ArchiveLock)> {
        let lock = Self::lock(root)?;
        let archive = Self::on_sinex(root, store);
        clear_directory(&archive.parts_dir())?;
        // The register is a copy of what Sinex holds: one with another
        // schema is rebuilt, not refused.
        if archive.archive_db().exists() {
            let register = open_read(&archive.archive_db())?;
            let current =
                schema::version(&register)? == 1 && schema::same_schema(&register, &schema::SINEX)?;
            drop(register);
            if !current {
                remove_database(&archive.archive_db())?;
            }
        }
        if !archive.archive_db().exists() {
            // The index describes the register it was built from.
            remove_database(&archive.index_db())?;
            let building = root.join("sinex.db.building");
            remove_database(&building)?;
            let mut a = open_write(&building, cache::SETUP)?;
            schema::ensure_sinex(&a)?;
            crate::primary::load(&mut a, sinex, &archive.blobs()?)?;
            drop(a);
            fs::rename(&building, archive.archive_db())?;
            File::open(root)?.sync_all()?;
        }
        let a = open_write(&archive.archive_db(), cache::SETUP)?;
        schema::ensure_sinex(&a)?;
        drop(a);
        archive.finish_open()?;
        Ok((archive, lock))
    }

    /// Where input bytes wait, in primary mode, until the writer sends them
    /// to Sinex; emptied when the archive opens.
    pub fn parts_dir(&self) -> PathBuf {
        self.root.join("parts")
    }

    fn lock(root: &Path) -> Result<ArchiveLock> {
        let fresh = !root.exists();
        fs::create_dir_all(root)?;
        if fresh {
            no_cow(root);
            let exports = root.join("exports");
            fs::create_dir_all(&exports)?;
            compressed(&exports);
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

    fn finish_open(&self) -> Result<()> {
        let e = open_write(&self.embeddings_db(), cache::SETUP)?;
        schema::ensure_embeddings(&e)?;
        drop(e);
        self.prepare_index()?;
        generation::open(self)
    }

    /// The caller holds the archive lock. An index built by another version
    /// or with another schema, left unsynced by a stop in the middle of a
    /// first build, or missing an excision the archive holds is deleted and
    /// created again empty; the loop then rebuilds it from the stored inputs.
    pub(crate) fn prepare_index(&self) -> Result<()> {
        let archive = open_read(&self.archive_db())?;
        let _write = crate::write_lock::hold(&archive)?;
        if self.index_db().exists() && self.index_out_of_date(&archive)? {
            remove_database(&self.index_db())?;
            File::open(&self.root)?.sync_all()?;
        }
        crate::fault::check("index.after_removal", "")?;
        let index = open_write(&self.index_db(), cache::SETUP)?;
        let fresh = schema::version(&index)? == 0;
        schema::ensure_index(&index)?;
        crate::fault::check("index.after_creation", "")?;
        if fresh {
            crate::excise::mark_applied(&archive, &index)?;
        }
        Ok(())
    }

    fn index_out_of_date(&self, archive: &Connection) -> Result<bool> {
        let index = open_read(&self.index_db())?;
        let found = schema::version(&index)?;
        if found == 0 {
            return Ok(false);
        }
        Ok(found != schema::INDEX_VERSION
            || !schema::same_schema(&index, &[schema::INDEX])?
            || schema::unsynced(&index)?
            || crate::excise::index_behind(archive, &index)?)
    }
}

/// Deletes a SQLite database and its journal files.
fn remove_database(path: &Path) -> Result<()> {
    for suffix in ["", "-wal", "-shm"] {
        let mut name = path.as_os_str().to_owned();
        name.push(suffix);
        match fs::remove_file(&name) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
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

/// Turns transparent compression on for the new, empty exports directory
/// on btrfs (clearing the copy-on-write flag it inherited, which would
/// forbid compression). Exports are append-only JSON Lines that the loop
/// also stores, compressed, as inputs; uncompressed they were 11 GB of a
/// 51 GB first build. Best effort: other filesystems ignore it.
fn compressed(dir: &Path) {
    // Two calls: btrfs refuses clearing one flag and setting the other at once.
    for flag in ["-C", "+c"] {
        let _ = std::process::Command::new("chattr")
            .arg(flag)
            .arg(dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

fn tune(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    conn.pragma_update(None, "temp_store", "memory")?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
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
    /// The writer's `archive.db` connection: inputs, revisions, extents.
    pub const ARCHIVE_WRITER: i64 = 256;
    /// Schema setup and one-off maintenance.
    pub const SETUP: i64 = 16;
}

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
    conn.pragma_update(None, "cache_size", -(cache_mib * 1024))?;
    // Mapped pages count as resident memory and speed up reads only; the
    // writer's reads go through its page cache.
    conn.pragma_update(None, "mmap_size", 0)?;
    // Checkpoint after about 1 GB of WAL rather than 4 MB, so a page written
    // by many transactions reaches the database file once.
    conn.pragma_update(None, "wal_autocheckpoint", 262_144)?;
    Ok(conn)
}

/// The writer's `archive.db` connection; its temporary tables spill to disk.
pub(crate) fn open_archive_writer(archive: &Archive) -> Result<Connection> {
    let conn = open_write(&archive.archive_db(), cache::ARCHIVE_WRITER)?;
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
    // One-off readers use the setup allocation; pools override both databases
    // from their aggregate budget after attaching archive.db.
    conn.pragma_update(None, "cache_size", -(cache::SETUP * 1024))?;
    conn.pragma_update(None, "mmap_size", 0)?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    crate::sql_functions::register(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests;
