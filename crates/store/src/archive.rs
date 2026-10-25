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
}

/// Held by the daemon for as long as it runs. Only the holder writes.
#[derive(Debug)]
pub struct ArchiveLock {
    _file: File,
}

impl Archive {
    pub fn archive_db(&self) -> PathBuf {
        self.root.join("archive.db")
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

    /// Logical exports of tool databases: one
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
    /// at their current versions, or opens an existing one. Takes the writer lock; a second daemon on the
    /// same archive is refused.
    pub fn open_or_create(root: &Path) -> Result<(Self, ArchiveLock)> {
        let fresh = !root.exists();
        fs::create_dir_all(root)?;
        if fresh {
            no_cow(root);
            let exports = root.join("exports");
            fs::create_dir_all(&exports)?;
            compressed(&exports);
        }
        let lock_path = root.join("lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        if file.try_lock().is_err() {
            return Err(StoreError::Locked {
                path: root.to_path_buf(),
            });
        }
        let archive = Self::at(root);
        fs::create_dir_all(archive.blobs_dir())?;
        let a = open_write(&archive.archive_db(), cache::SETUP)?;
        let identity_lock = crate::write_lock::hold(&a)?;
        schema::ensure_archive(&a)?;
        // The archive's identity: Sinex keys this archive's published rows
        // by it, so two archives never collide.
        a.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('archive_id', ?1)",
            [uuid::Uuid::now_v7().to_string()],
        )?;
        drop(identity_lock);
        let e = open_write(&archive.embeddings_db(), cache::SETUP)?;
        schema::ensure_embeddings(&e)?;
        drop(e);
        // Blob files an excision marked for deletion before a stop.
        let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600)?;
        crate::blobs::delete_marked(&a, &store)?;
        archive.prepare_index()?;
        generation::open(&archive)?;
        Ok((archive, ArchiveLock { _file: file }))
    }

    /// The caller holds the archive lock. An index built by another version,
    /// left unsynced by a stop in the middle of a first build, or missing an
    /// excision the archive holds is deleted and created again empty; the
    /// loop then rebuilds it from the stored inputs.
    pub(crate) fn prepare_index(&self) -> Result<()> {
        let archive = open_read(&self.archive_db())?;
        let _write = crate::write_lock::hold(&archive)?;
        if self.index_db().exists() && self.index_out_of_date(&archive)? {
            for suffix in ["", "-wal", "-shm"] {
                match fs::remove_file(self.root.join(format!("index.db{suffix}"))) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
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
            || schema::unsynced(&index)?
            || crate::excise::index_behind(archive, &index)?)
    }
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
