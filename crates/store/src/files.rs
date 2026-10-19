//! The archive directory, its single-writer lock and connection settings.

use crate::{Result, StoreError, schema};
use rusqlite::{Connection, OpenFlags};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

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

    /// Logical exports of tool databases (architecture 4.2): one
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
    /// at their current versions, or opens an existing one and applies
    /// archive migrations. Takes the writer lock; a second daemon on the
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
        let mut a = open_write(&archive.archive_db(), cache::SETUP)?;
        if schema::version(&a)? == 0 {
            schema::create_archive(&a)?;
        }
        schema::migrate_archive(&mut a)?;
        // The archive's identity: Sinex keys this archive's published rows
        // by it, so two archives never collide (architecture 7.1).
        a.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('archive_id', ?1)",
            [uuid::Uuid::now_v7().to_string()],
        )?;
        set_aside_old_index(&archive)?;
        let i = open_write(&archive.index_db(), cache::SETUP)?;
        schema::ensure_index(&i)?;
        let e = open_write(&archive.embeddings_db(), cache::SETUP)?;
        if schema::version(&e)? == 0 {
            schema::create_embeddings(&e)?;
        }
        Ok((archive, ArchiveLock { _file: file }))
    }
}

/// An index built by another version, or left unsynced by a stop in the
/// middle of a first build, is moved aside (`index.db.v<N>`,
/// `index.db.unsynced`), so a new one is built from `archive.db` by the
/// loop. Nothing in it is lost: every row is recomputed from stored bytes.
fn set_aside_old_index(archive: &Archive) -> Result<()> {
    let path = archive.index_db();
    if !path.exists() {
        return Ok(());
    }
    let conn = open_read(&path)?;
    let found = schema::version(&conn)?;
    let unsynced = found == schema::INDEX_VERSION && schema::unsynced(&conn)?;
    drop(conn);
    let suffix_for = if unsynced {
        "unsynced".to_owned()
    } else if found == 0 || found == schema::INDEX_VERSION {
        return Ok(());
    } else {
        format!("v{found}")
    };
    for suffix in ["", "-wal", "-shm"] {
        let from = archive.root.join(format!("index.db{suffix}"));
        let to = archive.root.join(format!("index.db.{suffix_for}{suffix}"));
        if from.exists() {
            if to.exists() {
                fs::remove_file(&to)?;
            }
            fs::rename(&from, to)?;
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

/// Page-cache budgets of the write connections, in MiB. Together with the
/// parse pool's bytes in flight they are the daemon's memory: the operator
/// allows a few GB, and a first build stays within about 2.5 GB.
pub mod cache {
    /// The writer's `index.db` connection. It holds the indexes probed on
    /// every insert (text hashes, message IDs) and the search segments it
    /// merges; at 256 MB a first build read back from disk more than twice
    /// its input.
    pub const INDEX_WRITER: i64 = 1536;
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

/// A read-only connection.
pub fn open_read(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.pragma_update(None, "cache_size", -65_536)?;
    conn.pragma_update(None, "mmap_size", 1_i64 << 32)?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    crate::functions::register(&conn)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_open_is_refused_while_the_lock_is_held() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("a");
        let (_archive, _lock) = Archive::open_or_create(&root).unwrap();
        assert!(matches!(
            Archive::open_or_create(&root),
            Err(StoreError::Locked { .. })
        ));
        drop(_lock);
        Archive::open_or_create(&root).unwrap();
    }

    #[test]
    fn an_index_left_unsynced_is_set_aside_and_rebuilt() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("a");
        let (archive, lock) = Archive::open_or_create(&root).unwrap();
        let writer = crate::Writer::start(&archive).unwrap();
        writer.flush().unwrap();
        drop(writer);
        // A stop in the middle of a first build leaves the mark set.
        open_write(&archive.index_db(), cache::SETUP)
            .unwrap()
            .execute("UPDATE meta SET value = '1' WHERE key = 'unsynced'", [])
            .unwrap();
        drop(lock);
        let (archive, _lock) = Archive::open_or_create(&root).unwrap();
        assert!(root.join("index.db.unsynced").exists());
        assert!(!schema::unsynced(&open_read(&archive.index_db()).unwrap()).unwrap());
    }
}
