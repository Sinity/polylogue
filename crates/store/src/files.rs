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

    /// Creates the directory with copy-on-write off and the three databases
    /// at their current versions, or opens an existing one and applies
    /// archive migrations. Takes the writer lock; a second daemon on the
    /// same archive is refused.
    pub fn open_or_create(root: &Path) -> Result<(Self, ArchiveLock)> {
        let fresh = !root.exists();
        fs::create_dir_all(root)?;
        if fresh {
            no_cow(root);
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
        let mut a = open_write(&archive.archive_db())?;
        if schema::version(&a)? == 0 {
            schema::create_archive(&a)?;
        }
        schema::migrate_archive(&mut a)?;
        set_aside_old_index(&archive)?;
        let i = open_write(&archive.index_db())?;
        schema::ensure_index(&i)?;
        let e = open_write(&archive.embeddings_db())?;
        if schema::version(&e)? == 0 {
            schema::create_embeddings(&e)?;
        }
        Ok((archive, ArchiveLock { _file: file }))
    }
}

/// An index built by another version is moved aside (`index.db.v<N>`), so
/// a new one is built from `archive.db` by the loop. Nothing in it is
/// lost: every row is recomputed from stored bytes.
fn set_aside_old_index(archive: &Archive) -> Result<()> {
    let path = archive.index_db();
    if !path.exists() {
        return Ok(());
    }
    let found = schema::version(&open_read(&path)?)?;
    if found == 0 || found == schema::INDEX_VERSION {
        return Ok(());
    }
    for suffix in ["", "-wal", "-shm"] {
        let from = archive.root.join(format!("index.db{suffix}"));
        if from.exists() {
            fs::rename(&from, archive.root.join(format!("index.db.v{found}{suffix}")))?;
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

fn tune(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "wal")?;
    conn.pragma_update(None, "synchronous", "normal")?;
    conn.pragma_update(None, "temp_store", "memory")?;
    // A large page cache: the writer and the readers run against tables in
    // the hundreds of MB, and the operator allows a few GB of RAM.
    conn.pragma_update(None, "cache_size", -262_144)?;
    conn.pragma_update(None, "mmap_size", 1_i64 << 32)?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    Ok(())
}

/// A connection the writer (or setup) uses.
pub fn open_write(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    tune(&conn)?;
    Ok(conn)
}

/// A read-only connection.
pub fn open_read(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI,
    )?;
    conn.pragma_update(None, "cache_size", -65_536)?;
    conn.pragma_update(None, "mmap_size", 1_i64 << 32)?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
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
}
