//! The archive directory and its single-writer lock.

use crate::Result;
use std::path::{Path, PathBuf};

/// An opened archive directory (`$XDG_DATA_HOME/polylogue`). The directory
/// has copy-on-write turned off on btrfs; Polylogue checks its own data
/// with BLAKE3 for blobs and SQLite's checksum VFS for the databases.
pub struct Archive {
    pub root: PathBuf,
}

/// Held by the daemon for as long as it runs. Only the holder writes.
pub struct ArchiveLock {
    _file: std::fs::File,
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

    /// Creates the directory with copy-on-write off and the three databases
    /// at their current versions, or opens an existing one and applies
    /// archive migrations.
    pub fn open_or_create(_root: &Path) -> Result<(Self, ArchiveLock)> {
        // Architecture 4.1: No_COW directory, lock, schema versions.
        todo!()
    }
}
