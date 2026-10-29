//! Inputs and chains over immutable shared segment extents.

use crate::Result;
use polylogue_model::BlobHash;
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;

mod chain;
pub use chain::{ChainReader, StoredChain};

mod copy;
pub(crate) use copy::copy_plain;

mod segments;
pub use segments::Blobs;
pub(crate) use segments::{insert_extent, load_extent, open_writer};

/// The read operations used by stored chains.
pub trait PlainBlobs {
    fn open_plain(&self, hash: &hashed_blobs::Hash, size: u64) -> Result<hashed_blobs::BlobReader>;
    fn verify_plain(
        &self,
        hash: &hashed_blobs::Hash,
        size: u64,
        visit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<hashed_blobs::VerificationStats>;
}
impl PlainBlobs for Blobs {
    fn open_plain(&self, hash: &hashed_blobs::Hash, size: u64) -> Result<hashed_blobs::BlobReader> {
        self.read_plain(hash, Some(size), || Ok(()))
    }
    fn verify_plain(
        &self,
        hash: &hashed_blobs::Hash,
        size: u64,
        visit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<hashed_blobs::VerificationStats> {
        self.verify_plain(hash, size, visit)
    }
}

/// zstd level for stored bytes: about 300 MB/s per core.
pub const LEVEL: i32 = 3;

/// What one look at a file found. Equal facts mean nothing is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileFacts {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

impl FileFacts {
    pub fn of(meta: &Metadata) -> Self {
        let ns = |s: i64, n: i64| i128::from(s) * 1_000_000_000 + i128::from(n);
        Self {
            device: meta.dev(),
            inode: meta.ino(),
            size: meta.size(),
            mtime_ns: ns(meta.mtime(), meta.mtime_nsec()),
            ctime_ns: ns(meta.ctime(), meta.ctime_nsec()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Never seen before: the input's first revision.
    First,
    /// Same facts as last time: nothing is read.
    Unchanged,
    /// Parse bytes already stored in this exact chain.
    Reparse { chain: i64 },
    /// The same file grew: only the bytes from `from` on are new. The caller
    /// confirms the old tail is intact before trusting this.
    Appended { from: u64 },
    /// Content changed in place (a head rewrite or a full rewrite), or the
    /// path now names another file: a new revision chain.
    Rewritten,
    /// The file is gone. Recorded as an observation, never a delete.
    Disappeared,
}

/// Compares a file's facts with its last observation. Pure: no IO.
pub fn classify(last: Option<FileFacts>, now: Option<FileFacts>) -> Change {
    match (last, now) {
        (None, None) => Change::Unchanged,
        (Some(_), None) => Change::Disappeared,
        (None, Some(_)) => Change::First,
        (Some(l), Some(n)) if l == n => Change::Unchanged,
        (Some(l), Some(n)) if l.device == n.device && l.inode == n.inode && n.size > l.size => {
            Change::Appended { from: l.size }
        }
        (Some(l), Some(n)) if l.device == n.device && l.inode == n.inode && n.size == l.size => {
            // Same size, new times: touched or rewritten in place. Only a
            // rewrite changes bytes; the caller compares the tail window.
            Change::Rewritten
        }
        _ => Change::Rewritten,
    }
}

/// The identity of prepared bytes. The writer commits their segment extent
/// together with every archive row that cites them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewBlob {
    pub hash: BlobHash,
    pub size: u64,
}

/// Read a committed blob with the shared frame checks.
pub fn read(archive: &rusqlite::Connection, store: &Blobs, hash: &BlobHash) -> Result<Vec<u8>> {
    let extent = load_extent(archive, "", &hashed_blobs::Hash(hash.0))?
        .ok_or_else(|| crate::StoreError::NotFound(format!("blob {hash}")))?;
    Ok(hashed_blobs::Reader::new(store.root()).read(&[extent], None)?)
}

#[cfg(test)]
mod tests;
