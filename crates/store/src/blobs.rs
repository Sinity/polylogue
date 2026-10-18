//! The blob layer: inputs, revisions and extents over `hashed-blobs`
//! (architecture 4.2). A blob under 128 KiB lives inline in `archive.db`,
//! zstd-compressed; larger ones are files of seekable zstd frames. Hashing
//! and compression run on the parse pool ([`Piece::prepare`]); the writer
//! only records rows, after the bytes are durable.

use crate::Result;
use polylogue_model::BlobHash;
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;

pub const INLINE_LIMIT: usize = 128 * 1024;

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

/// How a piece's bytes are stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stored {
    /// zstd-compressed bytes for the `blob` row in `archive.db`.
    Inline(Vec<u8>),
    /// Already durable under `blobs/`.
    File { stored_size: u64 },
}

/// Bytes of one revision step (an append, or a whole new content), hashed
/// and stored or compressed off the writer thread.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub hash: BlobHash,
    pub size: u64,
    pub stored: Stored,
}

impl Piece {
    /// Hashes `bytes`; compresses small ones for an inline row and writes
    /// large ones to the blob store, durable before this returns, so any row
    /// the writer adds later refers to bytes that exist.
    pub fn prepare(store: &hashed_blobs::Store, bytes: &[u8]) -> Result<Self> {
        let hash = BlobHash::of(bytes);
        let size = bytes.len() as u64;
        let stored = if bytes.len() < INLINE_LIMIT {
            Stored::Inline(zstd::bulk::compress(bytes, LEVEL)?)
        } else {
            let h = store.put(bytes, LEVEL)?;
            debug_assert_eq!(h.0, hash.0);
            Stored::File {
                stored_size: store.stored_size(&h)?,
            }
        };
        Ok(Self { hash, size, stored })
    }
}

impl Piece {
    /// Like [`Piece::prepare`], but a large blob goes into `batch` and is
    /// durable only once the batch commits. The caller commits before any
    /// row referring to the piece is sent, then calls [`Piece::settle`].
    /// One commit for many blobs costs two filesystem syncs instead of
    /// three per blob, which on a busy QLC drive is most of the cost.
    pub fn prepare_in(batch: &mut hashed_blobs::Batch<'_>, bytes: &[u8]) -> Result<Self> {
        let hash = BlobHash::of(bytes);
        let size = bytes.len() as u64;
        let stored = if bytes.len() < INLINE_LIMIT {
            Stored::Inline(zstd::bulk::compress(bytes, LEVEL)?)
        } else {
            batch.put(bytes, LEVEL)?;
            Stored::File { stored_size: 0 }
        };
        Ok(Self { hash, size, stored })
    }

    /// Records a committed file blob's stored size.
    pub fn settle(&mut self, store: &hashed_blobs::Store) -> Result<()> {
        if let Stored::File { stored_size } = &mut self.stored {
            *stored_size = store.stored_size(&hashed_blobs::Hash(self.hash.0))?;
        }
        Ok(())
    }
}

/// Reads a stored blob back whole: inline rows decompress, files go
/// through `hashed-blobs`, which checks the name.
pub fn read(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    hash: &BlobHash,
) -> Result<Vec<u8>> {
    use std::io::Read;
    let row: Option<(String, Option<Vec<u8>>, i64)> = archive
        .query_row(
            "SELECT place, data, size FROM blob WHERE hash = ?1",
            [&hash.0[..]],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map(Some)
        .or_else(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => Ok(None),
            e => Err(e),
        })?;
    match row {
        Some((place, Some(data), size)) if place == "inline" => {
            let bytes = zstd::bulk::decompress(&data, size as usize)?;
            if BlobHash::of(&bytes) != *hash {
                return Err(crate::StoreError::BlobCorrupt(*hash));
            }
            Ok(bytes)
        }
        _ => {
            let mut out = Vec::new();
            store
                .read(&hashed_blobs::Hash(hash.0))?
                .read_to_end(&mut out)?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(inode: u64, size: u64, mtime: i128) -> Option<FileFacts> {
        Some(FileFacts {
            device: 1,
            inode,
            size,
            mtime_ns: mtime,
            ctime_ns: mtime,
        })
    }

    #[test]
    fn classifies_every_kind_of_change() {
        assert_eq!(classify(None, facts(1, 10, 1)), Change::First);
        assert_eq!(
            classify(facts(1, 10, 1), facts(1, 10, 1)),
            Change::Unchanged
        );
        assert_eq!(
            classify(facts(1, 10, 1), facts(1, 25, 2)),
            Change::Appended { from: 10 }
        );
        assert_eq!(classify(facts(1, 10, 1), facts(1, 5, 2)), Change::Rewritten);
        assert_eq!(
            classify(facts(1, 10, 1), facts(2, 30, 2)),
            Change::Rewritten
        );
        assert_eq!(classify(facts(1, 10, 1), None), Change::Disappeared);
        assert_eq!(classify(None, None), Change::Unchanged);
    }

    #[test]
    fn small_pieces_are_inline_and_large_ones_files() {
        let dir = tempfile::tempdir().unwrap();
        let store = hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap();
        let small = Piece::prepare(&store, b"hello").unwrap();
        assert!(matches!(small.stored, Stored::Inline(_)));
        let big = vec![7u8; INLINE_LIMIT + 1];
        let large = Piece::prepare(&store, &big).unwrap();
        assert!(matches!(large.stored, Stored::File { .. }));
        assert!(store.contains(&hashed_blobs::Hash(large.hash.0)));
    }
}
