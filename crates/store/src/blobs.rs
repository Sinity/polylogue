//! The blob layer: inputs, revisions and extents over `hashed-blobs`
//! (architecture 4.2). A blob under 128 KiB lives inline in `archive.db`,
//! zstd-compressed; larger ones are files of seekable zstd frames. Hashing
//! and compression run before Writer admission. The Writer reserves canonical
//! ownership before installing private file stages, then finalizes completion.

use crate::Result;
use polylogue_model::BlobHash;
use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;

mod chain;
pub use chain::{ChainReader, StoredChain};

mod copy;
pub(crate) use copy::copy_plain;

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
    /// Interpret already retained bytes in this exact chain.
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
    /// Store an unknown-length stream without retaining its whole payload.
    pub(crate) fn prepare_reader(
        store: &hashed_blobs::Store,
        mut reader: impl std::io::Read,
    ) -> Result<Self> {
        use std::io::Read;
        let mut prefix = Vec::new();
        reader
            .by_ref()
            .take(INLINE_LIMIT as u64)
            .read_to_end(&mut prefix)?;
        if prefix.len() < INLINE_LIMIT {
            return Self::prepare(store, &prefix);
        }
        struct Count<R> {
            reader: R,
            bytes: u64,
        }
        impl<R: Read> Read for Count<R> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = self.reader.read(out)?;
                self.bytes = self
                    .bytes
                    .checked_add(n as u64)
                    .ok_or_else(|| std::io::Error::other("blob length overflow"))?;
                Ok(n)
            }
        }
        let mut reader = Count {
            reader: prefix.as_slice().chain(reader),
            bytes: 0,
        };
        let hash = store.put(&mut reader, LEVEL)?;
        Ok(Self {
            hash: BlobHash(hash.0),
            size: reader.bytes,
            stored: Stored::File {
                stored_size: store.stored_size(&hash)?,
            },
        })
    }

    /// Hashes `bytes`; compresses small ones for an inline row and writes
    /// large ones to the blob store, durable before this returns, so any row
    /// the writer adds later refers to bytes that exist.
    pub(crate) fn prepare(store: &hashed_blobs::Store, bytes: &[u8]) -> Result<Self> {
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

/// Marks a blob for deletion when no archive row refers to it, and says
/// whether it did. The caller holds the archive writer and has checked that
/// no index row refers to it; [`delete_marked`] removes the file after the
/// caller's transaction commits.
pub(crate) fn mark_for_deletion(archive: &rusqlite::Connection, hash: &BlobHash) -> Result<bool> {
    refuse_delta_base(archive, hash)?;
    Ok(archive.execute(
        "UPDATE blob SET state = 'deleting' WHERE hash = ?1 AND state = 'present'
         AND NOT EXISTS (SELECT 1 FROM extent WHERE blob = ?1)
         AND NOT EXISTS (SELECT 1 FROM extracted_blob WHERE blob = ?1)",
        [&hash.0[..]],
    )? > 0)
}

/// Refuses when `hash` is the base another blob is stored as a delta of.
pub(crate) fn refuse_delta_base(archive: &rusqlite::Connection, hash: &BlobHash) -> Result<()> {
    let delta_base: bool = archive.query_row(
        "SELECT EXISTS(SELECT 1 FROM blob WHERE base = ?1)",
        [&hash.0[..]],
        |r| r.get(0),
    )?;
    if delta_base {
        return Err(crate::StoreError::ExcisionBlobReferenced(*hash));
    }
    Ok(())
}

/// Deletes the files of blobs marked for deletion, then their rows. The
/// writer runs it after each excision and when it starts, so a stop between
/// the commit and the file deletion leaves nothing behind.
pub(crate) fn delete_marked(archive: &rusqlite::Connection, store: &hashed_blobs::Store) -> Result<u64> {
    let marked: Vec<(Vec<u8>, String)> = archive
        .prepare("SELECT hash, place FROM blob WHERE state = 'deleting'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (hash, place) in &marked {
        let referenced: bool = archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM extent WHERE blob = ?1)
                 OR EXISTS(SELECT 1 FROM extracted_blob WHERE blob = ?1)
                 OR EXISTS(SELECT 1 FROM blob WHERE base = ?1)",
            [hash],
            |r| r.get(0),
        )?;
        let hash = BlobHash(
            hash.as_slice()
                .try_into()
                .map_err(|_| crate::StoreError::Invalid("invalid blob hash".into()))?,
        );
        if referenced {
            return Err(crate::StoreError::ExcisionBlobReferenced(hash));
        }
        if place == "file" {
            crate::fault::check("blob.before_delete", &hash.to_string())?;
            match store.delete(&hashed_blobs::Hash(hash.0)) {
                Ok(()) | Err(hashed_blobs::Error::Missing(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        archive.execute("DELETE FROM blob WHERE hash = ?1", [&hash.0[..]])?;
    }
    Ok(marked.len() as u64)
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
mod tests;
