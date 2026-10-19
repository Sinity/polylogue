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

/// The stored bytes of one revision chain: its extents in offset order.
/// Inline blobs are held decompressed (each is under 128 KiB); file blobs
/// are read by range from `hashed-blobs`, so parsing a large stored input
/// never holds it whole. This is how the loop parses an input from what the
/// archive stored, when its file has changed, moved or gone, or when the
/// index is built again (architecture 4.4, 5.3).
#[derive(Debug, Clone, Default)]
pub struct StoredChain {
    pub length: u64,
    extents: Vec<StoredExtent>,
}

#[derive(Debug, Clone)]
struct StoredExtent {
    offset: u64,
    length: u64,
    blob: BlobHash,
    blob_offset: u64,
    inline: Option<std::sync::Arc<Vec<u8>>>,
}

impl StoredChain {
    /// The extents of `chain` covering `[0, length)`. `archive` is a
    /// connection to `archive.db` (`schema` is its prefix: `""` or `"a."`).
    pub fn load(
        archive: &rusqlite::Connection,
        schema: &str,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<Self> {
        let mut stmt = archive.prepare_cached(&format!(
            "SELECT e.offset, e.length, e.blob, e.blob_offset, b.place, b.data, b.size
             FROM {schema}extent e JOIN {schema}blob b ON b.hash = e.blob
             WHERE e.input = ?1 AND e.chain = ?2 AND e.offset < ?3 ORDER BY e.offset"
        ))?;
        let rows = stmt.query_map(rusqlite::params![input, chain, length as i64], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<Vec<u8>>>(5)?,
                r.get::<_, i64>(6)?,
            ))
        })?;
        let mut extents = Vec::new();
        for row in rows {
            let (offset, len, blob, blob_offset, place, data, size) = row?;
            let hash = BlobHash(
                blob.as_slice()
                    .try_into()
                    .map_err(|_| crate::StoreError::Invalid("blob hash is not 32 bytes".into()))?,
            );
            let inline = match (place.as_str(), data) {
                ("inline", Some(data)) => {
                    let bytes = zstd::bulk::decompress(&data, size as usize)?;
                    if BlobHash::of(&bytes) != hash {
                        return Err(crate::StoreError::BlobCorrupt(hash));
                    }
                    Some(std::sync::Arc::new(bytes))
                }
                _ => None,
            };
            extents.push(StoredExtent {
                offset: offset as u64,
                length: len as u64,
                blob: hash,
                blob_offset: blob_offset as u64,
                inline,
            });
        }
        Ok(Self { length, extents })
    }

    /// Bytes `[start, end)` of the chain. A range no extent covers is an
    /// error, never zeros.
    pub fn read_range(&self, store: &hashed_blobs::Store, start: u64, end: u64) -> Result<Vec<u8>> {
        let end = end.min(self.length);
        let mut out = Vec::with_capacity(end.saturating_sub(start) as usize);
        let mut at = start;
        let first = self
            .extents
            .partition_point(|e| e.offset + e.length <= start);
        for e in &self.extents[first..] {
            if at >= end {
                break;
            }
            if e.offset > at {
                break;
            }
            let from = at - e.offset;
            let take = (e.length - from).min(end - at);
            match &e.inline {
                Some(bytes) => {
                    let s = (e.blob_offset + from) as usize;
                    out.extend_from_slice(&bytes[s..s + take as usize]);
                }
                None => out.extend_from_slice(&store.read_range(
                    &hashed_blobs::Hash(e.blob.0),
                    e.blob_offset + from,
                    take,
                )?),
            }
            at += take;
        }
        if at < end {
            return Err(crate::StoreError::Invalid(format!(
                "stored bytes end at {at}, before {end}: an extent is missing"
            )));
        }
        Ok(out)
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
