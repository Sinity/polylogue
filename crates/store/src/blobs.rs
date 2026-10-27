//! The blob layer: inputs, revisions and extents over `hashed-blobs`
//!. A blob under 128 KiB lives inline in `archive.db`,
//! zstd-compressed; larger ones are files of seekable zstd frames. Hashing
//! and compression run before the writer; the writer writes a file's blob row
//! (`writing`) before installing the file, then marks it `present`.

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

/// How a new blob's bytes are stored.
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
pub struct NewBlob {
    pub hash: BlobHash,
    pub size: u64,
    pub stored: Stored,
}

impl NewBlob {
    /// Test rows: hashes `bytes`, compresses small ones for an inline row
    /// and writes large ones straight to the blob store.
    #[cfg(test)]
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
    Ok(archive.execute(
        "UPDATE blob SET state = 'deleting' WHERE hash = ?1 AND state IN ('present', 'writing')
         AND NOT EXISTS (SELECT 1 FROM extent WHERE blob = ?1)
         AND NOT EXISTS (SELECT 1 FROM chat_action_attachment WHERE blob = ?1)",
        [&hash.0[..]],
    )? > 0)
}

/// Brings every blob row a stop or a failed install left behind to a final
/// state, and returns how many became `present`. A `writing` row that an
/// extent refers to becomes `present` when its file is under its
/// content name (installation is a rename, so a file there is whole);
/// without the file it stays `writing`, is reported, and the loop reads its
/// input again. A `writing` row nothing in the archive refers to is an
/// extracted blob whose parsed rows never committed, because they are
/// written only after it is `present`: it is marked `deleting`, and parsing
/// its record again stores it again. Rows marked `deleting` then go with
/// their files ([`delete_marked`]). The caller holds the archive writer.
pub(crate) fn finish_pending(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
) -> Result<u64> {
    let tx = archive.unchecked_transaction()?;
    tx.execute(
        "UPDATE blob SET state = 'deleting' WHERE state = 'writing'
         AND NOT EXISTS (SELECT 1 FROM extent WHERE extent.blob = blob.hash)",
        [],
    )?;
    let waiting: Vec<Vec<u8>> = tx
        .prepare("SELECT hash FROM blob WHERE state = 'writing'")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut completed = 0;
    for hash in waiting {
        let file = hashed_blobs::Hash(
            hash.as_slice()
                .try_into()
                .map_err(|_| crate::StoreError::Invalid("invalid blob hash".into()))?,
        );
        if !store.contains(&file) {
            continue;
        }
        let stored_size = i64::try_from(store.stored_size(&file)?)
            .map_err(|_| crate::StoreError::Invalid("blob exceeds archive address space".into()))?;
        completed += tx.execute(
            "UPDATE blob SET state = 'present', stored_size = ?2 WHERE hash = ?1",
            rusqlite::params![hash, stored_size],
        )? as u64;
    }
    tx.commit()?;
    delete_marked(archive, Some(store))?;
    Ok(completed)
}

/// Deletes the files of blobs marked for deletion, then their rows. The
/// writer runs it after each excision, and [`finish_pending`] runs it when the
/// archive opens, so a stop between the commit and the file deletion leaves
/// nothing behind. Without a `store` (primary mode, where Sinex holds the
/// files and excised them) only the rows go.
pub(crate) fn delete_marked(
    archive: &rusqlite::Connection,
    store: Option<&hashed_blobs::Store>,
) -> Result<u64> {
    let marked: Vec<(Vec<u8>, String)> = archive
        .prepare("SELECT hash, place FROM blob WHERE state = 'deleting'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (hash, place) in &marked {
        let referenced: bool = archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM extent WHERE blob = ?1)
                 OR EXISTS(SELECT 1 FROM chat_action_attachment WHERE blob = ?1)",
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
        if let (Some(store), "file") = (store, place.as_str()) {
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
