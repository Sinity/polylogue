//! The blob layer: inputs, revisions and extents over `hashed-blobs`
//! (architecture 4.2). A blob under 128 KiB lives inline in `archive.db`;
//! larger ones are files of seekable zstd frames.

use crate::Result;
use polylogue_model::BlobHash;

pub const INLINE_LIMIT: usize = 128 * 1024;

/// What one look at a file found, compared with the last observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileFacts {
    pub device: u64,
    pub inode: u64,
    pub size: u64,
    pub mtime_ns: i128,
    pub ctime_ns: i128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    /// Same facts as last time: nothing is read.
    Unchanged,
    /// The file grew and its old tail is intact: only the new bytes count.
    Appended { from: u64 },
    /// Content changed in place (a head rewrite or a full rewrite).
    Rewritten,
    /// The file is gone. Recorded as an observation, never a delete.
    Disappeared,
}

/// The bytes one append added, hashed and compressed off the writer thread.
pub struct Piece {
    pub hash: BlobHash,
    pub bytes: Vec<u8>,
}

pub fn classify(_last: Option<FileFacts>, _now: Option<FileFacts>) -> Change {
    // Architecture 4.2: stat key first; tail window check on growth.
    todo!()
}

/// Stores a piece's bytes before any extent refers to them.
pub fn put_piece(_piece: &Piece) -> Result<()> {
    // Architecture 4.2: blob row state writing -> present; file after row.
    todo!()
}
