//! Reading input bytes (architecture 4.2, 5.2). Unchanged file facts mean
//! nothing is read; a grown file has its old tail compared before only the
//! appended bytes are read. Hashing and compression happen on the parse
//! pool, never on the writer thread.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// How many bytes before the old end are compared on growth.
pub const TAIL: usize = 4096;

/// Bytes `[start, end)` of a file. A file that shrank below `end` since it
/// was looked at is an error; the next pass sees it as rewritten.
pub fn read_range(path: &Path, start: u64, end: u64) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    f.seek(SeekFrom::Start(start))?;
    let mut buf = vec![0u8; (end - start) as usize];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

/// Whether the bytes just before `from` still equal `tail`, the end of what
/// was stored. False means the file changed in place, not only grew.
pub fn tail_intact(path: &Path, from: u64, tail: &[u8]) -> io::Result<bool> {
    let len = tail.len() as u64;
    if len == 0 || from < len {
        return Ok(true);
    }
    Ok(read_range(path, from - len, from)? == tail)
}

/// The last [`TAIL`] bytes of `bytes`, kept to check the next growth.
pub fn tail_of(bytes: &[u8]) -> Vec<u8> {
    bytes[bytes.len().saturating_sub(TAIL)..].to_vec()
}

/// One imported blob, verified once and decoded forward. Only an incomplete
/// record overlaps parser jobs; consuming a prefix releases that buffer.
pub struct Imported {
    reader: hashed_blobs::PlainReader,
    start: u64,
    pending: Vec<u8>,
}

impl std::fmt::Debug for Imported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Imported")
            .field("start", &self.start)
            .finish_non_exhaustive()
    }
}

impl Imported {
    pub fn open(
        store: &hashed_blobs::Store,
        hash: polylogue_model::BlobHash,
        length: u64,
    ) -> hashed_blobs::Result<Self> {
        Ok(Self {
            reader: store.read_plain(&hashed_blobs::Hash(hash.0), Some(length), || Ok(()))?,
            start: 0,
            pending: Vec::new(),
        })
    }

    pub fn read_range(&mut self, start: u64, end: u64) -> io::Result<Vec<u8>> {
        if start != self.start || end < start {
            return Err(io::Error::other(
                "import ranges must advance in parser order",
            ));
        }
        let length = (end - start) as usize;
        if length <= self.pending.len() {
            return Ok(self.pending[..length].to_vec());
        }
        let previous = self.pending.len();
        self.pending.resize(length, 0);
        self.reader.read_exact(&mut self.pending[previous..])?;
        Ok(self.pending.clone())
    }

    pub fn consume(&mut self, upto: u64) {
        let consumed = (upto - self.start) as usize;
        self.pending.drain(..consumed);
        self.start = upto;
    }
}

#[cfg(test)]
mod tests;
