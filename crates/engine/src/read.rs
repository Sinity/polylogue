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

#[cfg(test)]
mod tests;
