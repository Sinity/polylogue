//! Reading input bytes. Unchanged file facts mean
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
    read_exact_checked(&mut f, &mut buf)?;
    Ok(buf)
}

/// Reads in 64 KiB parts, checking the thread's stop request and request
/// limit before each ([`polylogue_store::cancellation::check_current`]).
fn read_exact_checked(reader: &mut impl Read, bytes: &mut [u8]) -> io::Result<()> {
    for part in bytes.chunks_mut(65536) {
        polylogue_store::cancellation::check_current().map_err(io::Error::other)?;
        reader.read_exact(part)?;
    }
    Ok(())
}

/// Bytes a reader already holds of a file: `bytes` end at offset `end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProvenTail {
    pub end: u64,
    pub bytes: Vec<u8>,
}

/// Reads `[start, end)` from one opened file and checks it against the file
/// facts sampled when the read was planned. A file that only grew since
/// then is read up to the sampled length, since bytes appended past `end`
/// leave the range as it was. Growth is the same file (device and inode)
/// with a larger size whose `proven` tail, and the last bytes of the range
/// just read, are still in place afterwards; anything else (another inode,
/// a shorter file, a write that did not grow it, a changed tail) is a
/// changed source and the read is refused.
pub(crate) fn read_source_range(
    path: &Path,
    start: u64,
    end: u64,
    expected: Option<polylogue_store::FileFacts>,
    proven: Option<&ProvenTail>,
) -> io::Result<(Vec<u8>, polylogue_store::capture_read::FileSnapshot)> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let observed = file_snapshot(&metadata)?;
    let before = polylogue_store::FileFacts::of(&metadata);
    let sampled = expected.unwrap_or(before);
    if !same_or_grown(&sampled, &before) {
        return Err(source_changed("source file changed before reading"));
    }
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = vec![0; (end - start) as usize];
    read_exact_checked(&mut file, &mut bytes)?;
    polylogue_store::fault::check("read.source_range", &path.to_string_lossy())
        .map_err(io::Error::other)?;
    let held = polylogue_store::FileFacts::of(&file.metadata()?);
    let at_path = polylogue_store::FileFacts::of(&std::fs::metadata(path)?);
    if !same_or_grown(&before, &held) || !same_or_grown(&held, &at_path) {
        return Err(source_changed("source file changed during reading"));
    }
    if at_path != sampled {
        // Grown: what the reader holds must still be where it was.
        let range_tail = ProvenTail {
            end,
            bytes: tail_of(&bytes),
        };
        for tail in proven.into_iter().chain([&range_tail]) {
            if !tail_intact(path, tail.end, &tail.bytes)? {
                return Err(source_changed("source file changed during reading"));
            }
        }
    }
    Ok((bytes, observed))
}

/// The same file, unchanged or longer than when `then` was taken.
fn same_or_grown(then: &polylogue_store::FileFacts, now: &polylogue_store::FileFacts) -> bool {
    then == now || (then.device == now.device && then.inode == now.inode && now.size > then.size)
}

fn source_changed(reason: &'static str) -> io::Error {
    io::Error::other(polylogue_store::StoreError::SourceChanged(reason))
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

/// Whether the file at `path` begins with exactly the stored bytes of
/// `chain`. A file whose identity changed (another inode, or gone and back)
/// continues its chain only when every stored byte is still there.
pub fn stored_prefix_intact(
    path: &Path,
    chain: &polylogue_store::blobs::StoredChain,
    blobs: &std::sync::Arc<polylogue_store::blobs::Blobs>,
) -> io::Result<bool> {
    let mut file = File::open(path)?.take(chain.length);
    let mut stored = chain.reader(std::sync::Arc::clone(blobs));
    let mut left = vec![0u8; 1 << 20];
    let mut right = vec![0u8; 1 << 20];
    let mut compared = 0u64;
    loop {
        let n = read_full(&mut file, &mut left)?;
        if n == 0 {
            return Ok(compared == chain.length);
        }
        if read_full(&mut stored, &mut right[..n])? != n || left[..n] != right[..n] {
            return Ok(false);
        }
        compared += n as u64;
    }
}

fn read_full(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// The last [`TAIL`] bytes of `bytes`, kept to check the next growth.
pub fn tail_of(bytes: &[u8]) -> Vec<u8> {
    bytes[bytes.len().saturating_sub(TAIL)..].to_vec()
}

/// One imported blob, verified once and decoded forward. Only an incomplete
/// record overlaps parser jobs; consuming a prefix releases that buffer.
pub struct Imported {
    reader: Decoder,
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

enum Decoder {
    /// Stored bytes not read yet. A pass plans every input it parses again
    /// before it reads any, so a plan opens no file until its first job.
    Unopened(
        Box<(
            polylogue_store::blobs::StoredChain,
            std::sync::Arc<polylogue_store::blobs::Blobs>,
        )>,
    ),
    Plain(Box<polylogue_store::blobs::BlobReader>),
    Chain(Box<polylogue_store::blobs::ChainReader>),
}

impl Read for Decoder {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if let Self::Unopened(stored) = self {
            let (chain, store) = &**stored;
            *self = match chain.single_file_blob() {
                Some(hash) => Self::Plain(Box::new(
                    store
                        .read_plain(&hashed_blobs::Hash(hash.0), chain.length, || Ok(()))
                        .map_err(io::Error::other)?,
                )),
                None => Self::Chain(Box::new(chain.reader(std::sync::Arc::clone(store)))),
            };
        }
        match self {
            Self::Unopened(_) => unreachable!("opened above"),
            Self::Plain(reader) => reader.read(out),
            Self::Chain(reader) => reader.read(out),
        }
    }
}

impl Imported {
    pub fn open(
        store: &polylogue_store::blobs::Blobs,
        hash: polylogue_model::BlobHash,
        length: u64,
    ) -> polylogue_store::Result<Self> {
        Ok(Self {
            reader: Decoder::Plain(Box::new(store.read_plain(
                &hashed_blobs::Hash(hash.0),
                length,
                || Ok(()),
            )?)),
            start: 0,
            pending: Vec::new(),
        })
    }

    /// A chain's stored bytes, opened when they are first read.
    pub fn from_chain(
        chain: polylogue_store::blobs::StoredChain,
        store: std::sync::Arc<polylogue_store::blobs::Blobs>,
    ) -> Self {
        Self {
            reader: Decoder::Unopened(Box::new((chain, store))),
            start: 0,
            pending: Vec::new(),
        }
    }

    pub fn read_range(&mut self, start: u64, end: u64) -> io::Result<Vec<u8>> {
        if start != self.start || end < start {
            return Err(io::Error::other(polylogue_store::StoreError::internal(
                "import ranges must advance in parser order",
            )));
        }
        let length = (end - start) as usize;
        if length <= self.pending.len() {
            return Ok(self.pending[..length].to_vec());
        }
        let previous = self.pending.len();
        self.pending.resize(length, 0);
        let read = self.reader.read_exact(&mut self.pending[previous..]);
        // The next range is read by a later job: the chain holds no read
        // transaction while that job waits in the pool.
        if let Decoder::Chain(chain) = &self.reader {
            chain.release();
        }
        read?;
        Ok(self.pending.clone())
    }

    pub fn consume(&mut self, upto: u64) {
        let consumed = (upto - self.start) as usize;
        if consumed > 0 {
            // Plans can outlive their jobs' byte reservations. Keep only the
            // incomplete suffix, not the allocation of a consumed chunk.
            self.pending = self.pending.split_off(consumed);
        }
        self.start = upto;
    }
}

/// The identity and times of an opened provider file, recorded with the
/// capture read that read it.
pub(crate) fn file_snapshot(
    metadata: &std::fs::Metadata,
) -> std::io::Result<polylogue_store::capture_read::FileSnapshot> {
    use polylogue_store::capture_read::{FileSnapshot, FileTime};
    use std::os::unix::fs::MetadataExt;
    let time = |seconds, nanos: i64| -> std::io::Result<FileTime> {
        let nanos = u32::try_from(nanos)
            .ok()
            .filter(|nanos| *nanos < 1_000_000_000)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "provider file has invalid timestamp nanoseconds",
                )
            })?;
        Ok(FileTime { seconds, nanos })
    };
    Ok(FileSnapshot {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified: time(metadata.mtime(), metadata.mtime_nsec())?,
        changed: time(metadata.ctime(), metadata.ctime_nsec())?,
    })
}

#[cfg(test)]
mod tests;
