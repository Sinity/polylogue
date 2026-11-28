//! Reading input bytes. Unchanged file facts mean
//! nothing is read. A changed file only extends its chain after the whole
//! stored prefix has been compared.

use polylogue_store::FileFacts;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

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

/// Reads from the file version the plan compared. A changed descriptor or
/// path invalidates that comparison, including when the file grew.
pub(crate) fn read_source_range(
    path: &Path,
    start: u64,
    end: u64,
    expected: Option<FileFacts>,
) -> io::Result<(Vec<u8>, polylogue_store::capture_read::FileSnapshot)> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let facts = FileFacts::of(&metadata);
    if expected.is_some_and(|expected| facts != expected) {
        return Err(source_changed("source file changed before reading"));
    }
    let observed = file_snapshot(&metadata)?;
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = vec![0; (end - start) as usize];
    let result = read_exact_checked(&mut file, &mut bytes);
    polylogue_store::fault::check("read.source_range", &path.to_string_lossy())
        .map_err(io::Error::other)?;
    check_source(&file, path, facts)?;
    result?;
    Ok((bytes, observed))
}

pub(crate) fn check_source(file: &File, path: &Path, expected: FileFacts) -> io::Result<()> {
    let at_path = std::fs::metadata(path);
    if FileFacts::of(&file.metadata()?) != expected
        || at_path
            .as_ref()
            .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
        || FileFacts::of(&at_path?) != expected
    {
        return Err(source_changed("source file changed during reading"));
    }
    Ok(())
}

fn source_changed(reason: &'static str) -> io::Error {
    io::Error::other(polylogue_store::StoreError::SourceChanged(reason))
}

/// Compare the opened file with exactly the archived prefix. An unreadable
/// archived prefix cannot establish continuity. Refuse a moving source
/// before returning either comparison result.
pub(crate) fn prefix_matches(
    file: &mut File,
    path: &Path,
    expected: FileFacts,
    mut length: u64,
    mut prior: impl Read,
    mut observed: impl FnMut(u64),
) -> io::Result<bool> {
    if FileFacts::of(&file.metadata()?) != expected || length > expected.size {
        return Err(source_changed(
            "source file changed before comparing its prefix",
        ));
    }
    file.rewind()?;
    let result = (|| {
        let mut current = [0; 65536];
        let mut previous = [0; 65536];
        while length > 0 {
            let size = length.min(current.len() as u64) as usize;
            read_exact_checked(file, &mut current[..size])?;
            observed(size as u64);
            let available = prior.read_exact(&mut previous[..size]).is_ok();
            polylogue_store::cancellation::check_current().map_err(io::Error::other)?;
            if !available || current[..size] != previous[..size] {
                return Ok(false);
            }
            length -= size as u64;
        }
        Ok(true)
    })();
    check_source(file, path, expected)?;
    result
}

pub(crate) fn stored_prefix_intact(
    path: &Path,
    expected: FileFacts,
    chain: polylogue_store::blobs::StoredChain,
    blobs: &std::sync::Arc<polylogue_store::blobs::Blobs>,
    bytes_read: &mut u64,
) -> io::Result<bool> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(source_changed(
                "source file disappeared before comparing its prefix",
            ));
        }
        Err(error) => return Err(error),
    };
    prefix_matches(
        &mut file,
        path,
        expected,
        chain.length,
        chain.reader(std::sync::Arc::clone(blobs)),
        |size| *bytes_read += size,
    )
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
