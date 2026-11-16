//! Segment-byte counts for fixture archives.
use std::io::{self, Read, Seek, SeekFrom};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub struct BlobReader {
    reader: hashed_blobs::BlobReader,
    bytes_read: Arc<AtomicU64>,
}

impl BlobReader {
    pub(super) fn new(reader: hashed_blobs::BlobReader, bytes_read: Arc<AtomicU64>) -> Self {
        Self { reader, bytes_read }
    }

    pub fn stats(&self) -> hashed_blobs::VerificationStats {
        self.reader.stats()
    }
}

impl Read for BlobReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = self.reader.read(out)?;
        self.bytes_read.fetch_add(count as u64, Ordering::Relaxed);
        Ok(count)
    }
}

impl Seek for BlobReader {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.reader.seek(from)
    }
}
