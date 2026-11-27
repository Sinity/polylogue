//! A stored chain: one chain's bytes, read through its extent list as it was
//! when the chain was loaded.
use super::{BlobHash, Result};
use crate::StoreError;
use crate::{OptionExt as _, ResultExt as _};
use rusqlite::Connection;
use std::path::PathBuf;
use std::sync::Arc;

mod extents;
#[cfg(test)]
pub(crate) use extents::declared_rows;
mod reader;
pub use extents::CHAIN_CACHE_KIB;
use extents::ExtentList;
pub use reader::ChainReader;
use reader::Cursor;

#[derive(Debug, Clone, Default)]
pub struct StoredChain {
    pub length: u64,
    extents: Option<Arc<ExtentList>>,
    single: Option<BlobHash>,
}

/// Where chains' extents are stored: archive.db (`extent`, `blob`), or the
/// same tables seen through an index connection's views (`source_`).
#[derive(Debug, Clone)]
pub struct ChainSource {
    pub path: PathBuf,
    pub table_prefix: &'static str,
}

impl ChainSource {
    pub fn of(archive: &crate::Archive) -> Self {
        Self {
            path: archive.archive_db(),
            table_prefix: "",
        }
    }
}

impl StoredChain {
    /// The chain's first `length` bytes, read from `source` as it is now.
    pub fn load(source: &ChainSource, input: &str, chain: i64, length: u64) -> Result<Self> {
        Self::load_range(source, input, chain, 0, length)
    }

    pub(crate) fn load_range(
        source: &ChainSource,
        input: &str,
        chain: i64,
        start: u64,
        length: u64,
    ) -> Result<Self> {
        if start > length || length > i64::MAX as u64 {
            return Err(StoreError::stored_data("invalid stored range"));
        }
        let (extents, single) = ExtentList::capture(source, input, chain, start, length)?;
        Ok(Self {
            length,
            extents: Some(Arc::new(extents)),
            single,
        })
    }

    pub fn reader<S: super::PlainBlobs>(&self, store: Arc<S>) -> ChainReader<S> {
        ChainReader::new(self.clone(), store)
    }

    /// Ends the chain's read transaction until its next extent lookup. A
    /// reader calls this before it waits for anything other than bytes.
    pub fn release(&self) {
        if let Some(extents) = &self.extents {
            extents.release();
        }
    }

    /// A complete single file can use the blob store's sequential reader directly.
    pub fn single_file_blob(&self) -> Option<BlobHash> {
        self.single
    }

    /// Read a range-sized result; missing or corrupt selected bytes are errors.
    pub fn read_range(&self, store: &crate::blobs::Blobs, start: u64, end: u64) -> Result<Vec<u8>> {
        let end = end.min(self.length);
        let length = end.checked_sub(start).or_stored("invalid stored range")?;
        let length = usize::try_from(length).or_stored("stored range exceeds address space")?;
        let mut out = Vec::new();
        out.try_reserve_exact(length)
            .map_err(|e| StoreError::internal(format!("allocate stored range: {e}")))?;
        out.resize(length, 0);
        let mut cursor = Cursor::range(start, end);
        let mut at = 0;
        while at < out.len() {
            let n = cursor.read(&mut out[at..], self.extents.as_deref(), store)?;
            if n == 0 {
                return Err(StoreError::stored_data("stored byte range missing"));
            }
            at += n;
        }
        Ok(out)
    }

    /// Where this chain's bytes and `other`'s first differ, comparing from
    /// `from` (bytes before it are known to be alike); the shorter length
    /// when one chain's bytes begin the other's.
    pub fn shared_prefix(
        &self,
        other: &StoredChain,
        store: &crate::blobs::Blobs,
        from: u64,
    ) -> Result<u64> {
        let end = self.length.min(other.length);
        if from >= end {
            return Ok(end);
        }
        let (mut mine, mut theirs) = (Cursor::range(from, end), Cursor::range(from, end));
        let (mut a, mut b) = (vec![0; 1 << 20], vec![0; 1 << 20]);
        let mut at = from;
        while at < end {
            let want = a.len().min(usize::try_from(end - at).unwrap_or(usize::MAX));
            fill(&mut mine, &mut a[..want], self.extents.as_deref(), store)?;
            fill(&mut theirs, &mut b[..want], other.extents.as_deref(), store)?;
            if let Some(differs) = a[..want].iter().zip(&b[..want]).position(|(x, y)| x != y) {
                // An abandoned range keeps no read transaction open.
                self.release();
                other.release();
                return Ok(at + differs as u64);
            }
            at += want as u64;
        }
        Ok(end)
    }
}

/// Reads exactly `out.len()` bytes of the range `cursor` reads.
fn fill(
    cursor: &mut Cursor,
    out: &mut [u8],
    extents: Option<&ExtentList>,
    store: &crate::blobs::Blobs,
) -> Result<()> {
    let mut at = 0;
    while at < out.len() {
        let n = cursor.read(&mut out[at..], extents, store)?;
        if n == 0 {
            return Err(StoreError::stored_data("stored byte range missing"));
        }
        at += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
