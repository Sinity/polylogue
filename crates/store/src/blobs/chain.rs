//! A stored chain: an immutable, disposable copy of one chain's extent list.
use super::{BlobHash, Result};
use crate::StoreError;
use rusqlite::Connection;
use std::sync::Arc;

mod extents;
mod reader;
use extents::ExtentList;
pub use reader::ChainReader;
use reader::Cursor;

#[derive(Debug, Clone, Default)]
pub struct StoredChain {
    pub length: u64,
    extents: Option<Arc<ExtentList>>,
    single: Option<BlobHash>,
}

impl StoredChain {
    /// Capture extents and inline compressed bytes under the caller's existing view.
    pub fn load(
        archive: &Connection,
        schema: &str,
        input: &str,
        chain: i64,
        length: u64,
    ) -> Result<Self> {
        Self::load_range(archive, schema, input, chain, 0, length)
    }

    pub(crate) fn load_range(
        archive: &Connection,
        schema: &str,
        input: &str,
        chain: i64,
        start: u64,
        length: u64,
    ) -> Result<Self> {
        if start > length || length > i64::MAX as u64 {
            return Err(StoreError::Invalid("invalid stored range".into()));
        }
        let (extents, single) = ExtentList::capture(archive, schema, input, chain, start, length)?;
        Ok(Self {
            length,
            extents: Some(Arc::new(extents)),
            single,
        })
    }

    pub fn reader(&self, store: Arc<hashed_blobs::Store>) -> ChainReader {
        ChainReader::new(self.clone(), store)
    }

    /// A complete single file can use the blob store's sequential reader directly.
    pub fn single_file_blob(&self) -> Option<BlobHash> {
        self.single
    }

    /// Read a range-sized result; missing or corrupt selected bytes are errors.
    pub fn read_range(&self, store: &hashed_blobs::Store, start: u64, end: u64) -> Result<Vec<u8>> {
        let end = end.min(self.length);
        let length = end
            .checked_sub(start)
            .ok_or_else(|| StoreError::Invalid("invalid stored range".into()))?;
        let length = usize::try_from(length)
            .map_err(|_| StoreError::Invalid("stored range exceeds address space".into()))?;
        let mut out = Vec::new();
        out.try_reserve_exact(length)
            .map_err(|e| StoreError::Invalid(format!("allocate stored range: {e}")))?;
        out.resize(length, 0);
        let mut cursor = Cursor::range(start, end);
        let mut at = 0;
        while at < out.len() {
            let n = cursor.read(&mut out[at..], self.extents.as_deref(), store)?;
            if n == 0 {
                return Err(StoreError::Invalid("stored extent missing".into()));
            }
            at += n;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
