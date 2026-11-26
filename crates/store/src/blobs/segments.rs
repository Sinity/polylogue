//! Product hash lookup over immutable extents in the shared segment container.
use crate::ConnExt as _;
use crate::ResultExt as _;
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OpenFlags};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The catalog belongs to Polylogue; the shared reader only knows extents.
pub struct Blobs {
    root: PathBuf,
    catalog: PathBuf,
    connection: Mutex<Option<Connection>>,
    #[cfg(feature = "fixture")]
    bytes_read: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl std::fmt::Debug for Blobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Blobs")
            .field("root", &self.root)
            .field("catalog", &self.catalog)
            .finish_non_exhaustive()
    }
}

impl Blobs {
    pub(crate) fn new(root: PathBuf, catalog: PathBuf) -> Self {
        Self {
            root,
            catalog,
            connection: Mutex::new(None),
            #[cfg(feature = "fixture")]
            bytes_read: Default::default(),
        }
    }

    /// Decoded segment bytes delivered by this catalog's readers in a fixture.
    #[cfg(feature = "fixture")]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn count_bytes(&self, reader: hashed_blobs::BlobReader) -> super::BlobReader {
        #[cfg(feature = "fixture")]
        {
            super::BlobReader::new(reader, self.bytes_read.clone())
        }
        #[cfg(not(feature = "fixture"))]
        {
            reader
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn extent(&self, hash: &hashed_blobs::Hash) -> Result<hashed_blobs::Extent> {
        let mut slot = self
            .connection
            .lock()
            .or_internal("blob catalog lock poisoned")?;
        if slot.is_none() {
            *slot = Some(Connection::open_with_flags(
                &self.catalog,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?);
        }
        let bytes: Option<Vec<u8>> = slot.as_ref().expect("opened catalog").scalar_optional(
            "SELECT storage FROM blob WHERE hash=?1",
            [hash.0.as_slice()],
        )?;
        bytes
            .map(|bytes| hashed_blobs::Extent::decode(&bytes).map_err(StoreError::from))
            .transpose()?
            .ok_or_else(|| StoreError::NotFound(format!("blob {}", BlobHash(hash.0))))
    }

    pub fn read(&self, hash: &hashed_blobs::Hash) -> Result<super::BlobReader> {
        let extent = self.extent(hash)?;
        Ok(self.count_bytes(hashed_blobs::Reader::new(&self.root).open(&[extent], None)?))
    }

    pub fn read_range(
        &self,
        hash: &hashed_blobs::Hash,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let extent = self.extent(hash)?;
        Ok(hashed_blobs::Reader::new(&self.root).read_range(&[extent], offset, length, None)?)
    }

    /// Check each committed extent without changing stored bytes.
    #[cfg(test)]
    pub fn scan(&self) -> Result<Vec<hashed_blobs::Mismatch>> {
        let conn = Connection::open_with_flags(&self.catalog, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut statement = conn.prepare("SELECT storage FROM blob ORDER BY hash")?;
        let mut rows = statement.query([])?;
        let reader = hashed_blobs::Reader::new(&self.root);
        let mut mismatches = Vec::new();
        while let Some(row) = rows.next()? {
            let bytes: Vec<u8> = row.get(0)?;
            let extent = hashed_blobs::Extent::decode(&bytes)?;
            mismatches.extend(reader.scan(&[extent]));
        }
        Ok(mismatches)
    }

    pub fn read_plain(
        &self,
        hash: &hashed_blobs::Hash,
        expected: u64,
        check: impl FnMut() -> hashed_blobs::Result<()> + Send + 'static,
    ) -> Result<super::BlobReader> {
        let extent = self.extent(hash)?;
        if extent.kind != hashed_blobs::Kind::Plain || extent.decoded_len != expected {
            return Err(StoreError::BlobCorrupt(BlobHash(hash.0)));
        }
        Ok(
            self.count_bytes(hashed_blobs::Reader::new(&self.root).open_checked(
                &[extent],
                None,
                check,
            )?),
        )
    }

    pub fn verify_plain<E: From<StoreError>>(
        &self,
        hash: &hashed_blobs::Hash,
        expected: u64,
        mut visit: impl FnMut(&[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<hashed_blobs::VerificationStats, E> {
        use std::io::Read;
        let mut reader = self.read_plain(hash, expected, || Ok(()))?;
        let mut digest = blake3::Hasher::new();
        let mut buffer = vec![0; 64 << 10];
        loop {
            let count = reader.read(&mut buffer).map_err(StoreError::from)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
            visit(&buffer[..count])?;
        }
        if digest.finalize().as_bytes() != &hash.0 {
            return Err(StoreError::BlobCorrupt(BlobHash(hash.0)).into());
        }
        Ok(reader.stats())
    }
}

pub(crate) fn load_extent(
    conn: &Connection,
    table_prefix: &str,
    hash: &hashed_blobs::Hash,
) -> Result<Option<hashed_blobs::Extent>> {
    let bytes: Option<Vec<u8>> = conn.scalar_optional(
        &format!("SELECT storage FROM {table_prefix}blob WHERE hash=?1"),
        [hash.0.as_slice()],
    )?;
    bytes
        .map(|bytes| hashed_blobs::Extent::decode(&bytes).map_err(StoreError::from))
        .transpose()
}

/// Catalogs a blob's extent; false when the blob was catalogued already.
pub(crate) fn insert_extent(conn: &Connection, extent: &hashed_blobs::Extent) -> Result<bool> {
    let old = load_extent(conn, "", &extent.hash)?;
    if let Some(old) = old {
        if old.decoded_len != extent.decoded_len {
            return Err(StoreError::BlobCorrupt(BlobHash(extent.hash.0)));
        }
        return Ok(false);
    }
    conn.execute(
        "INSERT INTO blob(hash,size,stored_size,storage) VALUES (?1,?2,?3,?4)",
        rusqlite::params![
            extent.hash.0.as_slice(),
            extent.decoded_len as i64,
            extent.len as i64,
            extent.encode()
        ],
    )?;
    Ok(true)
}

/// Records how far the current segment is committed, in the transaction that
/// stores the extents written up to `end`. The committed end only grows:
/// the segment is cut back to it when a writer opens, so an earlier end
/// recorded over a later one would cut off bytes committed rows name.
pub(crate) fn record_committed_end(conn: &Connection, end: u64) -> Result<()> {
    let end = crate::to_i64(end, "segment exceeds archive address space")?;
    let kept: i64 = conn.query_row(
        "INSERT INTO segment(name,committed_end) VALUES (?1,?2)
         ON CONFLICT(name) DO UPDATE SET committed_end=max(committed_end,excluded.committed_end)
         RETURNING committed_end",
        rusqlite::params![crate::bytes::CURRENT_SEGMENT, end],
        |row| row.get(0),
    )?;
    if kept != end {
        return Err(crate::StoreError::internal(format!(
            "the segment's committed end would move back from {kept} to {end}"
        )));
    }
    Ok(())
}
