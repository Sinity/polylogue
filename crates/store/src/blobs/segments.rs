//! Product hash lookup over immutable extents in the shared segment container.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum BlobCatalog { Archive, Projection }
impl BlobCatalog {
    fn table(self) -> &'static str {
        match self { Self::Archive => "blob", Self::Projection => "source_blob" }
    }
}

/// The catalog belongs to Polylogue; the shared reader only knows extents.
pub struct Blobs {
    root: PathBuf,
    catalog: PathBuf,
    layout: BlobCatalog,
    connection: Mutex<Option<Connection>>,
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
    pub(crate) fn new(root: PathBuf, catalog: PathBuf, layout: BlobCatalog) -> Self {
        Self {
            root,
            catalog,
            layout,
            connection: Mutex::new(None),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn extent(&self, hash: &hashed_blobs::Hash) -> Result<hashed_blobs::Extent> {
        let mut slot = self
            .connection
            .lock()
            .map_err(|_| StoreError::Invalid("blob catalog lock poisoned".into()))?;
        if slot.is_none() {
            *slot = Some(Connection::open_with_flags(
                &self.catalog,
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
            )?);
        }
        let bytes: Option<Vec<u8>>=slot.as_ref().expect("opened catalog").query_row(
            &format!("SELECT storage FROM {} WHERE hash=?1",self.layout.table()), [hash.0.as_slice()], |row|row.get(0)).optional()?;
        bytes.map(|bytes|hashed_blobs::Extent::decode(&bytes).map_err(StoreError::from)).transpose()?
            .ok_or_else(|| StoreError::NotFound(format!("blob {}", BlobHash(hash.0))))
    }

    pub fn read(&self, hash: &hashed_blobs::Hash) -> Result<hashed_blobs::BlobReader> {
        let extent = self.extent(hash)?;
        Ok(hashed_blobs::Reader::new(&self.root).open(&[extent], None)?)
    }

    pub fn read_range(&self, hash: &hashed_blobs::Hash, offset: u64, len: u64) -> Result<Vec<u8>> {
        let extent = self.extent(hash)?;
        Ok(hashed_blobs::Reader::new(&self.root).read_range(&[extent], offset, len, None)?)
    }

    /// Check each committed extent without changing stored bytes.
    pub fn scan(&self) -> Result<Vec<hashed_blobs::Mismatch>> {
        let conn = Connection::open_with_flags(&self.catalog, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let mut statement = conn.prepare(&format!("SELECT storage FROM {} ORDER BY hash",self.layout.table()))?;
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
        expected: Option<u64>,
        check: impl FnMut() -> hashed_blobs::Result<()> + Send + 'static,
    ) -> Result<hashed_blobs::BlobReader> {
        let extent = self.extent(hash)?;
        if extent.kind != hashed_blobs::Kind::Plain
            || expected.is_some_and(|length| extent.decoded_len != length)
        {
            return Err(StoreError::BlobCorrupt(BlobHash(hash.0)));
        }
        Ok(hashed_blobs::Reader::new(&self.root).open_checked(&[extent], None, check)?)
    }

    pub fn verify_plain<E: From<StoreError>>(
        &self,
        hash: &hashed_blobs::Hash,
        expected: u64,
        mut visit: impl FnMut(&[u8]) -> std::result::Result<(), E>,
    ) -> std::result::Result<hashed_blobs::VerificationStats, E> {
        use std::io::Read;
        let mut reader = self.read_plain(hash, Some(expected), || Ok(()))?;
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
    schema: &str,
    hash: &hashed_blobs::Hash,
) -> Result<Option<hashed_blobs::Extent>> {
    let bytes: Option<Vec<u8>> = conn
        .query_row(
            &format!("SELECT storage FROM {schema}blob WHERE hash=?1"),
            [hash.0.as_slice()],
            |row| row.get(0),
        )
        .optional()?;
    bytes
        .map(|bytes| hashed_blobs::Extent::decode(&bytes).map_err(StoreError::from))
        .transpose()
}

pub(crate) fn insert_extent(conn: &Connection, extent: &hashed_blobs::Extent) -> Result<()> {
    let old = load_extent(conn, "", &extent.hash)?;
    if let Some(old) = old {
        if old.decoded_len != extent.decoded_len {
            return Err(StoreError::BlobCorrupt(BlobHash(extent.hash.0)));
        }
        return Ok(());
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
    Ok(())
}

pub(crate) fn open_writer(conn: &Connection, root: &Path) -> Result<hashed_blobs::SegmentWriter> {
    let end: i64 = conn
        .query_row(
            "SELECT committed_end FROM segment WHERE name=?1",
            ["segments/current.hb"],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(0);
    let end = u64::try_from(end)
        .map_err(|_| StoreError::Invalid("negative committed segment end".into()))?;
    Ok(hashed_blobs::SegmentWriter::open(
        root,
        "segments/current.hb",
        end,
        &[],
        0o600,
    )?)
}
