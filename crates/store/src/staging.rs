//! Input bytes use one encoded preparation in both modes. The standalone
//! writer appends it to its segment; the primary producer adopts its custody.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};

/// One pool batch and the private encodings its new blobs were written to.
pub struct PreparedBatch {
    pub batch: crate::Batch,
    pub files: Vec<PreparedBytes>,
}

/// Encoded bytes whose inode stays pinned until the writer transfers custody.
pub struct PreparedBytes {
    pub(crate) owner: std::path::PathBuf,
    pub(crate) file: Box<hashed_blobs::Prepared>,
    temporary: tempfile::NamedTempFile,
}

impl PreparedBytes {
    pub fn hash(&self) -> BlobHash {
        BlobHash(self.file.hash().0)
    }
    pub fn len(&self) -> u64 {
        self.file.decoded_len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn reader(&self) -> Result<hashed_blobs::BlobReader> {
        Ok(self
            .file
            .reader(None, hashed_blobs::ReadLimits::default())?)
    }
    pub(crate) fn body(&self) -> Result<crate::primary::PreparedBody> {
        Ok(crate::primary::PreparedBody {
            file: self.temporary.as_file().try_clone()?,
            descriptor: self.file.descriptor(),
            path: Some(self.temporary.path().to_owned()),
        })
    }
    pub(crate) fn into_prepared(self) -> hashed_blobs::Prepared {
        *self.file
    }
}

impl crate::Archive {
    /// Prepare frames off the writer thread. Durability belongs to the group
    /// append or to the producer's accepted private custody.
    pub fn prepare_blob(&self, bytes: &[u8]) -> Result<(crate::NewBlob, Option<PreparedBytes>)> {
        let prepared = self.prepare_bytes(bytes)?;
        Ok((
            crate::NewBlob {
                hash: prepared.hash(),
                size: prepared.len(),
            },
            Some(prepared),
        ))
    }

    pub fn prepare_decoded(&self, bytes: &[u8]) -> Result<(crate::NewBlob, Option<PreparedBytes>)> {
        self.prepare_blob(bytes)
    }

    /// Encode a one-shot stream once, then inspect it through the seekable
    /// decoder. No plaintext scratch file is created.
    pub fn prepare_stream<T>(
        &self,
        reader: impl std::io::Read,
        inspect: impl FnOnce(&mut hashed_blobs::BlobReader) -> Result<T>,
    ) -> Result<(PreparedBytes, T)> {
        let prepared = self.prepare_bytes(reader)?;
        let facts = inspect(&mut prepared.reader()?)?;
        Ok((prepared, facts))
    }

    /// The same prepared encoding serves standalone storage and primary custody.
    pub fn prepare_bytes(&self, reader: impl std::io::Read) -> Result<PreparedBytes> {
        let directory = self.parts_dir()?;
        std::fs::create_dir_all(&directory)?;
        let temporary = tempfile::NamedTempFile::new_in(&directory)?;
        let prepared = hashed_blobs::Preparation::new(&directory)?
            .plain_into_checked::<StoreError>(
                temporary.as_file().try_clone()?,
                reader,
                crate::blobs::LEVEL,
                || Ok(()),
            )?;
        Ok(PreparedBytes {
            owner: self.root.clone(),
            file: Box::new(prepared),
            temporary,
        })
    }
}

/// Where a stored batch's bytes are: its input and revision chain.
#[derive(Debug, Clone)]
pub struct StoredInput {
    pub input: String,
    pub chain: i64,
    pub length: u64,
    pub already_stored: bool,
}

/// All catalogued extents are durable. Missing source ranges are detected
/// while constructing a stored chain; no blob can be partly installed.
pub(crate) fn require_range(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: i64,
    range: std::ops::Range<u64>,
) -> Result<()> {
    if range.start > range.end {
        return Err(StoreError::Invalid("invalid stored range".into()));
    }
    let length: Option<i64> = conn
        .query_row(
            &format!("SELECT max(length) FROM {schema}revision WHERE input=?1 AND chain=?2"),
            rusqlite::params![input, chain],
            |r| r.get(0),
        )
        .optional()?
        .flatten();
    if length.is_none_or(|length| length < 0 || range.end > length as u64) {
        return Err(StoreError::NotFound(format!(
            "stored input {input} chain {chain} range {}..{}",
            range.start, range.end
        )));
    }
    Ok(())
}
