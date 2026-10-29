//! Bytes hashed off the writer thread into a private file, waiting for the
//! writer to give them their rows: in standalone mode compressed, then moved
//! into the blob store; in primary mode as they are, then sent to Sinex.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};

/// One pool batch and the private files its new blobs were written to. Inline
/// new blobs stay in the batch.
pub struct PreparedBatch {
    pub batch: crate::Batch,
    pub files: Vec<PreparedBytes>,
}

/// A private file for one archive, stored only by the writer, after the
/// rows that place it exist.
pub struct PreparedBytes {
    pub(crate) owner: std::path::PathBuf,
    pub(crate) file: Prepared,
}

pub(crate) enum Prepared {
    /// Compressed, to be moved under its content name in `blobs/`.
    Local {
        prepared: Box<hashed_blobs::Prepared>,
    },
    /// The bytes as they are, in `parts/`, to be sent to Sinex.
    Part {
        file: tempfile::NamedTempFile,
        hash: BlobHash,
        length: u64,
    },
}

impl PreparedBytes {
    pub fn hash(&self) -> BlobHash {
        match &self.file {
            Prepared::Local { prepared, .. } => BlobHash(prepared.hash().0),
            Prepared::Part { hash, .. } => *hash,
        }
    }
    pub fn len(&self) -> u64 {
        match &self.file {
            Prepared::Local { prepared, .. } => prepared.decoded_len(),
            Prepared::Part { length, .. } => *length,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl crate::Archive {
    /// Prepare frames off the writer thread. Their bytes become durable
    /// together when the writer appends and syncs a group.
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

    /// Prepares bytes that can be read only once (a ZIP member), after
    /// `inspect` has read them from a scratch file. A file on disk is
    /// inspected in place and prepared with [`Self::prepare_bytes`] instead.
    pub fn prepare_stream<T>(
        &self,
        mut reader: impl std::io::Read,
        inspect: impl FnOnce(&mut std::fs::File) -> Result<T>,
    ) -> Result<(PreparedBytes, T)> {
        use std::io::Seek;
        let mut scratch = if self.is_primary() {
            let directory = self.parts_dir()?;
            std::fs::create_dir_all(&directory)?;
            tempfile::tempfile_in(directory)?
        } else {
            tempfile::tempfile()?
        };
        std::io::copy(&mut reader, &mut scratch)?;
        scratch.rewind()?;
        let facts = inspect(&mut scratch)?;
        scratch.rewind()?;
        Ok((self.prepare_bytes(&mut scratch)?, facts))
    }

    /// Hashes (and in standalone mode compresses) into a private file.
    /// Dropping the returned handle deletes the file.
    pub fn prepare_bytes(&self, mut reader: impl std::io::Read) -> Result<PreparedBytes> {
        if self.is_primary() {
            std::fs::create_dir_all(self.parts_dir()?)?;
            let mut file = tempfile::NamedTempFile::new_in(self.parts_dir()?)?;
            let mut hasher = blake3::Hasher::new();
            let mut buffer = vec![0; 1 << 16];
            let mut length = 0u64;
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
                std::io::Write::write_all(&mut file, &buffer[..n])?;
                length += n as u64;
            }
            crate::timing::time(crate::timing::Stage::Sync, || file.as_file().sync_all())?;
            return Ok(PreparedBytes {
                owner: self.root.clone(),
                file: Prepared::Part {
                    file,
                    hash: BlobHash(*hasher.finalize().as_bytes()),
                    length,
                },
            });
        }
        let prepared = hashed_blobs::Preparation::new(self.parts_dir()?)?
            .plain_checked::<StoreError>(reader, crate::blobs::LEVEL, || Ok(()))?;
        Ok(PreparedBytes {
            owner: self.root.clone(),
            file: Prepared::Local {
                prepared: Box::new(prepared),
            },
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
