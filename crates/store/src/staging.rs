//! Input bytes use one encoded preparation in both modes. The standalone
//! writer appends it to its segment; the primary producer sends the stored
//! encoding.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};

/// One pool batch and the private encodings its new blobs were written to.
pub struct PreparedBatch {
    pub batch: crate::Batch,
    pub files: Vec<PreparedBytes>,
}

/// One private encoded file and the descriptor produced by its encoder.
/// The archive lock owns its path until storing it opens a file handle.
/// Closed preparations keep read-ahead independent of the descriptor limit.
#[derive(Clone)]
pub struct PreparedBytes {
    pub(crate) owner: std::path::PathBuf,
    descriptor: hashed_blobs::EncodedBlob,
    temporary: std::sync::Arc<tempfile::TempPath>,
}

impl PreparedBytes {
    pub fn hash(&self) -> BlobHash {
        BlobHash(self.descriptor.hash.0)
    }
    pub fn len(&self) -> u64 {
        self.descriptor.decoded_len
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn reader(&self) -> Result<hashed_blobs::BlobReader> {
        Ok(self.descriptor.open(
            std::fs::File::open(self.temporary.as_ref())?,
            None,
            hashed_blobs::ReadLimits::default(),
        )?)
    }
    pub(crate) fn body(&self) -> Result<crate::primary::PreparedBody> {
        Ok(crate::primary::PreparedBody {
            file: std::fs::File::open(self.temporary.as_ref())?,
            descriptor: self.descriptor.clone(),
            path: Some(self.temporary.to_path_buf()),
        })
    }
    /// The encoding to append, reopened from its staged name: prepared files
    /// stay closed until their append, so a group of any size holds no
    /// descriptor per member. Its preparation computed the descriptor, so
    /// only the length is checked here; the append checks the encoded digest
    /// while it copies, and refuses a changed file.
    pub(crate) fn into_prepared(self) -> hashed_blobs::Result<hashed_blobs::Prepared> {
        hashed_blobs::Prepared::from_own_encoding(
            std::fs::File::open(self.temporary.as_ref()).map_err(hashed_blobs::Error::Io)?,
            self.descriptor,
        )
    }
}

impl crate::Archive {
    /// Prepare frames off the writer thread. The bytes become durable when
    /// a group append syncs them, or when the primary producer accepts them.
    pub fn prepare_blob(&self, bytes: &[u8]) -> Result<(crate::NewBlob, Option<PreparedBytes>)> {
        let prepared = self.prepare_bytes(bytes)?;
        Ok((
            crate::NewBlob::whole(prepared.hash(), prepared.len()),
            Some(prepared),
        ))
    }

    /// Prepare bytes an input's source rewrites whole as content-defined
    /// chunks (the shared chunker's default sizes), so a revision that changes
    /// one place stores only the chunks around it: the segment writer skips
    /// chunks the archive holds, and Sinex skips chunks it holds. Every chunk
    /// is prepared, held or not, so the bytes never depend on a chunk that is
    /// removed before they are stored. Bytes that are one chunk are one blob.
    pub fn prepare_chunked(
        &self,
        reader: impl std::io::Read,
    ) -> Result<(crate::NewBlob, Vec<PreparedBytes>)> {
        let chunker = hashed_blobs::Chunker::plain(hashed_blobs::ChunkSizes::default());
        let mut whole = blake3::Hasher::new();
        let mut size = 0u64;
        let mut chunks = Vec::new();
        let mut files: Vec<PreparedBytes> = Vec::new();
        for chunk in chunker.chunks(reader) {
            let chunk = chunk?;
            whole.update(&chunk);
            size += chunk.len() as u64;
            let hash = BlobHash::of(&chunk);
            if !files.iter().any(|file| file.hash() == hash) {
                files.push(self.prepare_bytes(chunk.as_slice())?);
            }
            chunks.push(crate::Chunk {
                hash,
                size: chunk.len() as u64,
            });
        }
        let hash = BlobHash(*whole.finalize().as_bytes());
        if chunks.len() == 1 {
            chunks.clear();
        }
        Ok((crate::NewBlob { hash, size, chunks }, files))
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

    /// The same prepared encoding serves standalone storage and the primary
    /// producer.
    pub fn prepare_bytes(&self, mut reader: impl std::io::Read) -> Result<PreparedBytes> {
        self.prepare_write(|output| {
            std::io::copy(&mut reader, output)?;
            Ok(())
        })
        .map(|(prepared, ())| prepared)
    }

    /// Stream generated bytes directly into the shared encoding. A refusal
    /// drops the private file before anything stores it.
    pub fn prepare_write<T>(
        &self,
        write: impl FnOnce(&mut dyn std::io::Write) -> Result<T>,
    ) -> Result<(PreparedBytes, T)> {
        let directory = self.staging_dir();
        std::fs::create_dir_all(&directory)?;
        let temporary = tempfile::NamedTempFile::new_in(&directory)?;
        let mut encoded = hashed_blobs::Preparation::new(&directory)?
            .plain_writer(temporary.as_file().try_clone()?, crate::blobs::LEVEL)?;
        let value = write(&mut encoded)?;
        let descriptor = encoded.finish()?.descriptor();
        Ok((
            PreparedBytes {
                owner: self.root.clone(),
                descriptor,
                temporary: std::sync::Arc::new(temporary.into_temp_path()),
            },
            value,
        ))
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

#[cfg(test)]
mod tests;
