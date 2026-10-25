//! Bytes hashed off the writer thread into a private file, waiting for the
//! writer to give them their rows: in standalone mode compressed, then moved
//! into the blob store; in primary mode as they are, then sent to Sinex.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;

/// One pool batch and the private files its pieces were written to. Inline
/// pieces stay in the batch.
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
        store: std::sync::Arc<hashed_blobs::Store>,
        prepared: hashed_blobs::PreparedPlain,
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
    /// An input's bytes: small ones inline in standalone mode, the rest in a
    /// private file. In primary mode every piece is a file, sent to Sinex.
    pub fn prepare_piece(&self, bytes: &[u8]) -> Result<(crate::Piece, Option<PreparedBytes>)> {
        if bytes.len() < crate::blobs::INLINE_LIMIT && !self.is_primary() {
            return inline(bytes);
        }
        let prepared = self.prepare_bytes(bytes)?;
        Ok((
            crate::Piece {
                hash: prepared.hash(),
                size: prepared.len(),
                stored: crate::blobs::Stored::File { stored_size: 0 },
            },
            Some(prepared),
        ))
    }

    /// Bytes no provider file holds as they are: what a parser decoded from
    /// inside a record (a base64 image), or a file uploaded for a chat
    /// action. A parse makes them again, so in primary mode they stay in
    /// the register inline and are not sent to Sinex.
    pub fn prepare_decoded(&self, bytes: &[u8]) -> Result<(crate::Piece, Option<PreparedBytes>)> {
        if self.is_primary() {
            return inline(bytes);
        }
        self.prepare_piece(bytes)
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
        let mut scratch = tempfile::tempfile()?;
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
            std::fs::create_dir_all(self.parts_dir())?;
            let mut file = tempfile::NamedTempFile::new_in(self.parts_dir())?;
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
            file.as_file().sync_all()?;
            return Ok(PreparedBytes {
                owner: self.root.clone(),
                file: Prepared::Part {
                    file,
                    hash: BlobHash(*hasher.finalize().as_bytes()),
                    length,
                },
            });
        }
        let store = std::sync::Arc::new(self.blobs()?);
        let prepared =
            store.prepare_plain_checked::<StoreError>(reader, crate::blobs::LEVEL, || Ok(()))?;
        Ok(PreparedBytes {
            owner: self.root.clone(),
            file: Prepared::Local { store, prepared },
        })
    }
}

fn inline(bytes: &[u8]) -> Result<(crate::Piece, Option<PreparedBytes>)> {
    Ok((
        crate::Piece {
            hash: BlobHash::of(bytes),
            size: bytes.len() as u64,
            stored: crate::blobs::Stored::Inline(zstd::bulk::compress(bytes, crate::blobs::LEVEL)?),
        },
        None,
    ))
}

/// Where a stored batch's bytes are: its input and revision chain.
#[derive(Debug, Clone)]
pub struct StoredInput {
    pub input: String,
    pub chain: i64,
    pub length: u64,
    pub already_stored: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingState {
    Writing,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PendingBlob {
    pub input: String,
    pub blob: BlobHash,
    pub length: u64,
    pub state: PendingState,
}

/// What finishing blobs left `writing` by a stop found.
#[derive(Debug, Default)]
pub struct Finished {
    pub completed: u64,
    pub pending: Vec<PendingBlob>,
}

pub(crate) fn require_present(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: Option<i64>,
) -> Result<()> {
    check(conn, schema, input, chain, None)
}

pub(crate) fn require_range(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: i64,
    range: std::ops::Range<u64>,
) -> Result<()> {
    check(conn, schema, input, Some(chain), Some(range))
}

fn check(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: Option<i64>,
    range: Option<std::ops::Range<u64>>,
) -> Result<()> {
    let position = |value| {
        i64::try_from(value)
            .map_err(|_| StoreError::Invalid("byte range exceeds archive address space".into()))
    };
    let from = range
        .as_ref()
        .map(|range| position(range.start))
        .transpose()?;
    let until = range
        .as_ref()
        .map(|range| position(range.end))
        .transpose()?;
    let sql=format!("SELECT b.state FROM {schema}extent e JOIN {schema}blob b ON b.hash=e.blob WHERE e.input=?1 AND (?2 IS NULL OR e.chain=?2)
        AND (?3 IS NULL OR (e.offset<?4 AND e.offset+e.length>?3) OR (?3=?4 AND e.offset=?3 AND e.length=0))
        AND b.state<>'present' LIMIT 1");
    let pending: Option<String> = conn
        .query_row(&sql, rusqlite::params![input, chain, from, until], |row| {
            row.get(0)
        })
        .optional()?;
    if let Some(state) = pending {
        return Err(StoreError::InputPending {
            input: input.into(),
            reason: format!("its bytes are {state}"),
        });
    }
    Ok(())
}

pub(crate) fn pending_blobs(conn: &Connection, schema: &str) -> Result<Vec<PendingBlob>> {
    let sql=format!("SELECT input,hash,size FROM (
        SELECT DISTINCT i.id AS input,b.hash,b.size FROM {schema}input i JOIN {schema}revision r ON r.input=i.id
        JOIN {schema}extent e ON e.input=r.input AND e.chain=r.chain JOIN {schema}blob b ON b.hash=e.blob WHERE b.state='writing')
        ORDER BY input,hash");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let hash: Vec<u8> = row.get(1)?;
        let length = row.get::<_, i64>(2)?;
        out.push(PendingBlob {
            input: row.get(0)?,
            blob: BlobHash(
                hash.try_into()
                    .map_err(|_| StoreError::Invalid("invalid pending blob hash".into()))?,
            ),
            length: u64::try_from(length)
                .map_err(|_| StoreError::Invalid("negative pending blob length".into()))?,
            state: PendingState::Writing,
        });
    }
    Ok(out)
}
