//! Python Polylogue's vector stores: sqlite-vec `message_embeddings` tables,
//! read through their plain backing tables, so no extension is loaded.
//!
//! Two layouts exist. An addressed store names each vector by the address
//! of the request that bought it: SHA-256 over the domain
//! `polylogue.embedding-input.v1\0` and the model and NFC text, each
//! prefixed by its length as eight big-endian bytes. A store by message
//! names each vector by Python Polylogue's message ID and a hash of that
//! message's stored fields (position, role and parser output among them);
//! the text that was embedded cannot be recovered from it.

use super::{open_source, tables_of};
use crate::ResultExt as _;
use crate::row::{Rows as _, row};
use crate::{Result, StoreError};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Addressed,
    ByMessage,
}

/// One vector's identity in an addressed store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Addressed {
    pub address: [u8; 32],
    /// The provider's model name, as sent in the request.
    pub model: String,
    pub dims: usize,
    pub embedded_at: Option<i64>,
    /// Validated NFC request identity, available only in a marked prepared source.
    pub normalized_hash: Option<[u8; 32]>,
}

pub struct VectorStore {
    conn: Connection,
    layout: Layout,
    prepared: bool,
    _copies: Option<tempfile::TempDir>,
}

const ADDRESSED: [&str; 4] = [
    "embedding_input_hash",
    "model",
    "dimension",
    "embedded_at_ms",
];
const BY_MESSAGE: [&str; 5] = [
    "message_id",
    "model",
    "dimension",
    "content_hash",
    "embedded_at_ms",
];

row! {
    struct ApplicationId {
        application_id: i64,
    }
    statements {
        const READ = "PRAGMA application_id";
    }
}

row! {
    struct UserVersion {
        user_version: i64,
    }
    statements {
        const READ = "PRAGMA user_version";
    }
}

row! {
    struct AddressedVector {
        embedding_input_hash: Vec<u8>,
        model: String,
        dimension: i64,
        embedded_at_ms: Option<i64>,
        request_text: Option<String>,
    }
    statements {
        const PREPARED = "SELECT embedding_input_hash, model, dimension, embedded_at_ms, request_text FROM message_embeddings_meta";
        const ORIGINAL = "SELECT embedding_input_hash, model, dimension, embedded_at_ms, NULL AS request_text FROM message_embeddings_meta";
    }
}

row! {
    struct OriginCount {
        origin: String,
        count: i64,
    }
    statements {
        const ALL = "SELECT substr(message_id, 1, instr(message_id, ':') - 1) AS origin, count(*) AS count
             FROM message_embeddings_meta GROUP BY 1";
    }
}

row! {
    struct VectorSlot {
        id: String,
        chunk_id: i64,
        chunk_offset: i64,
    }
    statements {
        const ALL = "SELECT r.id, r.chunk_id, r.chunk_offset FROM message_embeddings_rowids r
             WHERE r.chunk_id IS NOT NULL ORDER BY r.chunk_id, r.chunk_offset";
    }
}

row! {
    struct VectorChunk {
        validity: Vec<u8>,
        vectors: Vec<u8>,
    }
    statements {
        const OF_ID = "SELECT c.validity, v.vectors FROM message_embeddings_chunks c
             JOIN message_embeddings_vector_chunks00 v ON v.rowid = c.chunk_id
             WHERE c.chunk_id = ?1";
    }
}

/// Python Polylogue's document request address. The caller supplies NFC text;
/// prepared sources validate that normalization before accepting its vectors.
pub fn request_address(model: &str, nfc_text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"polylogue.embedding-input.v1\0");
    for segment in [model.as_bytes(), nfc_text.as_bytes()] {
        hasher.update((segment.len() as u64).to_be_bytes());
        hasher.update(segment);
    }
    hasher.finalize().into()
}

impl VectorStore {
    pub fn open(path: &Path, scratch: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(polylogue_model::Reason::NotAFile {
                path: path.display().to_string(),
            }
            .into());
        }
        let mut copies = None;
        let conn = open_source(path, &mut copies, scratch)?;
        let tables = tables_of(&conn)?;
        let has = |table: &str, columns: &[&str]| {
            tables
                .get(table)
                .is_some_and(|have| columns.iter().all(|c| have.iter().any(|n| n == c)))
        };
        let backed = has(
            "message_embeddings_rowids",
            &["id", "chunk_id", "chunk_offset"],
        ) && has(
            "message_embeddings_chunks",
            &["chunk_id", "size", "validity"],
        ) && has("message_embeddings_vector_chunks00", &["rowid", "vectors"]);
        let layout = if backed && has("message_embeddings_meta", &ADDRESSED) {
            Layout::Addressed
        } else if backed && has("message_embeddings_meta", &BY_MESSAGE) {
            Layout::ByMessage
        } else {
            return Err(polylogue_model::Reason::NotAVectorStore {
                path: path.display().to_string(),
            }
            .into());
        };
        let application = conn.one(&ApplicationId::READ, [])?.application_id;
        let version = conn.one(&UserVersion::READ, [])?.user_version;
        let prepared = application == 0x504c5650;
        if prepared
            && (version != 1
                || layout != Layout::Addressed
                || !has("message_embeddings_meta", &["request_text"]))
        {
            return Err(StoreError::internal(
                "unsupported prepared vector source format",
            ));
        }
        conn.execute_batch("BEGIN")?;
        Ok(Self {
            conn,
            layout,
            prepared,
            _copies: copies,
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Every vector's identity in an addressed store.
    pub fn addressed(&self) -> Result<Vec<Addressed>> {
        if self.layout != Layout::Addressed {
            return Err(StoreError::stored_data(
                "a store by message names no addresses",
            ));
        }
        let statement = if self.prepared {
            AddressedVector::PREPARED
        } else {
            AddressedVector::ORIGINAL
        };
        let mut out = Vec::new();
        let _ = self.conn.each(&statement, [], |row| {
            let AddressedVector {
                embedding_input_hash: address,
                model,
                dimension: dims,
                embedded_at_ms: embedded_at,
                request_text: request,
            } = row;
            let address: [u8; 32] = address
                .try_into()
                .or_stored("a vector address is not 32 bytes")?;
            let dims = usize::try_from(dims)
                .ok()
                .filter(|dims| *dims > 0)
                .ok_or_else(|| {
                    StoreError::stored_data(format!("dimension {dims} is not a size"))
                })?;
            let normalized_hash = if self.prepared {
                let request = request.ok_or_else(|| {
                    StoreError::stored_data("prepared vector has no request text")
                })?;
                if !unicode_normalization::is_nfc(&request)
                    || address != request_address(&model, &request)
                {
                    return Err(StoreError::stored_data(
                        "prepared vector request does not prove its address",
                    ));
                }
                Some(crate::encoding::normalized_text_hash(&request))
            } else {
                None
            };
            out.push(Addressed {
                address,
                model,
                dims,
                embedded_at,
                normalized_hash,
            });
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(out)
    }

    /// Vectors per Python Polylogue origin in a store by message.
    pub fn by_message_counts(&self) -> Result<BTreeMap<String, u64>> {
        Ok(self
            .conn
            .all(&OriginCount::ALL, [])?
            .into_iter()
            .map(|row| (row.origin, row.count as u64))
            .collect())
    }

    /// Calls `visit` with each stored vector's key (the address in hex, or
    /// the message ID) and its values as little-endian four-byte floats,
    /// `dims` of them, reading each backing chunk once.
    pub fn for_each(
        &self,
        dims: usize,
        mut visit: impl FnMut(&str, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let width = dims * 4;
        let mut current: Option<(i64, VectorChunk)> = None;
        let _ = self.conn.each(&VectorSlot::ALL, [], |row| {
            let VectorSlot {
                id,
                chunk_id: chunk,
                chunk_offset,
            } = row;
            let offset = usize::try_from(chunk_offset).map_err(|_| {
                StoreError::stored_data(format!("vector {id} has a negative offset"))
            })?;
            if current.as_ref().is_none_or(|(held, _)| *held != chunk) {
                current = Some((chunk, self.conn.one(&VectorChunk::OF_ID, [chunk])?));
            }
            let (_, VectorChunk { validity, vectors }) = current.as_ref().expect("chunk read");
            let valid = validity
                .get(offset / 8)
                .is_some_and(|byte| byte & (1 << (offset % 8)) != 0);
            let values = vectors.get(offset * width..(offset + 1) * width);
            match (valid, values) {
                (true, Some(values)) => visit(&id, values)?,
                _ => {
                    return Err(StoreError::stored_data(format!(
                        "vector {id} is not held at its recorded place"
                    )));
                }
            }
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(
    addressed: &mut crate::row::Check<'_>,
    by_message: &mut crate::row::Check<'_>,
) {
    addressed.row::<ApplicationId>();
    addressed.row::<UserVersion>();
    addressed.row::<AddressedVector>();
    addressed.row::<VectorSlot>();
    addressed.row::<VectorChunk>();
    by_message.row::<OriginCount>();
}

#[cfg(test)]
mod tests;
