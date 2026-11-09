//! The predecessor's vector stores: sqlite-vec `message_embeddings` tables,
//! read through their plain backing tables, so no extension is loaded.
//!
//! Two layouts exist. An addressed store names each vector by the address
//! of the request that bought it: SHA-256 over the domain
//! `polylogue.embedding-input.v1\0` and the model and NFC text, each
//! prefixed by its length as eight big-endian bytes. A store by message
//! names each vector by the predecessor's message ID and a hash of that
//! message's stored fields (position, role and parser output among them);
//! the text that was embedded cannot be recovered from it.

use crate::older_archive::{open_source, tables_of};
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::collections::BTreeMap;
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
}

pub struct VectorStore {
    conn: Connection,
    layout: Layout,
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

impl VectorStore {
    pub fn open(path: &Path, scratch: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(StoreError::Invalid(format!(
                "{} is not a file: name one of the predecessor's vector stores",
                path.display()
            )));
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
            return Err(StoreError::Invalid(format!(
                "{} is not one of the predecessor's vector stores: it has no message_embeddings tables in either layout",
                path.display()
            )));
        };
        conn.execute_batch("BEGIN")?;
        Ok(Self {
            conn,
            layout,
            _copies: copies,
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Every vector's identity in an addressed store.
    pub fn addressed(&self) -> Result<Vec<Addressed>> {
        if self.layout != Layout::Addressed {
            return Err(StoreError::Invalid(
                "a store by message names no addresses".into(),
            ));
        }
        let mut statement = self.conn.prepare(
            "SELECT embedding_input_hash, model, dimension, embedded_at_ms FROM message_embeddings_meta",
        )?;
        let rows = statement.query_map([], |r| {
            Ok((
                r.get::<_, Vec<u8>>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, Option<i64>>(3)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (address, model, dims, embedded_at) = row?;
            let address: [u8; 32] = address
                .try_into()
                .map_err(|_| StoreError::Invalid("a vector address is not 32 bytes".into()))?;
            let dims = usize::try_from(dims)
                .ok()
                .filter(|dims| *dims > 0)
                .ok_or_else(|| StoreError::Invalid(format!("dimension {dims} is not a size")))?;
            out.push(Addressed {
                address,
                model,
                dims,
                embedded_at,
            });
        }
        Ok(out)
    }

    /// Vectors per predecessor origin in a store by message.
    pub fn by_message_counts(&self) -> Result<BTreeMap<String, u64>> {
        let mut statement = self.conn.prepare(
            "SELECT substr(message_id, 1, instr(message_id, ':') - 1), count(*)
             FROM message_embeddings_meta GROUP BY 1",
        )?;
        let rows = statement.query_map([], |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as u64)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
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
        let mut rows = self.conn.prepare(
            "SELECT r.id, r.chunk_id, r.chunk_offset FROM message_embeddings_rowids r
             WHERE r.chunk_id IS NOT NULL ORDER BY r.chunk_id, r.chunk_offset",
        )?;
        let mut chunk_of = self.conn.prepare(
            "SELECT c.validity, v.vectors FROM message_embeddings_chunks c
             JOIN message_embeddings_vector_chunks00 v ON v.rowid = c.chunk_id
             WHERE c.chunk_id = ?1",
        )?;
        let mut current: Option<(i64, Vec<u8>, Vec<u8>)> = None;
        let mut rows = rows.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let chunk: i64 = row.get(1)?;
            let offset = usize::try_from(row.get::<_, i64>(2)?)
                .map_err(|_| StoreError::Invalid(format!("vector {id} has a negative offset")))?;
            if current.as_ref().is_none_or(|(held, _, _)| *held != chunk) {
                let (validity, vectors) =
                    chunk_of.query_row([chunk], |r| Ok((r.get(0)?, r.get(1)?)))?;
                current = Some((chunk, validity, vectors));
            }
            let (_, validity, vectors) = current.as_ref().expect("chunk read");
            let valid = validity
                .get(offset / 8)
                .is_some_and(|byte| byte & (1 << (offset % 8)) != 0);
            let values = vectors.get(offset * width..(offset + 1) * width);
            match (valid, values) {
                (true, Some(values)) => visit(&id, values)?,
                _ => {
                    return Err(StoreError::Invalid(format!(
                        "vector {id} is not held at its recorded place"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
