//! Synthetic Python Polylogue vector stores, declared with the tables
//! sqlite-vec and Python Polylogue created.

use rusqlite::{Connection, params};
use std::path::Path;

/// The backing tables sqlite-vec declares for `message_embeddings`, as
/// Python Polylogue's stores hold them, and each layout's metadata table.
pub const VECTOR_BACKING_SQL: &str = r#"
CREATE TABLE "message_embeddings_info" (key text primary key, value any);
CREATE TABLE "message_embeddings_chunks"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,validity BLOB NOT NULL,rowids BLOB NOT NULL);
CREATE TABLE "message_embeddings_rowids"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,chunk_id INTEGER,chunk_offset INTEGER);
CREATE TABLE "message_embeddings_vector_chunks00"(rowid PRIMARY KEY,vectors BLOB NOT NULL);
"#;
pub const ADDRESSED_META_SQL: &str = r#"
CREATE TABLE message_embeddings_meta (
    embedding_input_hash   BLOB PRIMARY KEY CHECK(length(embedding_input_hash) = 32),
    model                  TEXT NOT NULL,
    dimension              INTEGER NOT NULL,
    embedded_at_ms         INTEGER,
    recipe_hash            BLOB CHECK(recipe_hash IS NULL OR length(recipe_hash) = 32),
    output_contract_hash   BLOB CHECK(output_contract_hash IS NULL OR length(output_contract_hash) = 32)
) STRICT;
"#;
pub const BY_MESSAGE_META_SQL: &str = r#"
CREATE TABLE message_embeddings_meta (
    message_id      TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    dimension       INTEGER NOT NULL,
    content_hash    BLOB NOT NULL CHECK(length(content_hash) = 32),
    embedded_at_ms  INTEGER,
    needs_reindex   INTEGER NOT NULL DEFAULT 0 CHECK(needs_reindex IN (0, 1))
) STRICT;
"#;

/// Vectors per backing chunk in these fixtures; sqlite-vec uses 1024.
pub const CHUNK: usize = 2;

/// One vector of a fixture store: its key (an address, or a message ID) and values.
pub enum VectorKey {
    Address([u8; 32]),
    Message(String),
}

/// Writes a vector store at `path` holding `vectors` of model `model`,
/// `CHUNK` to a backing chunk, with one empty slot left unset in the last
/// chunk when the count is odd. Every vector is embedded at `at`.
pub fn vector_store(path: &Path, model: &str, vectors: &[(VectorKey, Vec<f32>)], at: i64) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(VECTOR_BACKING_SQL).unwrap();
    let addressed = matches!(vectors.first(), Some((VectorKey::Address(_), _)));
    conn.execute_batch(if addressed {
        ADDRESSED_META_SQL
    } else {
        BY_MESSAGE_META_SQL
    })
    .unwrap();
    let dims = vectors.first().map_or(0, |(_, values)| values.len());
    for (chunk_index, chunk) in vectors.chunks(CHUNK).enumerate() {
        let chunk_id = chunk_index as i64 + 1;
        let mut validity = vec![0u8; CHUNK.div_ceil(8)];
        let mut bytes = vec![0u8; CHUNK * dims * 4];
        for (offset, (key, values)) in chunk.iter().enumerate() {
            validity[offset / 8] |= 1 << (offset % 8);
            for (i, value) in values.iter().enumerate() {
                let at = (offset * dims + i) * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
            let id = match key {
                VectorKey::Address(address) => {
                    let hex: String = address.iter().map(|b| format!("{b:02x}")).collect();
                    conn.execute(
                        "INSERT INTO message_embeddings_meta(embedding_input_hash, model, dimension, embedded_at_ms) VALUES (?1, ?2, ?3, ?4)",
                        params![address.to_vec(), model, dims as i64, at],
                    )
                    .unwrap();
                    hex
                }
                VectorKey::Message(id) => {
                    conn.execute(
                        "INSERT INTO message_embeddings_meta(message_id, model, dimension, content_hash, embedded_at_ms) VALUES (?1, ?2, ?3, ?4, ?5)",
                        params![id, model, dims as i64, vec![0u8; 32], at],
                    )
                    .unwrap();
                    id.clone()
                }
            };
            conn.execute(
                "INSERT INTO message_embeddings_rowids(id, chunk_id, chunk_offset) VALUES (?1, ?2, ?3)",
                params![id, chunk_id, offset as i64],
            )
            .unwrap();
        }
        conn.execute(
            "INSERT INTO message_embeddings_chunks(chunk_id, size, validity, rowids) VALUES (?1, ?2, ?3, ?4)",
            params![chunk_id, CHUNK as i64, validity, vec![0u8; CHUNK * 8]],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message_embeddings_vector_chunks00(rowid, vectors) VALUES (?1, ?2)",
            params![chunk_id, bytes],
        )
        .unwrap();
    }
}
