//! Idempotent client requests: each request ID is stored with the hash of its
//! intent. What the request did is stored in its own tables.
use crate::{Error, Result, db};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::Path;
pub(crate) fn digest(intent: &serde_json::Value) -> Result<BlobHash> {
    let bytes = serde_json::to_vec(intent).map_err(|e| Error::Invariant(e.to_string()))?;
    Ok(BlobHash::of(&bytes))
}
/// Returns true if `id` was already recorded with the same intent hash; a
/// different hash is a conflict.
pub(crate) fn replay(conn: &Connection, path: &Path, id: &str, hash: BlobHash) -> Result<bool> {
    let old: Option<Vec<u8>> = db(path, || {
        conn.query_row(
            "SELECT content_hash FROM client_request WHERE id = ?1",
            [id],
            |r| r.get(0),
        )
        .optional()
    })?;
    match old {
        Some(old) if old.as_slice() != hash.bytes() => {
            Err(Error::ClientRequestConflict { id: id.to_owned() })
        }
        Some(_) => Ok(true),
        None => Ok(false),
    }
}
pub(crate) fn insert(
    conn: &Connection,
    path: &Path,
    id: &str,
    hash: BlobHash,
    created_at: i64,
) -> Result<usize> {
    db(path, || {
        conn.execute(
            "INSERT INTO client_request(id, content_hash, created_at) VALUES(?1, ?2, ?3)",
            params![id, hash.bytes().as_slice(), created_at],
        )
    })
}
