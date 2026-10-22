//! Retained message versions, sharing the text store with current blocks.
use crate::Result;
use crate::encoding::{
    PreparedMessage, block_text, restore_text, unpack_block, unpack_json, unpack_text,
};
use polylogue_model::{Block, Message};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use std::collections::HashMap;

/// Store a version once and return its local row ID.
pub(crate) fn retain(
    tx: &Transaction<'_>,
    prepared: &PreparedMessage,
    message: &Message,
    packed: &HashMap<[u8; 32], Vec<u8>>,
    bulk: bool,
) -> Result<i64> {
    let hash = &prepared.hash[..];
    if let Some(pk) = tx
        .prepare_cached("SELECT pk FROM message_version WHERE hash = ?1")?
        .query_row([hash], |r| r.get::<_, i64>(0))
        .optional()?
    {
        return Ok(pk);
    }
    tx.prepare_cached("INSERT INTO message_version(hash, attrs) VALUES (?1, ?2)")?
        .execute(params![hash, prepared.attrs])?;
    // Text interning also inserts rows on this connection.
    let version_pk = tx.last_insert_rowid();
    for (idx, (block, body)) in message.blocks.iter().zip(&prepared.blocks).enumerate() {
        let text_id = block_text(block)
            .map(|text| crate::text_rows::intern(tx, text, 0, Some(packed), bulk))
            .transpose()?;
        tx.prepare_cached(
            "INSERT INTO version_block(version_pk, idx, body, text_id, blob_hash) VALUES (?1, ?2, ?3, ?4, ?5)")?
            .execute(params![version_pk, idx as i64, body, text_id,match block {Block::Image(a)|Block::File(a)=>a.blob.map(|h|h.0.to_vec()),_=>None}],
        )?;
    }
    Ok(version_pk)
}

/// The local row ID and decoded message for a logical content hash.
pub(crate) fn load(conn: &Connection, hash: &[u8]) -> Result<(i64, Message)> {
    let (version_pk, attrs): (i64, Vec<u8>) = conn.query_row(
        "SELECT pk, attrs FROM message_version WHERE hash = ?1",
        [hash],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let mut message: Message = unpack_json(&attrs)?;
    let mut stmt = conn.prepare("SELECT b.body, t.body FROM version_block b LEFT JOIN text t ON t.id = b.text_id WHERE b.version_pk = ?1 ORDER BY b.idx")?;
    let rows = stmt.query_map([version_pk], |row| {
        Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<Vec<u8>>>(1)?))
    })?;
    for row in rows {
        let (body, text) = row?;
        let mut block = unpack_block(&body)?;
        if let Some(text) = text {
            restore_text(&mut block, unpack_text(&text)?);
        }
        message.blocks.push(block);
    }
    Ok((version_pk, message))
}

pub(crate) fn prune(tx: &Transaction<'_>) -> Result<()> {
    tx.execute("DELETE FROM message_version WHERE NOT EXISTS (SELECT 1 FROM message_source s WHERE s.version_hash = message_version.hash)", [])?;
    Ok(())
}

#[cfg(test)]
mod tests;
