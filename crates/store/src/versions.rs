//! Retained message versions, sharing the text store with current blocks.
use crate::Result;
use crate::encoding::{
    PreparedContent, PreparedMessage, block_text, restore_text, unpack_block, unpack_json,
    unpack_text,
};
use polylogue_model::{Block, Document, Message};
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
    retain_content(tx, "message", prepared, &message.blocks, packed, bulk)
}

pub(crate) fn retain_content(
    tx: &Transaction<'_>,
    kind: &str,
    prepared: &PreparedContent,
    blocks: &[Block],
    packed: &HashMap<[u8; 32], Vec<u8>>,
    bulk: bool,
) -> Result<i64> {
    let hash = &prepared.hash[..];
    if let Some(pk) = tx
        .prepare_cached("SELECT pk FROM content_version WHERE hash = ?1 AND kind = ?2")?
        .query_row(params![hash, kind], |r| r.get::<_, i64>(0))
        .optional()?
    {
        return Ok(pk);
    }
    tx.prepare_cached("INSERT INTO content_version(hash, attrs, kind) VALUES (?1, ?2, ?3)")?
        .execute(params![hash, prepared.attrs, kind])?;
    // Text interning also inserts rows on this connection.
    let version_pk = tx.last_insert_rowid();
    for (idx, (block, body)) in blocks.iter().zip(&prepared.blocks).enumerate() {
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
        "SELECT pk, attrs FROM content_version WHERE hash = ?1 AND kind='message'",
        [hash],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let mut message: Message = unpack_json(&attrs)?;
    message.blocks = load_blocks(conn, version_pk)?;
    Ok((version_pk, message))
}

pub(crate) fn load_document(conn: &Connection, hash: &[u8]) -> Result<(i64, Document)> {
    let (version_pk, attrs): (i64, Vec<u8>) = conn.query_row(
        "SELECT pk,attrs FROM content_version WHERE hash=?1 AND kind='document'",
        [hash],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let mut document: Document = unpack_json(&attrs)?;
    document.blocks = load_blocks(conn, version_pk)?;
    Ok((version_pk, document))
}

fn load_blocks(conn: &Connection, version_pk: i64) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();
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
        blocks.push(block);
    }
    Ok(blocks)
}

pub(crate) fn prune(tx: &Transaction<'_>) -> Result<()> {
    tx.execute("DELETE FROM content_version WHERE NOT EXISTS (SELECT 1 FROM message_source s WHERE s.version_hash = content_version.hash) AND NOT EXISTS (SELECT 1 FROM document_source s WHERE s.version_hash=content_version.hash)", [])?;
    Ok(())
}

#[cfg(test)]
mod tests;
