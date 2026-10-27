//! Packed readings and their shared text. Source rows choose what is shown.
use crate::{Result, encoding, first_build::FirstBuild};
use polylogue_model::Block;
use rusqlite::{Connection, params};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

pub(crate) fn store(
    conn: &Connection,
    kind: &str,
    hash: &[u8],
    body: &[u8],
    blocks: &[Block],
    packed: &HashMap<[u8; 32], Vec<u8>>,
    first_build: Option<&FirstBuild>,
) -> Result<()> {
    let inserted = conn
        .prepare_cached(&format!(
            "INSERT OR IGNORE INTO {kind}_version(version_hash,body) VALUES (?1,?2)"
        ))?
        .execute(params![hash, body])?;
    if inserted == 0 {
        return Ok(());
    }
    for (idx, block) in blocks.iter().enumerate() {
        if let Some(text) = encoding::block_text(block) {
            let id = crate::text_rows::intern(conn, text, 0, Some(packed), first_build)?;
            conn.prepare_cached(&format!(
                "INSERT INTO {kind}_version_text(version_hash,idx,text_id) VALUES (?1,?2,?3)"
            ))?
            .execute(params![hash, idx as i64, id])?;
        }
    }
    for block in blocks {
        let attachments: &[polylogue_model::Attachment] = match block {
            Block::Image(a) | Block::File(a) => std::slice::from_ref(a),
            Block::ToolResult { attachments, .. } => attachments,
            _ => &[],
        };
        for attachment in attachments {
            if let Some(blob) = attachment.blob {
                conn.prepare_cached(&format!("INSERT OR IGNORE INTO {kind}_version_blob(version_hash,blob_hash) VALUES (?1,?2)"))?.execute(params![hash,blob.0.as_slice()])?;
            }
        }
    }
    Ok(())
}

pub(crate) fn read<T: DeserializeOwned>(conn: &Connection, kind: &str, hash: &[u8]) -> Result<T> {
    let body: Vec<u8> = conn
        .prepare_cached(&format!(
            "SELECT body FROM {kind}_version WHERE version_hash=?1"
        ))?
        .query_row([hash], |r| r.get(0))?;
    let mut value: serde_json::Value = encoding::unpack_json(&body)?;
    let mut statement = conn.prepare_cached(&format!(
        "SELECT v.idx,t.body FROM {kind}_version_text v JOIN text t ON t.id=v.text_id WHERE v.version_hash=?1 ORDER BY v.idx"
    ))?;
    let mut rows = statement.query([hash])?;
    while let Some(row) = rows.next()? {
        let idx: i64 = row.get(0)?;
        let packed: Vec<u8> = row.get(1)?;
        let block = &mut value["blocks"][idx as usize];
        let field = if block.get("output").is_some() {
            "output"
        } else {
            "text"
        };
        block[field] = encoding::unpack_text(&packed)?.into();
    }
    Ok(serde_json::from_value(value)?)
}

/// Recheck only readings whose source references changed, using their hash indexes.
pub(crate) fn remove_readings(conn: &Connection) -> Result<()> {
    for kind in ["message", "document"] {
        conn.execute(&format!("DELETE FROM {kind}_version WHERE version_hash IN (SELECT version_hash FROM changed_version WHERE kind=?1)
            AND NOT EXISTS(SELECT 1 FROM {kind}_source s WHERE s.version_hash={kind}_version.version_hash)"), [kind])?;
        conn.execute("DELETE FROM changed_version WHERE kind=?1", [kind])?;
    }
    for kind in ["event", "link"] {
        conn.execute(&format!("DELETE FROM {kind}_value WHERE pk IN (SELECT value_pk FROM changed_value WHERE kind=?1)
            AND NOT EXISTS(SELECT 1 FROM {kind}_source s WHERE s.value_pk={kind}_value.pk)"), [kind])?;
        conn.execute("DELETE FROM changed_value WHERE kind=?1", [kind])?;
    }
    Ok(())
}

pub(crate) fn remove_unused(conn: &Connection, bulk: bool) -> Result<()> {
    remove_readings(conn)?;
    let mut statement = conn.prepare(
        "SELECT t.id,t.indexed FROM changed_text u JOIN text t ON t.id=u.text_id
         WHERE NOT EXISTS(SELECT 1 FROM block b WHERE b.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM document_block b WHERE b.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM message_version_text v WHERE v.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM document_version_text v WHERE v.text_id=t.id)",
    )?;
    let texts = statement
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (id, indexed) in texts {
        if bulk {
            conn.execute("DELETE FROM text WHERE id=?1", [id])?;
        } else {
            crate::text_rows::remove(conn, id, indexed)?;
        }
    }
    conn.execute("DELETE FROM changed_text", [])?;
    Ok(())
}
