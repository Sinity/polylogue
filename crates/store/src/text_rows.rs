//! Stored text identity and its deferred full-text entries. Callers group row changes
//! in their own transaction; indexing consumes committed queue entries.

use crate::Result;
use crate::encoding::{pack_text, unpack_text};
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::HashMap;

/// The text row for `body`, stored once per distinct content, and its
/// entry in the search index `bit` names (1 prose, 2 tool output).
pub(crate) fn intern(
    conn: &Connection,
    body: &str,
    bit: i64,
    prepared: Option<&HashMap<[u8; 32], Vec<u8>>>,
    bulk: bool,
) -> Result<i64> {
    let hash = blake3::hash(body.as_bytes());
    let found: Option<(i64, i64)> = conn
        .prepare_cached("SELECT id, indexed FROM text WHERE hash = ?1")?
        .query_row([&hash.as_bytes()[..]], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    let (id, indexed) = match found {
        Some(f) => f,
        None => {
            let packed = match prepared.and_then(|p| p.get(hash.as_bytes())) {
                Some(p) => std::borrow::Cow::Borrowed(p.as_slice()),
                None => pack_text(body)?.into(),
            };
            // A first build marks the text for its search index now and
            // writes the entry when the build ends.
            let claimed = if bulk { bit } else { 0 };
            conn.prepare_cached("INSERT INTO text(hash, body, indexed) VALUES (?1, ?2, ?3)")?
                .execute(params![&hash.as_bytes()[..], &*packed, claimed])?;
            (conn.last_insert_rowid(), claimed)
        }
    };
    // `indexed` marks a text for its search index; the entry itself is
    // written after this transaction commits: by the search thread, or by
    // a first build in bulk when it ends.
    if bit != 0 && indexed & bit == 0 {
        if !bulk {
            conn.prepare_cached(
                "INSERT OR IGNORE INTO search_pending(text_id, fts_table) VALUES (?1, ?2)",
            )?
            .execute(params![id, bit])?;
        }
        conn.prepare_cached("UPDATE text SET indexed = indexed | ?2 WHERE id = ?1")?
            .execute(params![id, bit])?;
    }
    Ok(id)
}

/// Writes waiting search entries, at most `limit` per transaction, until
/// none wait; returns how many it wrote.
pub(crate) fn index_pending(conn: &Connection, limit: i64) -> Result<u64> {
    let mut written = 0;
    loop {
        // Immediate: it writes, and a deferred transaction that has read
        // cannot wait for the writer's lock to upgrade.
        let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        let rows: Vec<(i64, i64, Vec<u8>)> = tx
            .prepare_cached(
                "SELECT p.text_id, p.fts_table, t.body FROM search_pending p JOIN text t ON t.id = p.text_id
                 ORDER BY p.text_id, p.fts_table LIMIT ?1",
            )?
            .query_map([limit], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        if rows.is_empty() {
            // Entries whose text was removed meanwhile (excision) are dropped.
            tx.execute(
                "DELETE FROM search_pending WHERE text_id NOT IN (SELECT id FROM text)",
                [],
            )?;
            tx.commit()?;
            return Ok(written);
        }
        for (id, bit, packed) in &rows {
            let folded = polylogue_model::text::fold(&unpack_text(packed)?);
            let sql = if *bit == 1 {
                "INSERT INTO text_fts(rowid, body) VALUES (?1, ?2)"
            } else {
                "INSERT INTO output_fts(rowid, body) VALUES (?1, ?2)"
            };
            tx.prepare_cached(sql)?.execute(params![id, folded])?;
            tx.prepare_cached("DELETE FROM search_pending WHERE text_id = ?1 AND fts_table = ?2")?
                .execute(params![id, bit])?;
        }
        written += rows.len() as u64;
        tx.commit()?;
    }
}
/// Historical versions keep their text row, but only current blocks contribute
/// documents to full-text ranking or embedding candidates.
pub(crate) fn unindex_unused(i: &Connection, id: i64, bulk: bool) -> Result<()> {
    for (bit, kind, table) in [
        (1, "b.kind<>'tool_result'", "text_fts"),
        (2, "b.kind='tool_result'", "output_fts"),
    ] {
        let used:bool=i.query_row(&format!("SELECT EXISTS(SELECT 1 FROM block b JOIN current_message m ON m.pk=b.message_pk WHERE b.text_id=?1 AND {kind})"),[id],|r|r.get(0))?;
        if !used {
            if !bulk {
                i.execute(&format!("DELETE FROM {table} WHERE rowid=?1"), [id])?;
            }
            i.execute(
                "DELETE FROM search_pending WHERE text_id=?1 AND fts_table=?2",
                params![id, bit],
            )?;
            i.execute(
                "UPDATE text SET indexed=indexed & ~?2 WHERE id=?1",
                params![id, bit],
            )?;
        }
    }
    Ok(())
}

pub(crate) fn remove(i: &Connection, id: i64, indexed: i64) -> Result<()> {
    if indexed & 1 != 0 {
        i.execute("DELETE FROM text_fts WHERE rowid = ?1", [id])?;
    }
    if indexed & 2 != 0 {
        i.execute("DELETE FROM output_fts WHERE rowid = ?1", [id])?;
    }
    i.execute("DELETE FROM search_pending WHERE text_id = ?1", [id])?;
    i.execute("DELETE FROM text WHERE id = ?1", [id])?;
    Ok(())
}

#[cfg(test)]
mod tests;
