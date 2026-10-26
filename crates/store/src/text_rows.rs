//! Stored text identity and its deferred full-text entries. Callers group row changes
//! in their own transaction; indexing consumes committed queue entries.

use crate::Result;
use crate::encoding::{pack_text, unpack_text};
use crate::first_build::FirstBuild;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use std::collections::HashMap;

/// The text row for `body`, stored once per distinct content, and its
/// entry in the search index `bit` names (1 prose, 2 tool output). During a
/// first build the text is found through `first_build`, not its hash index.
pub(crate) fn intern(
    conn: &Connection,
    body: &str,
    bit: i64,
    prepared: Option<&HashMap<[u8; 32], Vec<u8>>>,
    first_build: Option<&FirstBuild>,
) -> Result<i64> {
    let hash = blake3::hash(body.as_bytes());
    let id = match first_build {
        Some(first_build) => first_build.text(conn, hash.as_bytes())?,
        None => conn
            .prepare_cached("SELECT id FROM text WHERE hash = ?1")?
            .query_row([&hash.as_bytes()[..]], |r| r.get(0))
            .optional()?,
    };
    let bulk = first_build.is_some();
    let (id, indexed) = match id {
        Some(id) => (
            id,
            conn.prepare_cached("SELECT indexed FROM text WHERE id = ?1")?
                .query_row([id], |r| r.get(0))?,
        ),
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
            let id = conn.last_insert_rowid();
            if let Some(first_build) = first_build {
                first_build.add_text(hash.as_bytes(), id);
            }
            (id, claimed)
        }
    };
    // `indexed` marks a text for its search index; the entry itself is
    // written after this transaction commits: by the search thread, or by
    // a first build in bulk when it ends.
    if bit != 0 && indexed & bit == 0 {
        activate(conn, id, bit, bulk)?;
    }
    Ok(id)
}

/// Marks an already stored text for current-block search membership.
pub(crate) fn activate(conn: &Connection, id: i64, bit: i64, bulk: bool) -> Result<()> {
    let changed = conn
        .prepare_cached(
            "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1 AND indexed & ?2 = 0",
        )?
        .execute(params![id, bit])?;
    if changed != 0 && !bulk {
        conn.prepare_cached(
            "INSERT OR IGNORE INTO search_pending(text_id, fts_table) VALUES (?1, ?2)",
        )?
        .execute(params![id, bit])?;
    }
    Ok(())
}

// Limit transaction work independently of compressed row sizes. A single
// oversized text is still decoded whole and ends its transaction.
const SEARCH_ROWS: i64 = 2_000;
const SEARCH_BYTES: usize = 16 * 1024 * 1024;

/// Writes the first build's marked texts in ID order, both search indexes
/// from one scan of `text`. Only the FTS tables change while its source
/// cursor is open.
pub(crate) fn index_marked(conn: &Connection) -> Result<()> {
    let mut after = 0;
    loop {
        let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        let mut bytes = 0;
        let mut last = after;
        {
            let mut select = tx.prepare_cached(
                "SELECT id, indexed, body FROM text WHERE id > ?1 AND indexed <> 0
                 ORDER BY id LIMIT ?2",
            )?;
            let mut rows = select.query(params![after, SEARCH_ROWS])?;
            while let Some(row) = rows.next()? {
                last = row.get(0)?;
                let indexed: i64 = row.get(1)?;
                let packed: Vec<u8> = row.get(2)?;
                let (folded, length) = folded_text(&packed)?;
                for bit in [1, 2].into_iter().filter(|bit| indexed & bit != 0) {
                    insert_entry(&tx, last, bit, &folded)?;
                }
                bytes += length;
                if bytes >= SEARCH_BYTES {
                    break;
                }
            }
        }
        if last != after {
            crate::archive::bump_counter(&tx)?;
        }
        tx.commit()?;
        if last == after {
            return Ok(());
        }
        after = last;
    }
}

/// Drains committed search work in bounded transactions, each inside what
/// `turn` returns (the search thread's turn at writing the index).
pub(crate) fn index_pending<T>(conn: &Connection, mut turn: impl FnMut() -> T) -> Result<u64> {
    let mut written = 0;
    loop {
        let held = turn();
        let count = pending_batch(conn, SEARCH_BYTES)?;
        drop(held);
        written += count;
        if count == 0 {
            return Ok(written);
        }
    }
}

fn pending_batch(conn: &Connection, byte_limit: usize) -> Result<u64> {
    // Immediate: a deferred reader cannot wait to upgrade its writer lock.
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let before = tx.total_changes();
    // Finish the queue cursor before removing its entries. Bodies are fetched
    // individually, so the row-count limit does not multiply the text held.
    let ids: Vec<(i64, i64)> = tx
        .prepare_cached(
            "SELECT p.text_id, p.fts_table FROM search_pending p JOIN text t ON t.id = p.text_id
             ORDER BY p.text_id, p.fts_table LIMIT ?1",
        )?
        .query_map([SEARCH_ROWS], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    if ids.is_empty() {
        tx.execute(
            "DELETE FROM search_pending WHERE text_id NOT IN (SELECT id FROM text)",
            [],
        )?;
    }
    let mut written = 0;
    let mut bytes = 0;
    for (id, bit) in ids {
        {
            let mut select = tx.prepare_cached("SELECT body FROM text WHERE id = ?1")?;
            let mut rows = select.query([id])?;
            let row = rows.next()?.ok_or(rusqlite::Error::QueryReturnedNoRows)?;
            let packed: Vec<u8> = row.get(0)?;
            bytes += write_entry(&tx, id, bit, &packed)?;
        }
        tx.prepare_cached("DELETE FROM search_pending WHERE text_id = ?1 AND fts_table = ?2")?
            .execute(params![id, bit])?;
        written += 1;
        if bytes >= byte_limit {
            break;
        }
    }
    if tx.total_changes() != before {
        crate::archive::bump_counter(&tx)?;
    }
    tx.commit()?;
    Ok(written)
}

fn write_entry(conn: &Connection, id: i64, bit: i64, packed: &[u8]) -> Result<usize> {
    let (folded, length) = folded_text(packed)?;
    insert_entry(conn, id, bit, &folded)?;
    Ok(length)
}

/// A stored text as the search indexes hold it, and its length in bytes.
fn folded_text(packed: &[u8]) -> Result<(String, usize)> {
    let text = unpack_text(packed)?;
    let length = text.len();
    let folded = if text.is_ascii() {
        text
    } else {
        polylogue_model::text::fold(&text)
    };
    Ok((folded, length))
}

fn insert_entry(conn: &Connection, id: i64, bit: i64, folded: &str) -> Result<()> {
    let sql = if bit == 1 {
        "INSERT INTO text_fts(rowid, body) VALUES (?1, ?2)"
    } else {
        "INSERT INTO output_fts(rowid, body) VALUES (?1, ?2)"
    };
    conn.prepare_cached(sql)?.execute(params![id, folded])?;
    Ok(())
}

/// Historical versions keep their text row, but only current blocks contribute
/// documents to full-text ranking or embedding.
pub(crate) fn unindex_unused(i: &Connection, id: i64, bulk: bool) -> Result<()> {
    for (bit, kind, table) in [
        (1, "b.kind<>'tool_result'", "text_fts"),
        (2, "b.kind='tool_result'", "output_fts"),
    ] {
        let used:bool=i.query_row(&format!("SELECT EXISTS(SELECT 1 FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.text_id=?1 AND {kind}) OR (?2=1 AND EXISTS(SELECT 1 FROM document_block b JOIN document d ON d.pk=b.document_pk WHERE b.text_id=?1))"),params![id,bit],|r|r.get(0))?;
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
