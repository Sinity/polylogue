//! Stored texts and their deferred full-text entries. A text row is found
//! by its hash, or inserted under the next key AUTOINCREMENT gives, so a key a
//! removed text had is never given to another text while an entry may still
//! name it. Which search indexes a text belongs to follows the blocks that
//! hold it, through the index schema's triggers (`schema/search_triggers.sql`);
//! the search thread writes the queued entries after the writer commits.

use crate::ConnExt as _;
use crate::{Result, StoreError};
mod groups;
mod search_build;
pub(crate) use groups::{PreparedTexts, column, joined, read, slice};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
pub(crate) use search_build::SearchBuild;
#[cfg(test)]
pub(crate) use search_build::build_search_indexes;

/// The row of a complete text hash: the stored one, or a new row holding
/// the prepared body.
pub(crate) fn resolve(
    conn: &Connection,
    hash: &[u8; 32],
    prepared: Option<groups::Prepared<'_>>,
) -> Result<i64> {
    let stored = conn
        .prepare_cached("SELECT id FROM text WHERE hash=?1")?
        .query_row([hash.as_slice()], |row| row.get::<_, i64>(0))
        .optional()?;
    if let Some(id) = stored {
        return Ok(id);
    }
    let prepared =
        prepared.ok_or_else(|| StoreError::Invalid("a new text has no prepared bytes".into()))?;
    groups::insert(conn, hash, prepared)
}

/// The text row for `body`, stored once per distinct content.
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn intern(conn: &Connection, body: &str) -> Result<i64> {
    let hash = blake3::hash(body.as_bytes());
    let prepared = PreparedTexts::pack([body])?;
    resolve(conn, hash.as_bytes(), prepared.get(hash.as_bytes()))
}

/// Makes text `id` a member of the search index `bit` names and queues its
/// entry, as a block of that kind holding it would.
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn mark_searched(conn: &Connection, id: i64, bit: i64) -> Result<()> {
    let changed = conn.execute(
        "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1 AND indexed & ?2 = 0",
        params![id, bit],
    )?;
    if changed != 0 {
        conn.execute(
            "INSERT OR IGNORE INTO search_pending(text_id, fts_table) VALUES (?1, ?2)",
            params![id, bit],
        )?;
    }
    Ok(())
}

// Limit each batch of search entries independently of compressed row sizes.
// A single oversized text is still decoded whole and ends its batch.
const SEARCH_ROWS: i64 = 2_000;
pub(crate) const SEARCH_BYTES: usize = 16 * 1024 * 1024;

/// Search entries computed from committed rows on a read connection, for
/// the index writer to apply ([`apply_entries`]).
#[derive(Debug, Default)]
pub(crate) struct SearchEntries {
    entries: Vec<SearchEntry>,
    /// The queue holds entries whose text rows are gone.
    orphans: bool,
}

#[derive(Debug)]
struct SearchEntry {
    text_id: i64,
    fts_table: i64,
    /// The text's content hash when it was read: an entry applies only to
    /// the same content.
    hash: Vec<u8>,
    folded: String,
}

impl SearchEntries {
    /// Nothing to apply: the queue was empty when it was read.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty() && !self.orphans
    }
}

/// Reads the next queued search entries and their folded text, up to
/// `byte_limit` of text, in one read transaction. Writes nothing.
pub(crate) fn pending_entries(conn: &Connection, byte_limit: usize) -> Result<SearchEntries> {
    let tx = conn.unchecked_transaction()?;
    // Read the ordered queue first; a join can instead scan all historical
    // texts and sort on every batch. Finish the cursor before reading
    // bodies, which are fetched individually, so the row limit does not
    // multiply text held.
    let ids: Vec<(i64, i64)> = tx
        .prepare_cached(
            "SELECT p.text_id, p.fts_table FROM search_pending p
             WHERE EXISTS(SELECT 1 FROM text t WHERE t.id = p.text_id)
             ORDER BY p.text_id, p.fts_table LIMIT ?1",
        )?
        .query_map([SEARCH_ROWS], |r| <(_, _)>::try_from(r))?
        .collect::<rusqlite::Result<_>>()?;
    let orphans = ids.is_empty() && tx.exists("SELECT EXISTS(SELECT 1 FROM search_pending)", [])?;
    let mut entries = Vec::new();
    let mut bytes = 0;
    for (text_id, fts_table) in ids {
        let hash = tx
            .prepare_cached("SELECT hash FROM text WHERE id = ?1")?
            .query_row([text_id], |r| r.get(0))?;
        let (folded, length) = folded_text(read(&tx, text_id)?);
        bytes += length;
        entries.push(SearchEntry {
            text_id,
            fts_table,
            hash,
            folded,
        });
        if bytes >= byte_limit {
            break;
        }
    }
    tx.finish()?;
    Ok(SearchEntries { entries, orphans })
}

/// Writes computed search entries in one transaction, with the number
/// written. An entry whose queue row is gone, or whose text now holds other
/// content, was settled by a write after it was read and is skipped.
pub(crate) fn apply_entries(conn: &Connection, computed: &SearchEntries) -> Result<u64> {
    // Immediate: a deferred reader cannot wait to upgrade its writer lock.
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let before = tx.total_changes();
    if computed.orphans {
        tx.execute(
            "DELETE FROM search_pending WHERE text_id NOT IN (SELECT id FROM text)",
            [],
        )?;
    }
    let mut written = 0;
    for entry in &computed.entries {
        let removed = tx
            .prepare_cached(
                "DELETE FROM search_pending WHERE text_id = ?1 AND fts_table = ?2
                 AND EXISTS(SELECT 1 FROM text WHERE id = ?1 AND hash = ?3)",
            )?
            .execute(params![entry.text_id, entry.fts_table, entry.hash])?;
        if removed != 0 {
            insert_entry(&tx, entry.text_id, entry.fts_table, &entry.folded)?;
            written += 1;
        }
    }
    if tx.total_changes() != before {
        crate::archive::bump_counter(&tx)?;
    }
    tx.commit()?;
    Ok(written)
}

/// Drains committed search work in bounded transactions on one connection
/// that both reads and writes the index.
pub(crate) fn index_pending(conn: &Connection) -> Result<u64> {
    let mut written = 0;
    loop {
        let count = pending_batch(conn, SEARCH_BYTES)?;
        written += count;
        if count == 0 {
            return Ok(written);
        }
    }
}

fn pending_batch(conn: &Connection, byte_limit: usize) -> Result<u64> {
    apply_entries(conn, &pending_entries(conn, byte_limit)?)
}

/// A stored text as the search indexes hold it, and its length in bytes.
fn folded_text(text: String) -> (String, usize) {
    let length = text.len();
    let folded = if text.is_ascii() {
        text
    } else {
        polylogue_model::text::fold(&text)
    };
    (folded, length)
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

/// Removes a text row with its search entries; its group leaves with its
/// last text.
pub(crate) fn remove(i: &Connection, id: i64) -> Result<()> {
    i.execute("DELETE FROM text_fts WHERE rowid = ?1", [id])?;
    i.execute("DELETE FROM output_fts WHERE rowid = ?1", [id])?;
    i.execute("DELETE FROM search_pending WHERE text_id = ?1", [id])?;
    groups::delete(i, id)
}

/// A group body holding only `text`.
#[cfg(test)]
pub(crate) fn alone(text: &str) -> Vec<u8> {
    zstd::bulk::compress(text.as_bytes(), 1).unwrap()
}

/// The group holding only `text`, stored once.
#[cfg(test)]
fn alone_group(conn: &Connection, text: &str) -> i64 {
    let body = alone(text);
    let hash = blake3::hash(&body);
    conn.execute(
        "INSERT INTO text_group(hash,body) VALUES (?1,?2) ON CONFLICT(hash) DO NOTHING",
        params![hash.as_bytes().as_slice(), body],
    )
    .unwrap();
    conn.query_row(
        "SELECT id FROM text_group WHERE hash=?1",
        [hash.as_bytes().as_slice()],
        |r| r.get(0),
    )
    .unwrap()
}

/// Inserts a text row, under the hashes given, holding `text` in a group of
/// its own.
#[cfg(test)]
pub(crate) fn insert_alone(
    conn: &Connection,
    id: Option<i64>,
    hash: &[u8],
    normalized_hash: &[u8],
    text: &str,
    indexed: i64,
) -> i64 {
    let group = alone_group(conn, text);
    conn.execute(
        "INSERT INTO text(id,hash,normalized_hash,indexed,group_id,start,length)
         VALUES (?1,?2,?3,?4,?5,0,?6)",
        params![id, hash, normalized_hash, indexed, group, text.len() as i64],
    )
    .unwrap();
    conn.last_insert_rowid()
}

/// Points text `id` at a new group holding only `text`, leaving its hashes.
#[cfg(test)]
pub(crate) fn replace_text(conn: &Connection, id: i64, text: &str) {
    let group = alone_group(conn, text);
    let changed = conn
        .execute(
            "UPDATE text SET group_id=?2, start=0, length=?3 WHERE id=?1",
            params![id, group, text.len() as i64],
        )
        .unwrap();
    assert_eq!(changed, 1);
}

/// Replaces the body of the group holding text `id`.
#[cfg(test)]
pub(crate) fn replace_group_body(conn: &Connection, id: i64, body: &[u8]) {
    let changed = conn
        .execute(
            "UPDATE text_group SET body=?2 WHERE id=(SELECT group_id FROM text WHERE id=?1)",
            params![id, body],
        )
        .unwrap();
    assert_eq!(changed, 1);
}

#[cfg(test)]
mod tests;
