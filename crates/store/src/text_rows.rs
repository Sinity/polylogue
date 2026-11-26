//! Stored texts and their deferred full-text entries. A text row is found
//! by its hash, or inserted under the next key AUTOINCREMENT gives, so a key a
//! removed text had is never given to another text while an entry may still
//! name it. Which search indexes a text belongs to follows the blocks that
//! hold it, through the index schema's triggers (`schema/search_triggers.sql`);
//! the search thread computes the queued changes to entries after the writer
//! commits, and the writer applies them to the full-text files.

use crate::{Result, StoreError};
mod groups;
mod search_fill;
use crate::row::{Rows as _, row};
pub(crate) use groups::{PreparedTexts, column, joined, read, slice};
use rusqlite::{Connection, TransactionBehavior, params};
pub(crate) use search_fill::SearchFill;

/// The row of a complete text hash: the stored one, or a new row holding
/// the prepared body.
pub(crate) fn resolve(
    conn: &Connection,
    hash: &[u8; 32],
    prepared: Option<groups::Prepared<'_>>,
) -> Result<i64> {
    let stored: Option<i64> = conn.value("SELECT id FROM text WHERE hash=?1", [hash.as_slice()])?;
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
}

#[derive(Debug)]
struct SearchEntry {
    text_id: i64,
    fts_table: i64,
    /// What the entry becomes: `None` when the text is no longer a member
    /// of the index (or no longer exists), so its entry leaves.
    text: Option<EntryText>,
}

#[derive(Debug)]
struct EntryText {
    /// The text's content hash when it was read: an entry applies only to
    /// the same content.
    hash: Vec<u8>,
    folded: String,
}

impl SearchEntries {
    /// Nothing to apply: the queue was empty when it was read.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

row! {
    /// A queued change to a text's full-text entry.
    struct PendingEntry {
        text_id: i64,
        fts_table: i64,
    }
    statements {
        const NEXT = "SELECT text_id, fts_table FROM search_pending ORDER BY text_id, fts_table LIMIT ?1";
    }
}

/// Reads the next queued changes to search entries and the folded text of
/// those that enter, up to `byte_limit` of text, in one read transaction.
/// Writes nothing.
pub(crate) fn pending_entries(conn: &Connection, byte_limit: usize) -> Result<SearchEntries> {
    let tx = conn.unchecked_transaction()?;
    // Read the ordered queue first; a join can instead scan all historical
    // texts and sort on every batch. Finish the cursor before reading
    // bodies, which are fetched individually, so the row limit does not
    // multiply text held.
    let pending = tx.all(&PendingEntry::NEXT, [SEARCH_ROWS])?;
    let mut entries = Vec::new();
    let mut bytes = 0;
    for PendingEntry { text_id, fts_table } in pending {
        let member: Option<Vec<u8>> = tx.value(
            "SELECT hash FROM text WHERE id = ?1 AND indexed & ?2 <> 0",
            [text_id, fts_table],
        )?;
        let text = match member {
            Some(hash) => {
                let (folded, length) = folded_text(read(&tx, text_id)?);
                bytes += length;
                Some(EntryText { hash, folded })
            }
            None => None,
        };
        entries.push(SearchEntry {
            text_id,
            fts_table,
            text,
        });
        if bytes >= byte_limit {
            break;
        }
    }
    tx.finish()?;
    Ok(SearchEntries { entries })
}

/// Applies computed changes to search entries, with the number of queue rows
/// settled. A change whose queue row is gone, or whose text has changed
/// since it was read, was overtaken by a write after the read and is left
/// for the next read. The full-text files commit first and the queue rows
/// leave after, so a crash between the two leaves the rows, and applying
/// them again writes the same entries.
///
/// A build's writer has no full-text files attached
/// ([`crate::search::files`]); its queue waits for the installed index.
pub(crate) fn apply_entries(conn: &Connection, computed: &SearchEntries) -> Result<u64> {
    if computed.is_empty() || !crate::search::files::attached(conn)? {
        return Ok(0);
    }
    // Immediate: a deferred reader cannot wait to upgrade its writer lock.
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    let mut applied = Vec::new();
    for entry in &computed.entries {
        let current = match &entry.text {
            Some(text) => tx.value(
                "SELECT EXISTS(SELECT 1 FROM search_pending WHERE text_id = ?1 AND fts_table = ?2)
                 AND EXISTS(SELECT 1 FROM text WHERE id = ?1 AND indexed & ?2 <> 0 AND hash = ?3)",
                params![entry.text_id, entry.fts_table, text.hash],
            )?,
            None => tx.value(
                "SELECT EXISTS(SELECT 1 FROM search_pending WHERE text_id = ?1 AND fts_table = ?2)
                 AND NOT EXISTS(SELECT 1 FROM text WHERE id = ?1 AND indexed & ?2 <> 0)",
                params![entry.text_id, entry.fts_table],
            )?,
        };
        if current == Some(true) {
            set_entry(
                &tx,
                entry.text_id,
                entry.fts_table,
                entry.text.as_ref().map(|text| text.folded.as_str()),
            )?;
            applied.push((entry.text_id, entry.fts_table));
        }
    }
    tx.commit()?;
    if applied.is_empty() {
        return Ok(0);
    }
    crate::fault::check("search.before_queue", conn.path().unwrap_or_default())?;
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    for (text_id, fts_table) in &applied {
        tx.prepare_cached("DELETE FROM search_pending WHERE text_id = ?1 AND fts_table = ?2")?
            .execute(params![text_id, fts_table])?;
    }
    crate::archive::bump_counter(&tx)?;
    tx.commit()?;
    Ok(applied.len() as u64)
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

/// Makes text `id`'s entry in the index `bit` names hold `folded`, or
/// removes it, in the file that holds the text's entries. Writing the same
/// entry twice leaves one.
fn set_entry(conn: &Connection, id: i64, bit: i64, folded: Option<&str>) -> Result<()> {
    let file = crate::search::files::schema(crate::search::files::file_of(id));
    let table = if bit == 1 { "text_fts" } else { "output_fts" };
    write_entry(conn, &file, table, id, folded)
}

/// [`set_entry`] in the file attached as `file` (`main` for a file opened
/// alone), keeping the file's totals ([`crate::search::files`]): an entry
/// removed takes its tokens, read from FTS5's record of each entry's length,
/// out of them, and one written adds its own.
pub(crate) fn write_entry(
    conn: &Connection,
    file: &str,
    table: &str,
    id: i64,
    folded: Option<&str>,
) -> Result<()> {
    let length = |conn: &Connection| -> Result<Option<i64>> {
        let size: Option<Vec<u8>> = conn.value(
            &format!("SELECT sz FROM {file}.{table}_docsize WHERE id = ?1"),
            [id],
        )?;
        size.map(|size| {
            entry_length(&size)
                .ok_or_else(|| StoreError::stored(0, "a full-text entry has no length"))
        })
        .transpose()
    };
    let total = |conn: &Connection, entries: i64, tokens: i64| -> Result<()> {
        conn.prepare_cached(&format!(
            "UPDATE {file}.totals SET entries = entries + ?2, tokens = tokens + ?3 WHERE name = ?1"
        ))?
        .execute(params![table, entries, tokens])?;
        Ok(())
    };
    if let Some(tokens) = length(conn)? {
        conn.prepare_cached(&format!("DELETE FROM {file}.{table} WHERE rowid = ?1"))?
            .execute([id])?;
        total(conn, -1, -tokens)?;
    }
    if let Some(folded) = folded {
        conn.prepare_cached(&format!(
            "INSERT INTO {file}.{table}(rowid, body) VALUES (?1, ?2)"
        ))?
        .execute(params![id, folded])?;
        let tokens = length(conn)?
            .ok_or_else(|| StoreError::internal("a written full-text entry has no length"))?;
        total(conn, 1, tokens)?;
    }
    Ok(())
}

/// An entry's token count from its FTS5 size record: one SQLite varint for
/// the one column.
fn entry_length(size: &[u8]) -> Option<i64> {
    let mut value: u64 = 0;
    for (n, &byte) in size.iter().enumerate() {
        if n == 8 {
            return Some(((value << 8) | u64::from(byte)) as i64);
        }
        value = (value << 7) | u64::from(byte & 0x7f);
        if byte & 0x80 == 0 {
            return Some(value as i64);
        }
    }
    None
}

/// Removes a text row and queues the removal of its search entries; its
/// group leaves once nothing names it.
pub(crate) fn remove(i: &Connection, id: i64) -> Result<()> {
    i.execute(
        "INSERT OR IGNORE INTO search_pending(text_id, fts_table)
         SELECT ?1, bit FROM (SELECT 1 AS bit UNION ALL SELECT 2)
         WHERE (SELECT indexed FROM text WHERE id = ?1) & bit <> 0",
        [id],
    )?;
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
        "INSERT INTO content_group(hash,body) VALUES (?1,?2) ON CONFLICT(hash) DO NOTHING",
        params![hash.as_bytes().as_slice(), body],
    )
    .unwrap();
    crate::ConnExt::scalar(
        conn,
        "SELECT id FROM content_group WHERE hash=?1",
        [hash.as_bytes().as_slice()],
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
            "UPDATE content_group SET body=?2 WHERE id=(SELECT group_id FROM text WHERE id=?1)",
            params![id, body],
        )
        .unwrap();
    assert_eq!(changed, 1);
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<PendingEntry>();
    groups::declared_rows(check);
    search_fill::declared_rows(check);
}

#[cfg(test)]
mod tests;
