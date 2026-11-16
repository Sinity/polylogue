//! Stored text identity and its deferred full-text entries. Callers group row changes
//! in their own transaction; indexing consumes committed queue entries.

use crate::ConnExt as _;
use crate::OptionExt as _;
use crate::Result;
mod groups;
mod identities;
#[cfg(any(test, feature = "fixture"))]
use crate::first_build::FirstBuild;
pub(crate) use groups::{PreparedTexts, StoredGroups, column, joined, read, slice};
pub(crate) use identities::TextIds;
#[cfg(any(test, feature = "fixture"))]
use rusqlite::OptionalExtension;
use rusqlite::{Connection, TransactionBehavior, params};

/// The text row for `body`, stored once per distinct content, and its
/// entry in the search index `bit` names (1 prose, 2 tool output). During a
/// first build the text is found through `first_build`, not its hash index.
#[cfg(any(test, feature = "fixture"))]
pub(crate) fn intern(
    conn: &Connection,
    body: &str,
    bit: i64,
    prepared: Option<&PreparedTexts>,
    first_build: Option<&FirstBuild>,
) -> Result<i64> {
    let hash = blake3::hash(body.as_bytes());
    let id = conn
        .prepare_cached("SELECT id FROM text WHERE hash=?1")?
        .query_row([hash.as_bytes().as_slice()], |r| r.get::<_, i64>(0))
        .optional()?;
    let bulk = first_build.is_some();
    let (id, indexed) = match id {
        Some(id) => (
            id,
            conn.prepare_cached("SELECT indexed FROM text WHERE id = ?1")?
                .query_row([id], |r| r.get(0))?,
        ),
        None => {
            let own;
            let prepared = match prepared.filter(|p| p.contains(hash.as_bytes())) {
                Some(prepared) => prepared,
                None => {
                    own = PreparedTexts::pack([body])?;
                    &own
                }
            };
            let prepared = prepared
                .get(hash.as_bytes())
                .expect("the text was just prepared");
            // A first build marks the text for its search index now and
            // writes the entry when the build ends.
            let marked = if bulk { bit } else { 0 };
            let id = groups::insert(
                conn,
                &StoredGroups::default(),
                None,
                hash.as_bytes(),
                prepared,
                marked,
            )?
            .or_invalid("the text row already exists")?;
            (id, marked)
        }
    };
    // `indexed` marks a text for its search index; the entry itself is
    // written after this transaction commits: by the search thread, or by
    // a first build in bulk when it ends.
    if bit != 0 && indexed & bit == 0 {
        join_index(conn, id, bit, bulk)?;
    }
    Ok(id)
}

/// The search indexes new text rows join when they are written: those the
/// blocks a step makes current name, by text hash (bit 1 prose, 2 tool
/// output). A first build (`bulk`) writes the entries when it ends; otherwise
/// they are queued for the search thread.
pub(crate) struct NewMembership<'a> {
    pub bits: &'a std::collections::HashMap<[u8; 32], i64>,
    pub bulk: bool,
}

/// Marks an already stored text for current-block search membership.
pub(crate) fn activate(
    conn: &Connection,
    texts: &TextIds,
    id: i64,
    bit: i64,
    bulk: bool,
) -> Result<()> {
    // A text marked in this transaction needs no statement per block.
    if texts.is_marked(id, bit) {
        return Ok(());
    }
    join_index(conn, id, bit, bulk)?;
    texts.mark(id, bit);
    Ok(())
}

fn join_index(conn: &Connection, id: i64, bit: i64, bulk: bool) -> Result<()> {
    let changed = conn
        .prepare_cached(
            "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1 AND indexed & ?2 = 0",
        )?
        .execute(params![id, bit])?;
    if changed != 0 && !bulk {
        queue_entry(conn, id, bit)?;
    }
    Ok(())
}

fn queue_entry(conn: &Connection, id: i64, bit: i64) -> Result<()> {
    conn.prepare_cached(
        "INSERT OR IGNORE INTO search_pending(text_id, fts_table) VALUES (?1, ?2)",
    )?
    .execute(params![id, bit])?;
    Ok(())
}

// Limit the search thread's transaction work independently of compressed
// row sizes. A single oversized text is still decoded whole and ends its
// transaction.
const SEARCH_ROWS: i64 = 2_000;
const SEARCH_BYTES: usize = 16 * 1024 * 1024;

/// Writes the first build's marked texts in ID order, both search indexes
/// from one scan of `text`, in one transaction. The full-text index then
/// writes each segment once: committing every few megabytes of text flushed
/// a small segment per commit and merged them again, which wrote twice what
/// the index keeps without a journal and four times with one (2026-11-15
/// measurement). Prior postings and statistics are replaced in the same
/// transaction; inserting the same contentless row IDs again would count
/// words twice. A failure or a stop leaves the search indexes as they were.
pub(crate) fn index_marked(conn: &Connection) -> Result<()> {
    let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
    tx.execute("INSERT INTO text_fts(text_fts) VALUES ('delete-all')", [])?;
    tx.execute(
        "INSERT INTO output_fts(output_fts) VALUES ('delete-all')",
        [],
    )?;
    let mut written = false;
    {
        let mut select = tx.prepare(
            "SELECT t.id, t.indexed, g.body, t.start, t.length FROM text t
             JOIN text_group g ON g.id = t.group_id WHERE t.indexed <> 0 ORDER BY t.id",
        )?;
        let mut rows = select.query([])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let indexed: i64 = row.get(1)?;
            let (folded, _) = folded_text(joined(row, 2)?)?;
            for bit in [1, 2].into_iter().filter(|bit| indexed & bit != 0) {
                insert_entry(&tx, id, bit, &folded)?;
            }
            written = true;
            #[cfg(feature = "faults")]
            crate::fault::check("search.indexed", conn.path().unwrap_or_default())?;
        }
    }
    if written {
        crate::archive::bump_counter(&tx)?;
    }
    tx.commit()?;
    Ok(())
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
        .query_map([SEARCH_ROWS], |r| <(_, _)>::try_from(r))?
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
        bytes += write_entry(&tx, id, bit, read(&tx, id)?)?;
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

fn write_entry(conn: &Connection, id: i64, bit: i64, text: String) -> Result<usize> {
    let (folded, length) = folded_text(text)?;
    insert_entry(conn, id, bit, &folded)?;
    Ok(length)
}

/// A stored text as the search indexes hold it, and its length in bytes.
fn folded_text(text: String) -> Result<(String, usize)> {
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
pub(crate) fn unindex_unused(i: &Connection, texts: &TextIds, id: i64, bulk: bool) -> Result<()> {
    for (bit, kind, table) in [
        (1, "b.kind NOT IN ('tool_result','tool_call')", "text_fts"),
        (2, "b.kind IN ('tool_result','tool_call')", "output_fts"),
    ] {
        let used:bool=i.exists(&format!("SELECT EXISTS(SELECT 1 FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.text_id=?1 AND {kind}) OR (?2=1 AND EXISTS(SELECT 1 FROM document_block b JOIN document d ON d.pk=b.document_pk WHERE b.text_id=?1))"), params![id,bit])?;
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
            texts.unmark(id, bit);
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
    groups::delete(i, id)
}

/// Removes a text row during a first build, whose search entries are written
/// only when it ends.
pub(crate) fn remove_unindexed(i: &Connection, id: i64) -> Result<()> {
    groups::delete(i, id)
}

/// A group body holding only `text`.
#[cfg(test)]
pub(crate) fn alone(text: &str) -> Vec<u8> {
    zstd::bulk::compress(text.as_bytes(), 1).unwrap()
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
    conn.execute("INSERT INTO text_group(body) VALUES (?1)", [alone(text)])
        .unwrap();
    let group = conn.last_insert_rowid();
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
    conn.execute("INSERT INTO text_group(body) VALUES (?1)", [alone(text)])
        .unwrap();
    let changed = conn
        .execute(
            "UPDATE text SET group_id=?2, start=0, length=?3 WHERE id=?1",
            params![id, conn.last_insert_rowid(), text.len() as i64],
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
