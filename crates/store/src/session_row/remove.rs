//! Remove derived session rows, promoting surviving copies before orphan cleanup.
use crate::Result;
use rusqlite::{Connection, Transaction, params};

mod references;
use references::References;

/// Remove derived rows without collecting references the caller does not need.
pub(crate) fn remove(i: &Transaction<'_>, pk: i64, id: &str, counter: i64) -> Result<()> {
    remove_inner(i, pk, id, counter, None)
}

/// Visit frozen references after deletion, copy promotion and version pruning.
/// Staged references belong to this operation and transaction; errors require
/// its rollback, just as errors from the row deletion itself do.
pub(crate) fn remove_with_references(
    i: &Transaction<'_>,
    pk: i64,
    id: &str,
    counter: i64,
    mut consume: impl FnMut(&References<'_>) -> Result<()>,
) -> Result<()> {
    remove_inner(i, pk, id, counter, Some(&mut consume))
}

type Consumer<'a> = &'a mut dyn FnMut(&References<'_>) -> Result<()>;

fn remove_inner(
    i: &Transaction<'_>,
    pk: i64,
    id: &str,
    counter: i64,
    consume: Option<Consumer<'_>>,
) -> Result<()> {
    i.execute("DELETE FROM snapshot WHERE pk IN (SELECT snapshot_pk FROM snapshot_session WHERE session_pk=?1)", [pk])?;
    let references = consume
        .as_ref()
        .map(|_| References::capture(i, pk))
        .transpose()?;
    // Callers may hold other TEMP cursors. Keep table definitions for the
    // connection lifetime: DROP would lock their shared TEMP schema. Rows are
    // scoped to this removal and cleared below; errors roll them back.
    i.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS session_remove_copy(
            ordinal INTEGER PRIMARY KEY, message_pk INTEGER NOT NULL UNIQUE
         );",
    )?;
    i.execute(
        "INSERT INTO temp.session_remove_copy(message_pk)
         SELECT DISTINCT s.message_pk FROM message_source s JOIN message m ON m.pk=s.message_pk
         WHERE s.session_pk=?1 AND m.session_pk<>?1",
        [pk],
    )?;
    i.execute("DELETE FROM message_source WHERE session_pk=?1 OR message_pk IN (SELECT pk FROM message WHERE session_pk=?1)",[pk])?;
    // Search entries go before their tool-call rows. Only one command/path pair
    // is decoded at a time; the tool-call table is unchanged during this cursor.
    let mut calls = i.prepare("SELECT pk,command,path FROM tool_call WHERE session_pk=?1")?;
    let mut rows = calls.query([pk])?;
    while let Some(row) = rows.next()? {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram,rowid,command,path)
             VALUES ('delete',?1,?2,?3)",
            params![
                row.get::<_, i64>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?
            ],
        )?;
    }
    drop(rows);
    drop(calls);
    // Every table that names the session, its messages or its tool calls,
    // whichever component added it.
    let tables = crate::schema::index_columns(i)?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("tool_call_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE tool_call_pk IN
                     (SELECT pk FROM tool_call WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("message_pk") && !cols.contains("session_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE message_pk IN
                     (SELECT pk FROM message WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    i.execute("DELETE FROM tool_call WHERE session_pk = ?1", [pk])?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" {
            continue;
        }
        if cols.contains("session_pk") {
            i.execute(&format!("DELETE FROM {table} WHERE session_pk = ?1"), [pk])?;
        }
    }
    i.execute(
        "DELETE FROM link WHERE parent_id = ?1 OR child_id = ?1",
        [id],
    )?;
    i.execute("DELETE FROM message WHERE session_pk = ?1", [pk])?;
    i.execute("DELETE FROM session WHERE pk = ?1", [pk])?;
    let mut copies =
        i.prepare("SELECT message_pk FROM temp.session_remove_copy ORDER BY ordinal")?;
    crate::writer::promote_current(i, copies.query_map([], |r| r.get(0))?, counter)?;
    drop(copies);
    crate::versions::prune(i)?;
    if let (Some(consume), Some(references)) = (consume, references) {
        consume(&references)?;
        references.finish()?;
    }
    i.execute_batch("DELETE FROM temp.session_remove_copy")?;
    Ok(())
}

#[cfg(test)]
mod tests;
