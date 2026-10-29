//! Remove a session's derived rows.
use crate::Result;
use rusqlite::{Transaction, params};

/// Removes a session's derived rows within its reading-replacement transaction.
pub(crate) fn remove(i: &Transaction<'_>, pk: i64, id: &str) -> Result<()> {
    i.execute(
        "DELETE FROM message_source WHERE message_pk IN (SELECT pk FROM message WHERE session_pk=?1)",
        [pk],
    )?;
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
    crate::versions::remove_readings(i)?;
    Ok(())
}

#[cfg(test)]
mod tests;
