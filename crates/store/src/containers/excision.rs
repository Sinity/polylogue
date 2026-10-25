//! An export ZIP holds many unrelated conversations, and its compressed bytes
//! cannot be rewritten in place, so excision never reaches into one.
use super::*;
use rusqlite::OptionalExtension;
use std::collections::BTreeSet;

/// Refuses when any of `inputs` is an export ZIP or a file read from one:
/// excising it would also remove the unrelated conversations the ZIP holds.
pub(crate) fn refuse_inside_zip(conn: &Connection, inputs: &BTreeSet<String>) -> Result<()> {
    let mut read_from = conn.prepare_cached(
        "SELECT c.container_input FROM observation_container c
         JOIN observation o ON o.id = c.observation_id WHERE o.input = ?1 LIMIT 1",
    )?;
    for input in inputs {
        let zip = if is_zip(conn, input)? {
            Some(input.clone())
        } else {
            read_from
                .query_row([input], |r| r.get::<_, String>(0))
                .optional()?
        };
        if let Some(zip) = zip {
            return Err(inside_zip(conn, zip)?);
        }
    }
    Ok(())
}

/// The refusal for an excision inside the ZIP stored as `input`, naming the
/// file it was imported from.
fn inside_zip(conn: &Connection, input: String) -> Result<StoreError> {
    let file: Option<String> = conn
        .query_row(
            "SELECT path FROM observation WHERE input = ?1 ORDER BY rowid LIMIT 1",
            [&input],
            |r| r.get(0),
        )
        .optional()?;
    Ok(StoreError::ExcisionInsideZip {
        file: file.unwrap_or_else(|| input.clone()),
        input,
    })
}

/// Whether `input` is an export ZIP.
fn is_zip(conn: &Connection, input: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM observation_container WHERE container_input=?1)",
        [input],
        |r| r.get(0),
    )?)
}
