//! Numbered evolution of the kept body's executable metadata.
use crate::{Result, StoreError};
use rusqlite::{Connection, params};

pub(crate) fn evolve_body(id: &str, body: &str) -> Result<String> {
    polylogue_model::kept::evolve_body(body).map_err(|error| {
        StoreError::Invalid(format!("kept result {id:?} field rows_json: {error}"))
    })
}

/// Archive version 2 -> 3. Original client requests and effects retain their bytes.
pub(crate) fn archive_v3(conn: &Connection) -> Result<()> {
    evolve(conn, crate::source::Source::Ledger)
}

pub(crate) fn index_v3(conn: &Connection, primary: bool) -> Result<()> {
    if primary {
        evolve(conn, crate::source::Source::Cache)
    } else {
        Ok(())
    }
}

fn evolve(conn: &Connection, source: crate::source::Source) -> Result<()> {
    let mut statement =
        conn.prepare(&source.sql("SELECT id,rows_json FROM @source.kept_result ORDER BY id"))?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let original: String = row.get(1)?;
        let body = evolve_body(&id, &original)?;
        crate::selection::check_kept_body(conn, &id, &body)?;
        if body != original {
            conn.execute(
                &source.sql("UPDATE @source.kept_result SET rows_json=?1 WHERE id=?2"),
                params![body, id],
            )?;
        }
    }
    Ok(())
}
