//! Validate recorded request encodings before replacing their digests.
use crate::{Result, StoreError};
use polylogue_model::user::{UserRequest, UserWrite};
use rusqlite::{Transaction, params};

// The predecessor sorted assertion values, but serialized the surrounding
// request structs in declaration order. This recipe belongs only to the
// transition; current reads and writes use the canonical request hash.
fn prior_hash(request: &UserRequest) -> Result<[u8; 32]> {
    let mut intent = request.write.intent();
    match &mut intent {
        UserWrite::Assert { assertions } => {
            for assertion in assertions {
                if let Some(value) = &mut assertion.value {
                    value.sort_all_objects();
                }
            }
        }
        UserWrite::AssertQuery { assertion, .. } => {
            if let Some(value) = &mut assertion.value {
                value.sort_all_objects();
            }
        }
        UserWrite::ReviseAssertions { patch, .. } => {
            if let polylogue_model::assertion::Change::Set(value) = &mut patch.value {
                value.sort_all_objects();
            }
        }
        _ => {}
    }
    Ok(*blake3::hash(&serde_json::to_vec(&(
        &request.machine,
        &request.authorship,
        intent,
    ))?)
    .as_bytes())
}

pub(super) fn apply(tx: &Transaction<'_>, primary: bool) -> Result<()> {
    let source = if primary {
        crate::source::Source::Cache
    } else {
        crate::source::Source::Ledger
    };
    let mut statement = tx.prepare(
        &source.sql("SELECT id,content_hash,request FROM @source.client_request ORDER BY id"),
    )?;
    let mut rows = statement.query([])?;
    let file: String = tx.query_row(
        "SELECT file FROM pragma_database_list WHERE name='main'",
        [],
        |row| row.get(0),
    )?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let packed: Vec<u8> = row.get(2)?;
        let request: UserRequest = crate::encoding::unpack_json(&packed)?;
        if request.id != id {
            return Err(StoreError::Conflict(id));
        }
        let canonical = crate::primary::authored::request_hash(&request)?;
        if hash.as_slice() == canonical {
            continue;
        }
        if hash.as_slice() != prior_hash(&request)? {
            return Err(StoreError::Conflict(id));
        }
        tx.execute(
            &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
            params![&canonical[..], id],
        )?;
        crate::fault::check("archive.request_hash_transition", &file)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
