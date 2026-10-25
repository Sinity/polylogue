//! ZIP imports: the original ZIP is one stored input, and each member read
//! from it is its own input whose import observation names the ZIP it came
//! from (`observation_zip`).
use crate::{Batch, InZip, Result, StoreError};
use rusqlite::Connection;

mod excision;
pub(crate) use excision::refuse_inside_zip;

/// The ZIP a batch's member was read from.
pub(crate) fn zip_of(batch: &Batch) -> Option<&InZip> {
    batch.observation.as_ref()?.import.as_ref()?.zip.as_ref()
}

/// Refuses unless `input` is a stored import named by its bytes.
pub(crate) fn require_original(conn: &Connection, input: &str) -> Result<()> {
    let stored: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM revision WHERE input = ?1 AND produced = 'imported')",
        [input],
        |r| r.get(0),
    )?;
    if !stored {
        return Err(StoreError::Invalid(
            "a ZIP member names a ZIP that is not stored".into(),
        ));
    }
    Ok(())
}

/// The ZIP a member observation of `input` was read from, and its index
/// there; the ZIP must be stored already, and is never its own member.
pub(crate) fn check(conn: &Connection, input: &str, zip: &crate::InZip) -> Result<(String, i64)> {
    require_original(conn, &zip.input)?;
    if input == zip.input {
        return Err(StoreError::Invalid("a ZIP cannot be its own member".into()));
    }
    let index = i64::try_from(zip.member_index).map_err(|_| {
        StoreError::Invalid("ZIP member index exceeds archive address space".into())
    })?;
    Ok((zip.input.clone(), index))
}

#[cfg(test)]
mod tests;
