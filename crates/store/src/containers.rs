//! ZIP imports: the original ZIP is one stored input, and each member read
//! from it is its own input whose import observation names the ZIP it came
//! from (`observation_container`).
use crate::{Batch, ImportContainer, Result, StoreError};
use rusqlite::Connection;

mod excision;
pub(crate) use excision::refuse_inside_zip;

/// The ZIP a batch's member was read from.
pub(crate) fn coordinate(batch: &Batch) -> Option<&ImportContainer> {
    batch
        .observation
        .as_ref()?
        .import
        .as_ref()?
        .container
        .as_ref()
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

/// Records which ZIP the member observed by `observation` was read from.
/// The ZIP a member observation of `input` was read from, and its index
/// there; the ZIP must be stored already, and is never its own member.
pub(crate) fn check(
    conn: &Connection,
    input: &str,
    container: &crate::ImportContainer,
) -> Result<(String, i64)> {
    require_original(conn, &container.input)?;
    if input == container.input {
        return Err(StoreError::Invalid("a ZIP cannot be its own member".into()));
    }
    let index = i64::try_from(container.member_index).map_err(|_| {
        StoreError::Invalid("ZIP member index exceeds archive address space".into())
    })?;
    Ok((container.input.clone(), index))
}

#[cfg(test)]
mod tests;
