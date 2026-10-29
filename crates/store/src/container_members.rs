//! Exact membership in retained container bytes.
use crate::{Batch, InContainer, Result, StoreError};
use rusqlite::Connection;
pub(crate) fn container_of(batch: &Batch) -> Option<&InContainer> {
    batch.observation.as_ref()?.container.as_ref()
}
pub(crate) fn require_original(conn: &Connection, input: &str) -> Result<()> {
    let source = crate::source::Source::of(conn)?;
    let stored: bool = conn.query_row(
        &source.sql("SELECT EXISTS(SELECT 1 FROM @source.revision WHERE input=?1)"),
        [input],
        |r| r.get(0),
    )?;
    if !stored {
        return Err(StoreError::Invalid(
            "a container member names unstored original bytes".into(),
        ));
    }
    Ok(())
}
pub(crate) fn check(
    conn: &Connection,
    input: &str,
    container: &InContainer,
) -> Result<InContainer> {
    require_original(conn, &container.input)?;
    if input == container.input {
        return Err(StoreError::Invalid(
            "a container cannot be its own member".into(),
        ));
    }
    i64::try_from(container.member_index).map_err(|_| {
        StoreError::Invalid("container member index exceeds archive address space".into())
    })?;
    if container.kind.is_empty() {
        return Err(StoreError::Invalid("container kind is empty".into()));
    }
    Ok(container.clone())
}
#[cfg(test)]
mod tests;
