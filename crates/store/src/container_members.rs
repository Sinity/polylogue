//! Checks a container membership before it is stored: the container is an original input, the member index fits, and the kind is named.
use crate::ConnExt as _;
use crate::{Batch, InContainer, Result, StoreError};
use rusqlite::Connection;
pub(crate) fn container_of(batch: &Batch) -> Option<&InContainer> {
    batch.observation.as_ref()?.container.as_ref()
}
pub(crate) fn require_original(conn: &Connection, input: &str) -> Result<()> {
    let source = crate::source::Source::of(conn)?;
    let stored: bool = conn.scalar(
        &source.sql("SELECT EXISTS(SELECT 1 FROM @source.revision WHERE input=?1)"),
        [input],
    )?;
    if !stored {
        return Err(StoreError::Internal(
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
    check_shape(input, container)
}

/// The membership's own shape, for a container staged in the same group and
/// so not yet in the archive.
pub(crate) fn check_shape(input: &str, container: &InContainer) -> Result<InContainer> {
    if input == container.input {
        return Err(StoreError::Internal(
            "a container cannot be its own member".into(),
        ));
    }
    i64::try_from(container.member_index).map_err(|_| {
        StoreError::Internal("container member index exceeds archive address space".into())
    })?;
    if container.kind.is_empty() {
        return Err(StoreError::Internal("container kind is empty".into()));
    }
    Ok(container.clone())
}
#[cfg(test)]
mod tests;
