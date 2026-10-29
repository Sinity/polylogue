//! Read-derived actions retain their admission generation through the queue.
use crate::{Archive, ReadScopeToken, Result, StoreError};

pub(super) fn require_current(archive: &Archive, selected: Option<&ReadScopeToken>) -> Result<()> {
    if let Some(selected) = selected {
        let current = archive.scope_token()?;
        if !selected.same(&current) {
            return Err(StoreError::ScopeChanged);
        }
        selected.check()?;
    }
    Ok(())
}
