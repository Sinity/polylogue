//! Publication retains both the selected flight and the receiving admission.
use polylogue_store::{ReadScopeToken, Result};

#[derive(Clone, Default)]
pub(crate) struct ReadScopes {
    waiter: Option<ReadScopeToken>,
    owner: Option<ReadScopeToken>,
}
impl ReadScopes {
    pub(crate) fn single(waiter: Option<ReadScopeToken>) -> Self {
        Self {
            waiter,
            owner: None,
        }
    }
    pub(crate) fn dispatched(
        result: &command_surface::DispatchResult,
        waiter: Option<ReadScopeToken>,
    ) -> Self {
        let owner = result
            .owner
            .identity::<super::caller::Accepted>()
            .and_then(|accepted| accepted.read_scope.clone());
        Self { waiter, owner }
    }
    pub(crate) fn check(&self) -> Result<()> {
        if let Some(owner) = &self.owner {
            owner.check()?;
        }
        if let Some(waiter) = &self.waiter {
            waiter.check()?;
        }
        Ok(())
    }
    pub(crate) fn has_scope(&self) -> bool {
        self.owner.is_some() || self.waiter.is_some()
    }
    pub(crate) fn with_current<T>(&self, publish: impl FnOnce() -> Result<T>) -> Result<T> {
        // Both tokens were captured from this operation table's one daemon
        // archive. One native fence protects their epoch checks and publication;
        // nesting the same fence for owner and waiter would deadlock.
        match self.owner.as_ref().or(self.waiter.as_ref()) {
            Some(scope) => scope.with_current(|| {
                self.check()?;
                publish()
            }),
            None => publish(),
        }
    }
}

pub(crate) fn answer<T>(
    answer: command_surface::Answer<T>,
    scopes: &ReadScopes,
) -> command_surface::Answer<T> {
    match scopes.check() {
        Ok(()) => answer,
        Err(error) => command_surface::Answer::error(super::failure(
            super::store_code(&error),
            error.to_string(),
            None,
        )),
    }
}

#[cfg(test)]
mod tests;
