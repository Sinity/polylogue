//! Publication retains the native scope captured for this request.
use polylogue_store::{ReadScopeToken, Result};

#[derive(Clone, Default)]
pub(crate) struct ReadScopes {
    scope: Option<ReadScopeToken>,
}
impl ReadScopes {
    pub(crate) fn single(scope: Option<ReadScopeToken>) -> Self {
        Self { scope }
    }
    pub(crate) fn check(&self) -> Result<()> {
        if let Some(scope) = &self.scope {
            scope.check()?;
        }
        Ok(())
    }
    pub(crate) fn has_scope(&self) -> bool {
        self.scope.is_some()
    }
    pub(crate) fn with_current<T>(&self, publish: impl FnOnce() -> Result<T>) -> Result<T> {
        match self.scope.as_ref() {
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
        Err(error) => command_surface::Answer::error(super::store_failure(&error)),
    }
}

#[cfg(test)]
mod tests;
