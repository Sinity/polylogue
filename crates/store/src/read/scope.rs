//! Access-scope changes invalidate in-flight reads before the cache changes.
use crate::{Result, StoreError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[derive(Debug, Default)]
pub(crate) struct ReadScopeState {
    epoch: AtomicU64,
    resetting: AtomicBool,
}
#[derive(Debug, Clone)]
pub struct ReadScopeToken {
    state: Arc<ReadScopeState>,
    epoch: u64,
}

pub(crate) fn state(root: &Path) -> Arc<ReadScopeState> {
    static ARCHIVES: OnceLock<Mutex<HashMap<PathBuf, Weak<ReadScopeState>>>> = OnceLock::new();
    let key = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let mut archives = ARCHIVES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    archives.retain(|_, state| state.strong_count() > 0);
    if let Some(state) = archives.get(&key).and_then(Weak::upgrade) {
        return state;
    }
    let state = Arc::new(ReadScopeState::default());
    archives.insert(key, Arc::downgrade(&state));
    state
}
impl ReadScopeState {
    pub(crate) fn token(self: &Arc<Self>) -> Result<ReadScopeToken> {
        let token = ReadScopeToken {
            state: Arc::clone(self),
            epoch: self.epoch.load(Ordering::Acquire),
        };
        token.check()?;
        Ok(token)
    }
    pub(crate) fn begin_reset(&self) -> Result<()> {
        self.resetting.store(true, Ordering::Release);
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map_err(|_| StoreError::Invalid("read scope epoch exhausted".into()))?;
        Ok(())
    }
    pub(crate) fn finish_reset(&self) {
        self.resetting.store(false, Ordering::Release);
    }
}
impl ReadScopeToken {
    pub fn check(&self) -> Result<()> {
        if self.state.resetting.load(Ordering::Acquire)
            || self.state.epoch.load(Ordering::Acquire) != self.epoch
        {
            Err(StoreError::ScopeChanged)
        } else {
            Ok(())
        }
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        self.epoch == other.epoch && Arc::ptr_eq(&self.state, &other.state)
    }
}

#[cfg(test)]
mod tests;
