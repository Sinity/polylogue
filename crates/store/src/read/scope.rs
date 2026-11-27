//! Access-scope changes invalidate in-flight reads before the cache changes.
use crate::ResultExt as _;
use crate::{Result, StoreError};
use polylogue_model::LockExt;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// Wakes a reader waiting for a pool connection, so a scope change ends
/// its wait.
pub(crate) type ResetWake = dyn Fn() + Send + Sync;

#[derive(Default)]
pub(crate) struct ReadScopeState {
    epoch: AtomicU64,
    resetting: AtomicBool,
    wakes: Mutex<Vec<Weak<ResetWake>>>,
}

impl std::fmt::Debug for ReadScopeState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadScopeState")
            .field("epoch", &self.epoch)
            .field("resetting", &self.resetting)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct ReadScopeToken {
    state: Arc<ReadScopeState>,
    epoch: u64,
}

pub(crate) fn state(root: &Path) -> Arc<ReadScopeState> {
    static ARCHIVES: OnceLock<Mutex<HashMap<PathBuf, Weak<ReadScopeState>>>> = OnceLock::new();
    let key = root_key(root);
    let mut archives = ARCHIVES.get_or_init(Default::default).locked();
    archives.retain(|_, state| state.strong_count() > 0);
    if let Some(state) = archives.get(&key).and_then(Weak::upgrade) {
        return state;
    }
    let state = Arc::new(ReadScopeState::default());
    archives.insert(key, Arc::downgrade(&state));
    state
}

pub(crate) fn root_key(root: &Path) -> PathBuf {
    if let Ok(canonical) = root.canonicalize() {
        return canonical;
    }
    match (root.parent(), root.file_name()) {
        (Some(parent), Some(name)) => {
            let parent = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            root_key(parent).join(name)
        }
        _ => root.to_path_buf(),
    }
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
    /// Calls `wake` at the start of every reset for as long as the caller
    /// keeps it.
    pub(crate) fn on_reset(&self, wake: &Arc<ResetWake>) {
        let mut wakes = self.wakes.locked();
        wakes.retain(|wake| wake.strong_count() > 0);
        wakes.push(Arc::downgrade(wake));
    }

    /// Ends every token taken before it; tokens taken until
    /// [`Self::finish_reset`] fail their check too.
    pub(crate) fn begin_reset(&self) -> Result<()> {
        self.resetting.store(true, Ordering::Release);
        let wakes: Vec<_> = self
            .wakes
            .locked()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for wake in wakes {
            wake();
        }
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .or_internal("read scope epoch exhausted")?;
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
