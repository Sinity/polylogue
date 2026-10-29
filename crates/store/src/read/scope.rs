//! Access-scope changes invalidate in-flight reads before the cache changes.
use crate::{Result, StoreError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

type Invalidator = dyn Fn() -> Result<()> + Send + Sync;

#[derive(Default)]
pub(crate) struct ReadScopeState {
    epoch: AtomicU64,
    resetting: AtomicBool,
    fence: Mutex<Vec<Weak<Invalidator>>>,
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

/// Retain this registration for the daemon lifetime while a derived cache pointer
/// can be read, including times when its publisher is stopped.
pub struct ScopeInvalidation {
    _state: Arc<ReadScopeState>,
    _callback: Arc<Invalidator>,
}

pub(crate) struct ScopeReset<'a> {
    callbacks: MutexGuard<'a, Vec<Weak<Invalidator>>>,
}

impl ScopeReset<'_> {
    pub(crate) fn invalidate(&mut self) -> Result<()> {
        self.callbacks
            .retain(|callback| callback.strong_count() > 0);
        let mut failed = None;
        for callback in self.callbacks.iter().filter_map(Weak::upgrade) {
            if let Err(error) = callback() {
                failed.get_or_insert(error);
            }
        }
        failed.map_or(Ok(()), Err)
    }
}

fn poisoned() -> StoreError {
    StoreError::Invalid("scope publication fence poisoned".into())
}
#[derive(Debug, Clone)]
pub struct ReadScopeToken {
    state: Arc<ReadScopeState>,
    epoch: u64,
}

pub(crate) fn state(root: &Path) -> Arc<ReadScopeState> {
    static ARCHIVES: OnceLock<Mutex<HashMap<PathBuf, Weak<ReadScopeState>>>> = OnceLock::new();
    let key = root_key(root);
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
    pub(crate) fn register(
        self: &Arc<Self>,
        callback: impl Fn() -> Result<()> + Send + Sync + 'static,
    ) -> Result<ScopeInvalidation> {
        let callback: Arc<Invalidator> = Arc::new(callback);
        let mut fence = self.fence.lock().map_err(|_| poisoned())?;
        // Install before the first snapshot. Startup cannot expose a pointer
        // from an earlier process while its current scope is still unknown.
        callback()?;
        fence.retain(|callback| callback.strong_count() > 0);
        fence.push(Arc::downgrade(&callback));
        Ok(ScopeInvalidation {
            _state: Arc::clone(self),
            _callback: callback,
        })
    }

    pub(crate) fn token(self: &Arc<Self>) -> Result<ReadScopeToken> {
        let token = ReadScopeToken {
            state: Arc::clone(self),
            epoch: self.epoch.load(Ordering::Acquire),
        };
        token.check()?;
        Ok(token)
    }
    pub(crate) fn begin_reset(&self) -> Result<ScopeReset<'_>> {
        self.resetting.store(true, Ordering::Release);
        let callbacks = self.fence.lock().map_err(|_| poisoned())?;
        self.epoch
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |epoch| {
                epoch.checked_add(1)
            })
            .map_err(|_| StoreError::Invalid("read scope epoch exhausted".into()))?;
        Ok(ScopeReset { callbacks })
    }
    pub(crate) fn finish_reset(&self) {
        self.resetting.store(false, Ordering::Release);
    }
}
impl ReadScopeToken {
    /// Fence one pointer activation against scope invalidation. Build the cache
    /// outside this guard; the callback must not re-enter the store or scope API.
    pub fn with_current<T>(&self, publish: impl FnOnce() -> Result<T>) -> Result<T> {
        self.check()?;
        let _fence = self.state.fence.lock().map_err(|_| poisoned())?;
        self.check()?;
        publish()
    }

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
