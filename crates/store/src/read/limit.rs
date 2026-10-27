//! One request's clock and cancellation, shared by every reader it opens.
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

thread_local! {
    static CURRENT: RefCell<Option<ReadLimit>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, Default)]
pub struct ReadCancellation(Arc<CancellationState>);

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    parent: Option<ReadCancellation>,
}

impl ReadCancellation {
    pub fn child(&self) -> Self {
        Self(Arc::new(CancellationState {
            cancelled: AtomicBool::new(false),
            parent: Some(self.clone()),
        }))
    }

    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
            || self.0.parent.as_ref().is_some_and(Self::is_cancelled)
    }
}

#[derive(Clone, Debug)]
pub struct ReadLimit {
    deadline: Instant,
    steps: Arc<AtomicU64>,
    stage: Arc<Mutex<Option<&'static str>>>,
    pub cancellation: ReadCancellation,
}

impl ReadLimit {
    pub fn new(duration: Duration, cancellation: ReadCancellation) -> Self {
        Self {
            deadline: Instant::now() + duration,
            cancellation,
            steps: Arc::new(AtomicU64::new(0)),
            stage: Arc::new(Mutex::new(None)),
        }
    }

    pub fn stage(&self) -> Option<&'static str> {
        *self.stage.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub fn set_stage(stage: &'static str) {
        if let Some(limit) = Self::current() {
            *limit
                .stage
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(stage);
        }
    }

    pub fn sqlite_steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            Err(StoreError::ReadCancelled)
        } else if Instant::now() >= self.deadline {
            Err(StoreError::ReadDeadlineExceeded)
        } else {
            Ok(())
        }
    }

    /// Synchronous application work may open several readers sequentially.
    /// They share the request's original deadline, including time waiting for a slot.
    pub fn scope<T>(&self, work: impl FnOnce() -> T) -> T {
        struct Restore(Option<ReadLimit>);
        impl Drop for Restore {
            fn drop(&mut self) {
                CURRENT.with(|current| current.replace(self.0.take()));
            }
        }
        let _restore = Restore(CURRENT.with(|current| current.replace(Some(self.clone()))));
        work()
    }

    pub fn current() -> Option<Self> {
        CURRENT.with(|current| current.borrow().clone())
    }

    pub(crate) fn install(&self, conn: &Connection) -> Result<()> {
        self.check()?;
        let limit = self.clone();
        // A single SQLite step can scan the whole archive without yielding a row.
        conn.progress_handler(
            1000,
            Some(move || {
                limit.steps.fetch_add(1000, Ordering::Relaxed);
                limit.check().is_err()
            }),
        )?;
        conn.busy_handler(Some(|_| {
            std::thread::sleep(Duration::from_millis(5));
            Self::current().is_some_and(|limit| limit.check().is_ok())
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
