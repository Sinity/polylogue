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

type Wake = dyn Fn() + Send + Sync;

#[derive(Default)]
struct CancellationState {
    cancelled: AtomicBool,
    parent: Option<ReadCancellation>,
    /// Waiters to wake when this cancellation or an ancestor is cancelled.
    wakers: Mutex<Vec<(u64, Arc<Wake>)>>,
}

impl std::fmt::Debug for CancellationState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CancellationState")
            .field("cancelled", &self.cancelled)
            .field("parent", &self.parent)
            .finish_non_exhaustive()
    }
}

/// Keeps a waker registered on a cancellation and its ancestors; dropping
/// it removes the waker.
pub struct CancellationWake {
    registered: Vec<(ReadCancellation, u64)>,
}

impl Drop for CancellationWake {
    fn drop(&mut self) {
        for (cancellation, id) in &self.registered {
            cancellation
                .0
                .wakers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retain(|(registered, _)| registered != id);
        }
    }
}

impl ReadCancellation {
    pub fn child(&self) -> Self {
        Self(Arc::new(CancellationState {
            cancelled: AtomicBool::new(false),
            parent: Some(self.clone()),
            wakers: Mutex::default(),
        }))
    }

    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Release);
        let wakers: Vec<_> = self
            .0
            .wakers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .map(|(_, wake)| Arc::clone(wake))
            .collect();
        for wake in wakers {
            wake();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
            || self.0.parent.as_ref().is_some_and(Self::is_cancelled)
    }

    /// Calls `wake` when this cancellation or an ancestor is cancelled, for
    /// as long as the returned guard lives, so a blocked waiter needs no
    /// timer to notice.
    pub fn on_cancel(&self, wake: impl Fn() + Send + Sync + 'static) -> CancellationWake {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let wake: Arc<Wake> = Arc::new(wake);
        let mut registered = Vec::new();
        let mut current = Some(self.clone());
        while let Some(cancellation) = current {
            cancellation
                .0
                .wakers
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((id, Arc::clone(&wake)));
            current = cancellation.0.parent.clone();
            registered.push((cancellation, id));
        }
        CancellationWake { registered }
    }
}

#[derive(Clone, Debug, Default)]
struct Stages {
    current: Option<(&'static str, Instant)>,
    totals: Vec<(&'static str, Duration)>,
}

impl Stages {
    fn close(&mut self, now: Instant) {
        if let Some((name, began)) = self.current.take() {
            let spent = now.saturating_duration_since(began);
            match self.totals.iter_mut().find(|(total, _)| *total == name) {
                Some((_, total)) => *total += spent,
                None => self.totals.push((name, spent)),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReadLimit {
    deadline: Option<Instant>,
    steps: Arc<AtomicU64>,
    stages: Arc<Mutex<Stages>>,
    scopes: Arc<Mutex<Vec<super::scope::ReadScopeToken>>>,
    pub cancellation: ReadCancellation,
}

impl ReadLimit {
    pub fn new(duration: Duration, cancellation: ReadCancellation) -> Self {
        Self::with_deadline(Instant::now().checked_add(duration), cancellation)
    }

    /// Bind an already admitted absolute deadline without renewing its budget.
    pub fn at(deadline: Instant, cancellation: ReadCancellation) -> Self {
        Self::with_deadline(Some(deadline), cancellation)
    }

    /// Export has cancellation even when its caller has asked for no deadline.
    pub fn without_deadline(cancellation: ReadCancellation) -> Self {
        Self::with_deadline(None, cancellation)
    }

    fn with_deadline(deadline: Option<Instant>, cancellation: ReadCancellation) -> Self {
        Self {
            deadline,
            cancellation,
            steps: Arc::new(AtomicU64::new(0)),
            stages: Default::default(),
            scopes: Default::default(),
        }
    }

    /// When this request's reads stop, when they have a deadline.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn stage(&self) -> Option<&'static str> {
        self.stages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .current
            .map(|(name, _)| name)
    }

    /// Enters a named stage; the time since the previous stage began is added
    /// to that stage's total. Re-entering the current stage changes nothing.
    pub fn set_stage(stage: &'static str) {
        if let Some(limit) = Self::current() {
            let mut stages = limit
                .stages
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if stages.current.is_some_and(|(name, _)| name == stage) {
                return;
            }
            let now = Instant::now();
            stages.close(now);
            stages.current = Some((stage, now));
        }
    }

    /// Elapsed time per stage in first-entered order, the current stage
    /// counted up to now.
    pub fn stage_timings(&self) -> Vec<(&'static str, Duration)> {
        let mut stages = self
            .stages
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        stages.close(Instant::now());
        stages.totals
    }

    pub fn sqlite_steps(&self) -> u64 {
        self.steps.load(Ordering::Relaxed)
    }

    pub(crate) fn bind_scope(&self, scope: super::scope::ReadScopeToken) -> Result<()> {
        scope.check()?;
        let mut scopes = self
            .scopes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !scopes.iter().any(|existing| existing.same(&scope)) {
            scopes.push(scope);
        }
        Ok(())
    }

    /// The time left before the deadline; `None` with no deadline.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    pub fn check(&self) -> Result<()> {
        for scope in self
            .scopes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
        {
            scope.check()?;
        }
        if self.cancellation.is_cancelled() {
            Err(StoreError::ReadCancelled)
        } else if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
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

    /// Validate the completed read preparation before handing work to a writer.
    /// The enclosing scope restores its previous state when that work returns.
    pub fn finish_scope(stage: &'static str) -> Result<()> {
        if let Some(limit) = Self::current() {
            limit.check()?;
        }
        Self::set_stage(stage);
        CURRENT.with(|current| current.replace(None));
        Ok(())
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
        conn.busy_handler(Some(|attempt| {
            crate::probe_tmp::log(&format!("busy_handler(attempt={attempt})"), Duration::ZERO);
            std::thread::sleep(Duration::from_millis(5));
            Self::current().map_or(attempt < 6000, |limit| limit.check().is_ok())
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
