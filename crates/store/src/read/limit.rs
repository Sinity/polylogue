//! One request's clock and cancellation, shared by every reader it opens.
use crate::cancellation::Cancellation;
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

thread_local! {
    static CURRENT: RefCell<Option<ReadLimit>> = const { RefCell::new(None) };
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
    pub cancellation: Cancellation,
}

impl ReadLimit {
    pub fn new(duration: Duration, cancellation: Cancellation) -> Self {
        Self::with_deadline(Instant::now().checked_add(duration), cancellation)
    }

    /// Bind an already admitted absolute deadline without renewing its budget.
    pub fn at(deadline: Instant, cancellation: Cancellation) -> Self {
        Self::with_deadline(Some(deadline), cancellation)
    }

    /// Export has cancellation even when its caller has asked for no deadline.
    pub fn without_deadline(cancellation: Cancellation) -> Self {
        Self::with_deadline(None, cancellation)
    }

    fn with_deadline(deadline: Option<Instant>, cancellation: Cancellation) -> Self {
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
        self.cancellation.check()?;
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            Err(StoreError::ReadDeadlineExceeded)
        } else {
            Ok(())
        }
    }

    /// [`Self::check`] from a progress handler, which runs every thousand
    /// SQLite steps.
    pub(crate) fn count_and_check(&self) -> Result<()> {
        self.steps.fetch_add(1000, Ordering::Relaxed);
        self.check()
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

    /// Makes a connection the store opens observe what this thread's work
    /// runs under: its stop request and its request's limit, if any.
    pub(crate) fn observe_current(conn: &Connection) -> Result<()> {
        let limit = Self::current();
        let stop = crate::cancellation::thread_stop();
        if let Some(limit) = &limit {
            limit.install(conn)?;
        }
        if let Some(stop) = stop {
            stop.check()?;
            crate::cancellation::interrupt_on(conn, move || {
                stop.check()?;
                limit.as_ref().map_or(Ok(()), Self::count_and_check)
            })?;
        }
        Ok(())
    }

    pub(crate) fn install(&self, conn: &Connection) -> Result<()> {
        self.check()?;
        let limit = self.clone();
        crate::cancellation::interrupt_on(conn, move || limit.count_and_check())?;
        conn.busy_handler(Some(|attempt| {
            std::thread::sleep(Duration::from_millis(5));
            Self::current().map_or(attempt < 6000, |limit| limit.check().is_ok())
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
