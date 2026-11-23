//! Cancellation: the process's stop request and the requests beneath it.
//!
//! Every [`Archive`](crate::Archive) carries one stop request
//! ([`Cancellation::stop_request`]); the daemon cancels it to stop. Each
//! request's own cancellation (a hang-up, a dropped stream) is a child of it,
//! so one stop reaches every read, wait and long write. Store connections
//! observe it through their SQLite progress handler ([`interrupt_on`]),
//! blocking waits through [`Cancellation::on_cancel`], and loops between
//! steps through [`Cancellation::check`]. A thread doing a process's
//! background work runs [`under`] its stop request, so the connections the
//! store opens there and the byte reads that call [`check_current`] observe it
//! without being handed it.
use crate::{Result, StoreError};
use polylogue_model::LockExt;
use rusqlite::Connection;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

thread_local! {
    /// Why this thread's last statement was interrupted, recorded by the
    /// progress handler that interrupted it ([`interrupt_on`]).
    static INTERRUPTED_BY: RefCell<Option<StoreError>> = const { RefCell::new(None) };
    /// The stop request this thread's background work runs under ([`under`]).
    static THREAD_STOP: RefCell<Option<Cancellation>> = const { RefCell::new(None) };
}

#[derive(Clone, Debug, Default)]
pub struct Cancellation(Arc<State>);

type Wake = dyn Fn() + Send + Sync;

#[derive(Default)]
struct State {
    cancelled: AtomicBool,
    /// This is a process's stop request rather than one caller's withdrawal.
    stops_process: bool,
    parent: Option<Cancellation>,
    /// Waiters to wake when this cancellation or an ancestor is cancelled.
    wakers: Mutex<Vec<(u64, Arc<Wake>)>>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Cancellation")
            .field("cancelled", &self.cancelled)
            .field("stops_process", &self.stops_process)
            .field("parent", &self.parent)
            .finish_non_exhaustive()
    }
}

/// Cancels a request's reads when the request is dropped before it is
/// finished, so a hang-up or an abandoned future stops the blocking reader.
/// [`disarm`](Self::disarm) hands the cancellation to whatever outlives the
/// guard, such as a response stream.
pub struct CancelOnDrop {
    cancellation: Cancellation,
    armed: bool,
}

impl CancelOnDrop {
    pub fn new(cancellation: Cancellation) -> Self {
        Self {
            cancellation,
            armed: true,
        }
    }

    pub fn disarm(&mut self) {
        self.armed = false;
    }

    pub fn check(&self) -> Result<()> {
        self.cancellation.check()
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.cancel();
        }
    }
}

/// Keeps a waker registered on a cancellation and its ancestors; dropping
/// it removes the waker.
pub struct CancellationWake {
    registered: Vec<(Cancellation, u64)>,
}

impl Drop for CancellationWake {
    fn drop(&mut self) {
        for (cancellation, id) in &self.registered {
            cancellation
                .0
                .wakers
                .locked()
                .retain(|(registered, _)| registered != id);
        }
    }
}

impl Cancellation {
    /// A process's stop request. Cancelling it ends the process's reads,
    /// waits and long writes with [`StoreError::Stopped`].
    pub(crate) fn stop_request() -> Self {
        Self(Arc::new(State {
            stops_process: true,
            ..State::default()
        }))
    }

    pub fn child(&self) -> Self {
        Self(Arc::new(State {
            parent: Some(self.clone()),
            ..State::default()
        }))
    }

    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Release);
        let wakers: Vec<_> = self
            .0
            .wakers
            .locked()
            .iter()
            .map(|(_, wake)| Arc::clone(wake))
            .collect();
        for wake in wakers {
            wake();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.check().is_err()
    }

    /// [`StoreError::ReadCancelled`] when the caller withdrew,
    /// [`StoreError::Stopped`] when only the process's stop request is
    /// cancelled. A withdrawal is reported first: it is the narrower cause.
    pub fn check(&self) -> Result<()> {
        let mut stopped = false;
        let mut current = Some(self);
        while let Some(cancellation) = current {
            if cancellation.0.cancelled.load(Ordering::Acquire) {
                if !cancellation.0.stops_process {
                    return Err(StoreError::ReadCancelled);
                }
                stopped = true;
            }
            current = cancellation.0.parent.as_ref();
        }
        if stopped {
            Err(StoreError::Stopped)
        } else {
            Ok(())
        }
    }

    /// Calls `wake` when this cancellation or an ancestor is cancelled, for
    /// as long as the returned guard lives, so a blocked waiter needs no
    /// timer to notice. A cancellation made before the registration calls
    /// `wake` at once.
    pub fn on_cancel(&self, wake: impl Fn() + Send + Sync + 'static) -> CancellationWake {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let wake: Arc<Wake> = Arc::new(wake);
        let mut registered = Vec::new();
        let mut current = Some(self.clone());
        while let Some(cancellation) = current {
            cancellation.0.wakers.locked().push((id, Arc::clone(&wake)));
            current = cancellation.0.parent.clone();
            registered.push((cancellation, id));
        }
        // A cancel that copied its wakers before this registration has
        // already passed; this waiter learns of it here.
        if self.is_cancelled() {
            wake();
        }
        CancellationWake { registered }
    }
}

/// Runs `work`, a process's background work on this thread, under `stop`:
/// connections the store opens on this thread and byte reads that call
/// [`check_current`] end with [`StoreError::Stopped`] once it is cancelled.
pub fn under<T>(stop: &Cancellation, work: impl FnOnce() -> T) -> T {
    struct Restore(Option<Cancellation>);
    impl Drop for Restore {
        fn drop(&mut self) {
            THREAD_STOP.with(|current| current.replace(self.0.take()));
        }
    }
    let _restore = Restore(THREAD_STOP.with(|current| current.replace(Some(stop.clone()))));
    work()
}

/// The stop request this thread runs [`under`], if any.
pub(crate) fn thread_stop() -> Option<Cancellation> {
    THREAD_STOP.with(|current| current.borrow().clone())
}

/// Checks what the current thread's work runs under: its stop request
/// ([`under`]) and its request's deadline and cancellation
/// ([`ReadLimit::scope`](crate::read::limit::ReadLimit::scope)). Long byte
/// reads call it between buffers.
pub fn check_current() -> Result<()> {
    if let Some(stop) = thread_stop() {
        stop.check()?;
    }
    crate::read::limit::ReadLimit::current().map_or(Ok(()), |limit| limit.check())
}

/// Installs `check` as `conn`'s SQLite progress handler: an error from it
/// interrupts the running statement, and that error, not SQLite's
/// interruption, becomes the statement's error.
pub(crate) fn interrupt_on(
    conn: &Connection,
    check: impl Fn() -> Result<()> + Send + 'static,
) -> Result<()> {
    // A single SQLite step can scan the whole archive without yielding a row.
    conn.progress_handler(
        1000,
        Some(move || match check() {
            Ok(()) => false,
            Err(error) => {
                INTERRUPTED_BY.with(|cause| cause.replace(Some(error)));
                true
            }
        }),
    )?;
    Ok(())
}

/// The cause [`interrupt_on`] recorded for this thread's interrupted statement.
pub(crate) fn interrupted_by() -> Option<StoreError> {
    INTERRUPTED_BY.with(RefCell::take)
}

#[cfg(test)]
mod tests;
