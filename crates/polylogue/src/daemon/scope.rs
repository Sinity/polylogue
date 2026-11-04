//! The daemon's scope index ([`ScopeIndex`]): read in full when the daemon
//! starts and brought forward after every committed write, on a worker of
//! its own, so a brief or a resume list never reads it from the archive.
//! A caller copies what it needs out under the lock and reads the archive
//! after letting go of it.

use polylogue_store::completion::CommitSubscription;
use polylogue_store::{Reader, Result, ScopeIndex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

/// The scope index and a signal for its first full read.
#[derive(Default)]
pub struct SharedScopeIndex {
    index: Mutex<Option<ScopeIndex>>,
    loaded: Condvar,
}

impl SharedScopeIndex {
    /// Brings the index forward to `reader`'s snapshot: in full the first
    /// time, then the sessions changed since its position.
    pub fn refresh(&self, reader: &Reader<'_>) -> Result<()> {
        let after = self
            .index
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .and_then(|index| index.position().cloned());
        // Read without the lock: briefs keep answering from the last state.
        let changes = reader.scope_changes(after.as_ref())?;
        let mut held = self.index.lock().unwrap_or_else(PoisonError::into_inner);
        match held.as_mut() {
            // Another refresh already applied a newer snapshot.
            Some(index) if index.is_newer_than(&changes) => {}
            Some(index) => index.apply(changes),
            None => *held = Some(ScopeIndex::new(changes)),
        }
        self.loaded.notify_all();
        Ok(())
    }

    /// Runs `read` over the index, waiting for its first full read until
    /// `until` at most; `None` when that read has not finished by then.
    pub fn read<T>(
        &self,
        until: Option<Instant>,
        read: impl FnOnce(&ScopeIndex) -> T,
    ) -> Option<T> {
        let mut held = self.index.lock().unwrap_or_else(PoisonError::into_inner);
        while held.is_none() {
            match until {
                Some(until) => {
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return None;
                    }
                    held = self
                        .loaded
                        .wait_timeout(held, left)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
                None => {
                    held = self
                        .loaded
                        .wait(held)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
        held.as_ref().map(read)
    }
}

/// The worker that keeps [`SharedScopeIndex`] current and, after each
/// commit, hands delivery records that found the writer's queue full to the
/// writer.
pub(crate) struct ScopeWorker {
    stop: Arc<AtomicBool>,
    expected_exit: Arc<AtomicBool>,
    wake: Option<SyncSender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl ScopeWorker {
    /// Reads the index in full, then again after each commit `subscription`
    /// reports. Without a subscription nothing writes, and it ends after the
    /// first read.
    pub fn start(
        daemon: Arc<super::service::Daemon>,
        subscription: Option<CommitSubscription>,
        worker_events: tokio::sync::watch::Sender<Option<&'static str>>,
    ) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let expected_exit = Arc::new(AtomicBool::new(false));
        let completed = Arc::clone(&expected_exit);
        let wake = subscription.as_ref().map(CommitSubscription::wake_handle);
        let thread = std::thread::Builder::new()
            .name("polylogue-scope".into())
            .spawn(move || {
                let _exit = super::lifecycle::WorkerExit::new(
                    worker_events,
                    "scope worker terminated",
                    Some(Arc::clone(&completed)),
                );
                let mut last_error = None;
                loop {
                    if stopped.load(Ordering::Acquire) {
                        completed.store(true, Ordering::Release);
                        return;
                    }
                    let result = daemon
                        .reads
                        .reader()
                        .and_then(|reader| daemon.scope.refresh(&reader));
                    if let Err(error) = result {
                        if stopped.load(Ordering::Acquire) {
                            completed.store(true, Ordering::Release);
                            return;
                        }
                        let error = error.to_string();
                        if last_error.as_ref() != Some(&error) {
                            eprintln!("polylogue: scope index: {error}");
                            last_error = Some(error);
                        }
                        // Retry from the same position; a commit cue is not needed.
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        continue;
                    }
                    last_error = None;
                    // A commit leaves room in the writer's queue for records
                    // that found it full.
                    if let Some(writes) = &daemon.writes
                        && let Err(error) = daemon.queued_deliveries.hand_over(writes)
                    {
                        eprintln!("polylogue: queued deliveries: {error}");
                    }
                    match &subscription {
                        Some(subscription) => {
                            if subscription.recv().is_err() {
                                if stopped.load(Ordering::Acquire) {
                                    completed.store(true, Ordering::Release);
                                }
                                return;
                            }
                        }
                        None => {
                            completed.store(true, Ordering::Release);
                            return;
                        }
                    }
                }
            })?;
        Ok(Self {
            stop,
            expected_exit,
            wake,
            thread: Some(thread),
        })
    }

    /// An exit nobody asked for, a panic included, is a failure.
    pub(crate) fn worker_failure(&self) -> Option<&'static str> {
        (self.thread.as_ref().is_some_and(JoinHandle::is_finished)
            && !self.stop.load(Ordering::Acquire)
            && !self.expected_exit.load(Ordering::Acquire))
        .then_some("scope worker terminated")
    }

    /// Stops and joins the worker.
    pub(crate) fn shutdown(&mut self) -> std::io::Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(wake) = &self.wake {
            let _ = wake.try_send(());
        }
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| std::io::Error::other("scope worker panicked"))?;
            if !self.expected_exit.load(Ordering::Acquire) {
                return Err(std::io::Error::other("scope worker terminated"));
            }
        }
        Ok(())
    }
}

impl Drop for ScopeWorker {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: scope worker shutdown: {error}");
        }
    }
}
