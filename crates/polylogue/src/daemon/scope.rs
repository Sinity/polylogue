//! The daemon's scope index ([`ScopeIndex`]): read in full when the daemon
//! starts and brought forward after every committed write, on a worker of
//! its own, so a brief or a resume list never reads it from the archive.
//! The same worker reads the status snapshot from the same reader, so a
//! status request never reads the archive either.
//! A caller copies what it needs out under the lock and reads the archive
//! after letting go of it.

use polylogue_model::LockExt as _;
use polylogue_store::completion::CommitSubscription;
use polylogue_store::{Reader, Result, ScopeIndex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

const TERMINATED: &str = "scope worker terminated";

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
            .locked()
            .as_ref()
            .and_then(|index| index.position().cloned());
        // Read without the lock: briefs keep answering from the last state.
        let changes = reader.scope_changes(after.as_ref())?;
        let mut held = self.index.locked();
        match held.as_mut() {
            // Another refresh already applied a newer snapshot.
            Some(index) if index.is_newer_than(&changes) => {}
            Some(index) => index.apply(changes),
            None => *held = Some(ScopeIndex::new(changes)),
        }
        self.loaded.notify_all();
        Ok(())
    }

    /// Whether the first full read has finished.
    pub fn is_loaded(&self) -> bool {
        self.index.locked().is_some()
    }

    /// Optional hints never wait for a scope change or its lock.
    pub fn try_read<T>(&self, read: impl FnOnce(&ScopeIndex) -> T) -> Option<T> {
        // A lock an earlier owner panicked under is read like any other.
        let index = match self.index.try_lock() {
            Ok(index) => index,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        index.as_ref().map(read)
    }

    /// Runs `read` over the index, waiting for its first full read until
    /// `until` at most; `None` when that read has not finished by then.
    pub fn read<T>(
        &self,
        until: Option<Instant>,
        read: impl FnOnce(&ScopeIndex) -> T,
    ) -> Option<T> {
        let mut held = self.index.locked();
        while held.is_none() {
            match until {
                Some(until) => {
                    let left = until.saturating_duration_since(Instant::now());
                    if left.is_zero() {
                        return None;
                    }
                    held = polylogue_model::unpoisoned(self.loaded.wait_timeout(held, left)).0;
                }
                None => {
                    held = polylogue_model::unpoisoned(self.loaded.wait(held));
                }
            }
        }
        held.as_ref().map(read)
    }
}

/// The worker that keeps [`SharedScopeIndex`] and the status snapshot
/// current and, after each commit, hands delivery records that found the
/// writer's queue full to the writer.
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
        let thread = polylogue_model::spawn_named("polylogue-scope", move || {
            let _exit = super::lifecycle::WorkerExit::new(
                worker_events,
                TERMINATED,
                Some(Arc::clone(&completed)),
            );
            let expected = polylogue_store::follow::follow(
                "scope index",
                subscription.as_ref(),
                &stopped,
                || daemon.config.requests.now(),
                || daemon.status.changed(),
                || {
                    let commits_read = daemon
                        .writes
                        .as_ref()
                        .map_or(0, polylogue_store::WriterSender::commits_made);
                    let result = daemon.reads.reader().and_then(|reader| {
                        daemon.scope.refresh(&reader)?;
                        daemon
                            .status_snapshot
                            .refresh(&daemon, &reader, commits_read);
                        Ok(())
                    });
                    if matches!(result, Err(polylogue_store::StoreError::IndexBuilding(_))) {
                        daemon.status_snapshot.clear(&daemon);
                    }
                    result?;
                    if let Some(writes) = &daemon.writes
                        && let Err(error) = daemon.queued_deliveries.hand_over(writes)
                    {
                        polylogue_store::log!(warning, "queued deliveries: {error}");
                    }
                    Ok(())
                },
            );
            completed.store(expected, Ordering::Release);
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
        .then_some(TERMINATED)
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
                return Err(std::io::Error::other(TERMINATED));
            }
        }
        Ok(())
    }
}

impl Drop for ScopeWorker {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            polylogue_store::log!(error, "scope worker shutdown: {error}");
        }
    }
}
