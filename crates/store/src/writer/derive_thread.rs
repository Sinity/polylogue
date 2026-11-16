//! The derive thread: it runs derive passes beside the writer, reading each
//! session's rows on its own read connection and computing its derived
//! values there, and sends each wave's values to the writer, which only
//! applies them ([`crate::derive::apply`]). Parse batches and user writes
//! are served while a derivation computes.

use super::*;
use std::sync::mpsc::{Sender, channel};

/// Who waits for a derive pass.
pub(super) enum Waiting {
    /// [`WriterSender::derive`]: the pairs derived.
    Derive(SyncSender<Result<u64>>),
    /// A flush that also derives: its report, sent once the pass is done.
    Flush(SyncSender<Result<WriteReport>>, Box<WriteReport>),
    /// Nobody: a pass after a page of the confirmed stream.
    Nobody,
}

/// The derive thread's handle. Dropping it stops the thread: a computation
/// in progress is interrupted.
pub(super) struct DeriveThread {
    requests: Option<Sender<Waiting>>,
    /// A child of the archive's stop request, also cancelled when the writer ends.
    stopping: crate::cancellation::Cancellation,
    interrupt: Arc<Mutex<Option<rusqlite::InterruptHandle>>>,
    handle: Option<JoinHandle<()>>,
}

impl DeriveThread {
    pub(super) fn start(
        index_db: std::path::PathBuf,
        queue: WriterQueue,
        stopping: crate::cancellation::Cancellation,
    ) -> Result<Self> {
        let (requests, rx) = channel::<Waiting>();
        let interrupt = Arc::new(Mutex::new(None));
        let thread = Worker {
            index_db,
            queue,
            stopping: stopping.clone(),
            interrupt: Arc::clone(&interrupt),
            conn: None,
            unreported: Vec::new(),
        };
        let handle = polylogue_model::spawn_named("polylogue-derive", move || thread.run(&rx))?;
        Ok(Self {
            requests: Some(requests),
            stopping,
            interrupt,
            handle: Some(handle),
        })
    }

    /// Queues a pass; `waiting` is answered when it is done. The writer
    /// has marked what moved and committed everything the pass must see.
    pub(super) fn request(&self, waiting: Waiting) {
        let refused = match &self.requests {
            Some(requests) => requests.send(waiting).err().map(|e| e.0),
            None => Some(waiting),
        };
        match refused {
            Some(Waiting::Derive(reply)) => {
                let _ = reply.send(Err(StoreError::WriterGone));
            }
            Some(Waiting::Flush(reply, report)) => {
                let _ = reply.send(Ok(*report));
            }
            Some(Waiting::Nobody) | None => {}
        }
    }

    pub(super) fn stop(&mut self) {
        self.requests.take();
        self.stopping.cancel();
        if let Ok(interrupt) = self.interrupt.lock()
            && let Some(interrupt) = interrupt.as_ref()
        {
            interrupt.interrupt();
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DeriveThread {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Worker {
    index_db: std::path::PathBuf,
    queue: WriterQueue,
    stopping: crate::cancellation::Cancellation,
    interrupt: Arc<Mutex<Option<rusqlite::InterruptHandle>>>,
    /// Opened at the first pass: a first build's index has one connection.
    conn: Option<Connection>,
    /// Failures of passes nobody waited for, reported with the next flush.
    unreported: Vec<String>,
}

impl Worker {
    fn run(mut self, requests: &Receiver<Waiting>) {
        let _worker = crate::timing::worker();
        while let Ok(first) = requests.recv() {
            // Every request waiting now is answered by one pass, which
            // starts after all of them were made.
            let mut waiting = vec![first];
            waiting.extend(requests.try_iter());
            let result = crate::timing::time(crate::timing::Stage::Derive, || self.pass());
            for waiting in waiting {
                match waiting {
                    Waiting::Derive(reply) => {
                        let _ = reply.send(match &result {
                            Ok(n) => Ok(*n),
                            Err(StoreError::WriterGone) => Err(StoreError::WriterGone),
                            Err(e) => Err(StoreError::Invalid(format!("derive: {e}"))),
                        });
                    }
                    Waiting::Flush(reply, mut report) => {
                        report.errors.append(&mut self.unreported);
                        if let Err(e) = &result {
                            report.errors.push(format!("derive: {e}"));
                        }
                        let _ = reply.send(Ok(*report));
                    }
                    Waiting::Nobody => {
                        if let Err(e) = &result {
                            self.unreported.push(format!("derive: {e}"));
                        }
                    }
                }
            }
        }
    }

    fn pass(&mut self) -> Result<u64> {
        if self.conn.is_none() {
            let conn = crate::archive::open_read(&self.index_db)?;
            conn.pragma_update(None, "cache_size", -(crate::archive::cache::DERIVE * 1024))?;
            let stopping = self.stopping.clone();
            conn.progress_handler(10_000, Some(move || stopping.is_cancelled()))?;
            *self.interrupt.lock().map_err(|_| StoreError::WriterGone)? =
                Some(conn.get_interrupt_handle());
            self.conn = Some(conn);
        }
        let conn = self.conn.as_ref().expect("opened above");
        let pass = crate::derive::Pass::builtin()?;
        let queue = &self.queue;
        let stopping = &self.stopping;
        let result = pass.run(
            conn,
            &mut |write| {
                #[cfg(any(test, feature = "fixture"))]
                queue
                    .dispatch
                    .reach(DERIVE_COMPUTED, DispatchPhase::Owned)?;
                queue.ask(|reply| Msg::DeriveWrite(Box::new(write), reply))
            },
            &|| stopping.is_cancelled(),
        );
        match result {
            Err(StoreError::Sqlite(error))
                if stopping.is_cancelled()
                    && error.sqlite_error_code()
                        == Some(rusqlite::ErrorCode::OperationInterrupted) =>
            {
                Err(StoreError::WriterGone)
            }
            result => result,
        }
    }
}

/// The pause point between a wave's computation and its write, for tests
/// that hold a derivation as if it computed slowly.
#[cfg(any(test, feature = "fixture"))]
pub(super) const DERIVE_COMPUTED: &str = "derive: computed";

#[cfg(any(test, feature = "fixture"))]
impl WriterSender {
    /// Holds the derive thread once it computed its next wave, before the
    /// writer is sent anything, until the pause is released.
    #[cfg(any(test, feature = "fixture"))]
    pub fn pause_derive_for_test(&self) -> DispatchPause {
        self.pause_user_dispatch_for_test(DERIVE_COMPUTED, DispatchPhase::Owned)
    }
}
