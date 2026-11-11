//! The derive thread: it runs derive passes beside the writer, reading each
//! session's rows on its own read connection and computing its derived
//! values there, and sends each wave's values to the writer, which only
//! applies them ([`crate::derive::apply`]). Parse batches and user writes
//! are served while a derivation computes.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Sender, channel};

/// Who waits for a derive pass.
pub(super) enum Waiting {
    /// [`Writer::derive`]: the pairs derived.
    Derive(SyncSender<Result<u64>>),
    /// A flush that also derives: its report, sent once the pass is done.
    Flush(SyncSender<WriteReport>, Box<WriteReport>),
    /// Nobody: a pass after a page of the confirmed stream.
    Nobody,
}

/// The derive thread's handle. Dropping it stops the thread: a computation
/// in progress is interrupted.
pub(super) struct DeriveThread {
    requests: Option<Sender<Waiting>>,
    stopping: Arc<AtomicBool>,
    interrupt: Arc<Mutex<Option<rusqlite::InterruptHandle>>>,
    handle: Option<JoinHandle<()>>,
}

impl DeriveThread {
    pub(super) fn start(
        index_db: std::path::PathBuf,
        queue: WriterQueue,
        stopping: Arc<AtomicBool>,
    ) -> Result<Self> {
        let (requests, rx) = channel::<Waiting>();
        let interrupt = Arc::new(Mutex::new(None));
        let thread = Worker {
            index_db,
            queue,
            stopping: Arc::clone(&stopping),
            interrupt: Arc::clone(&interrupt),
            conn: None,
            unreported: Vec::new(),
        };
        let handle = std::thread::Builder::new()
            .name("polylogue-derive".into())
            .spawn(move || thread.run(&rx))?;
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
                let _ = reply.send(*report);
            }
            Some(Waiting::Nobody) | None => {}
        }
    }

    pub(super) fn stop(&mut self) {
        self.requests.take();
        self.stopping.store(true, Ordering::Release);
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
    stopping: Arc<AtomicBool>,
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
                            Err(e) => Err(StoreError::Invalid(format!("derive: {e}"))),
                        });
                    }
                    Waiting::Flush(reply, mut report) => {
                        report.errors.append(&mut self.unreported);
                        if let Err(e) = &result {
                            report.errors.push(format!("derive: {e}"));
                        }
                        let _ = reply.send(*report);
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
            let stopping = Arc::clone(&self.stopping);
            conn.progress_handler(10_000, Some(move || stopping.load(Ordering::Acquire)))?;
            *self.interrupt.lock().map_err(|_| StoreError::WriterGone)? =
                Some(conn.get_interrupt_handle());
            self.conn = Some(conn);
        }
        let conn = self.conn.as_ref().expect("opened above");
        let pass = crate::derive::Pass::builtin()?;
        let queue = &self.queue;
        let stopping = &self.stopping;
        pass.run(
            conn,
            &mut |write| {
                #[cfg(any(test, feature = "fixture"))]
                queue
                    .dispatch
                    .reach(DERIVE_COMPUTED, DispatchPhase::Owned)?;
                let (reply, wait) = sync_channel(1);
                queue
                    .send(Msg::DeriveWrite(Box::new(write), reply))
                    .map_err(|_| StoreError::WriterGone)?;
                wait.recv().map_err(|_| StoreError::WriterGone)?
            },
            &|| stopping.load(Ordering::Acquire),
        )
    }
}

/// The pause point between a wave's computation and its write, for tests
/// that hold a derivation as if it computed slowly.
#[cfg(any(test, feature = "fixture"))]
pub(super) const DERIVE_COMPUTED: &str = "derive: computed";

#[cfg(any(test, feature = "fixture"))]
impl WriterSender {
    /// Holds the derive thread once it computed its next wave, before the
    /// writer is sent anything, until the barrier is released.
    pub fn pause_derive_for_test(&self) -> DispatchBarrier {
        self.pause_user_dispatch_for_test(DERIVE_COMPUTED, DispatchPhase::Owned)
    }
}
