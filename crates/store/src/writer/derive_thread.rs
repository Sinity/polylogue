//! The derive thread: it runs derive passes beside the writer, reading each
//! session's rows on its own read connection and computing its derived
//! values there, and sends each wave's values to the writer, which only
//! applies them ([`crate::summary::apply`]). Parse batches and user writes
//! are served while a derivation computes. A reader waiting for one due
//! session's summary asks this thread directly
//! ([`crate::summary::ask`]); it is answered before the next session of the
//! pass in progress, or at once when none runs.

use super::*;
use std::sync::mpsc::{Sender, channel};

/// Who waits for a derive pass.
pub(super) enum Waiting {
    /// [`WriterSender::derive`]: the pairs derived.
    Derive(SyncSender<Result<u64>>),
    /// A flush that also derives: its report, sent once the pass is done.
    Flush(SyncSender<Result<WriteReport>>, Box<WriteReport>),
    /// A background pass, requested without waiting for its derived values.
    Nobody,
    /// One session a reader waits for.
    Session(crate::summary::ask::Asked),
}

/// The derive thread's handle. Dropping it stops the thread: a computation
/// in progress is interrupted.
pub(super) struct DeriveThread {
    requests: Option<Sender<Waiting>>,
    /// A child of the archive's stop request, also cancelled when the writer ends.
    stopping: crate::cancellation::Cancellation,
    interrupt: Arc<Mutex<Option<rusqlite::InterruptHandle>>>,
    handle: Option<JoinHandle<()>>,
    /// Where readers of this archive ask for sessions, and this thread's
    /// registration there.
    asking: (Arc<crate::summary::ask::Asking>, u64),
}

impl DeriveThread {
    pub(super) fn start(
        index_db: std::path::PathBuf,
        queue: WriterQueue,
        stopping: crate::cancellation::Cancellation,
        cache_mib: i64,
    ) -> Result<Self> {
        let (requests, rx) = channel::<Waiting>();
        let asking = Arc::clone(&queue.location.summaries);
        let asked = Mutex::new(requests.clone());
        let registration = asking.serve(move |question| {
            if let Ok(asked) = asked.lock()
                && let Err(refused) = asked.send(Waiting::Session(question))
                && let Waiting::Session(question) = refused.0
            {
                question.answer(Ok(false));
            }
        });
        let interrupt = Arc::new(Mutex::new(None));
        let thread = Worker {
            index_db,
            cache_mib,
            queue,
            stopping: stopping.clone(),
            interrupt: Arc::clone(&interrupt),
            conn: None,
            unreported: Unreported::default(),
        };
        let handle = polylogue_model::spawn_named("polylogue-derive", move || thread.run(&rx))?;
        Ok(Self {
            requests: Some(requests),
            stopping,
            interrupt,
            handle: Some(handle),
            asking: (asking, registration),
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
            Some(Waiting::Flush(reply, mut report)) => {
                report.errors.push(polylogue_model::Failure::new(
                    "derive",
                    &StoreError::WriterGone,
                ));
                let _ = reply.send(Ok(*report));
            }
            Some(Waiting::Session(question)) => question.answer(Err(StoreError::WriterGone)),
            Some(Waiting::Nobody) | None => {}
        }
    }

    pub(super) fn stop(&mut self) {
        // Readers stop asking first: the registration holds a sender.
        self.asking.0.end(self.asking.1);
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
    cache_mib: i64,
    queue: WriterQueue,
    stopping: crate::cancellation::Cancellation,
    interrupt: Arc<Mutex<Option<rusqlite::InterruptHandle>>>,
    /// Opened at the first pass: a first build's index has one connection.
    conn: Option<Connection>,
    /// Failures of passes nobody waited for, reported with the next flush.
    unreported: Unreported,
}

/// Failures of background passes, kept as the latest message and a count so
/// a failure that repeats for the life of the process does not grow memory.
#[derive(Default)]
pub(super) struct Unreported {
    latest: Option<polylogue_model::Failure>,
    count: u64,
}

impl Unreported {
    pub(super) fn push(&mut self, error: polylogue_model::Failure) {
        self.latest = Some(error);
        self.count += 1;
    }

    /// Moves what was kept into `errors`, leaving nothing kept.
    pub(super) fn drain_into(&mut self, errors: &mut Vec<polylogue_model::Failure>) {
        let count = std::mem::take(&mut self.count);
        if let Some(mut latest) = self.latest.take() {
            if count > 1 {
                latest.message = format!("{} (and {} earlier failures)", latest.message, count - 1);
            }
            errors.push(latest);
        }
    }
}

impl Worker {
    fn run(mut self, requests: &Receiver<Waiting>) {
        let _worker = crate::timing::worker();
        let mut queued = Vec::new();
        loop {
            if queued.is_empty() {
                match requests.recv() {
                    Ok(first) => queued.push(first),
                    Err(_) => break,
                }
            }
            queued.extend(requests.try_iter());
            // Readers' sessions first; every other request waiting now is
            // answered by one pass, which starts after all of them were made.
            let (asked, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut queued)
                .into_iter()
                .partition(|w| matches!(w, Waiting::Session(_)));
            self.answer(
                asked
                    .into_iter()
                    .filter_map(|w| match w {
                        Waiting::Session(question) => Some(question),
                        _ => None,
                    })
                    .collect(),
            );
            if waiting.is_empty() {
                continue;
            }
            // Requests that arrive during the pass: readers' sessions are
            // answered between its sessions, the rest after it.
            let later = std::cell::RefCell::new(Vec::new());
            let asked = || {
                let mut asked = Vec::new();
                for w in requests.try_iter() {
                    match w {
                        Waiting::Session(question) => asked.push(question),
                        other => later.borrow_mut().push(other),
                    }
                }
                asked
            };
            let result = crate::timing::time(crate::timing::Stage::Derive, || self.pass(&asked));
            queued.extend(later.into_inner());
            for waiting in waiting {
                match waiting {
                    Waiting::Derive(reply) => {
                        let _ = reply.send(match &result {
                            Ok(n) => Ok(*n),
                            Err(StoreError::WriterGone) => Err(StoreError::WriterGone),
                            Err(e) => Err(StoreError::Failed(polylogue_model::Failure::new(
                                "derive", e,
                            ))),
                        });
                    }
                    Waiting::Flush(reply, mut report) => {
                        self.unreported.drain_into(&mut report.errors);
                        if let Err(e) = &result {
                            report
                                .errors
                                .push(polylogue_model::Failure::new("derive", e));
                        }
                        let _ = reply.send(Ok(*report));
                    }
                    Waiting::Nobody => {
                        if let Err(e) = &result {
                            self.unreported
                                .push(polylogue_model::Failure::new("derive", e));
                        }
                    }
                    Waiting::Session(_) => unreachable!("answered before the pass"),
                }
            }
        }
    }

    /// Computes the sessions readers wait for. While no index this build
    /// reads is installed none is computed: there is nothing to read.
    fn answer(&mut self, asked: Vec<crate::summary::ask::Asked>) {
        if asked.is_empty() {
            return;
        }
        if !self
            .queue
            .location
            .index_files()
            .is_ok_and(|files| files.serves())
        {
            for question in asked {
                question.answer(Ok(false));
            }
            return;
        }
        let pass = match self.open().and_then(|_| crate::summary::Pass::builtin()) {
            Ok(pass) => pass,
            Err(e) => {
                let failure = polylogue_model::Failure::new("derive", &e);
                for question in asked {
                    question.answer(Err(StoreError::Failed(failure.clone())));
                }
                return;
            }
        };
        let conn = self.conn.as_ref().expect("opened above");
        let queue = &self.queue;
        let stopping = &self.stopping;
        let result = pass.answer(
            conn,
            &mut |write| queue.ask(|reply| Msg::DeriveWrite(Box::new(write), reply)),
            &|| stopping.is_cancelled(),
            asked,
        );
        if let Err(e) = result {
            self.unreported
                .push(polylogue_model::Failure::new("derive", &e));
        }
    }

    fn open(&mut self) -> Result<()> {
        if self.conn.is_none() {
            let conn = crate::archive::open_read(&self.index_db)?;
            conn.pragma_update(None, "cache_size", -(self.cache_mib * 1024))?;
            let stopping = self.stopping.clone();
            conn.progress_handler(10_000, Some(move || stopping.is_cancelled()))?;
            *self.interrupt.lock().map_err(|_| StoreError::WriterGone)? =
                Some(conn.get_interrupt_handle());
            self.conn = Some(conn);
        }
        Ok(())
    }

    fn pass(&mut self, asked: &dyn Fn() -> Vec<crate::summary::ask::Asked>) -> Result<u64> {
        self.open()?;
        let conn = self.conn.as_ref().expect("opened above");
        let pass = crate::summary::Pass::builtin()?;
        let queue = &self.queue;
        let stopping = &self.stopping;
        let result = pass.run(
            conn,
            &mut |write| {
                #[cfg(any(test, feature = "fixture"))]
                queue
                    .archive
                    .dispatch()
                    .reach(DERIVE_COMPUTED, DispatchPhase::Owned)?;
                queue.ask(|reply| Msg::DeriveWrite(Box::new(write), reply))
            },
            &|| stopping.is_cancelled(),
            asked,
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
        self.0
            .archive
            .pause_user_dispatch_for_test(DERIVE_COMPUTED, DispatchPhase::Owned)
    }

    /// Holds the archive writer at `phase` of the user write `id`.
    pub fn pause_user_dispatch_for_test(&self, id: &str, phase: DispatchPhase) -> DispatchPause {
        self.0.archive.pause_user_dispatch_for_test(id, phase)
    }
}
