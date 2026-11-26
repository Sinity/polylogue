//! The search thread: shortly after rows commit, it reads the full-text
//! entries the writer queued in `search_pending` on its own read
//! connection, folds their text there, and sends each batch to the writer,
//! which only applies it ([`crate::text_rows::apply_entries`]). The writer
//! is the one connection that writes the index.

use super::*;

/// The search thread's handle. Dropping it stops the thread.
pub(super) struct Search {
    wake: Option<SyncSender<()>>,
    ready: Arc<std::sync::atomic::AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Search {
    pub(super) fn start(
        index_db: std::path::PathBuf,
        queue: WriterQueue,
        ready: bool,
        cache_mib: i64,
    ) -> Result<Self> {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(ready));
        let (wake, rx) = sync_channel::<()>(1);
        let seen = flag.clone();
        let handle = polylogue_model::spawn_named("polylogue-search", move || {
            search_thread(&index_db, &queue, &seen, &rx, cache_mib)
        })?;
        Ok(Self {
            wake: Some(wake),
            ready: flag,
            handle: Some(handle),
        })
    }

    pub(super) fn wake(&self) {
        if let Some(w) = &self.wake {
            let _ = w.try_send(());
        }
    }

    /// The index has left its first build: the thread may open it.
    pub(super) fn ready(&self) {
        self.ready.store(true, std::sync::atomic::Ordering::Release);
        self.wake();
    }

    /// Ends the thread. The writer has stopped taking messages, so a batch
    /// the thread sends is refused rather than waiting.
    pub(super) fn stop(&mut self) {
        self.wake.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Search {
    fn drop(&mut self) {
        // A writer unwinding from a panic may still hold a batch the thread
        // waits on; it ends on its own once the queue is gone.
        if std::thread::panicking() {
            self.wake.take();
            return;
        }
        self.stop();
    }
}

/// Computes waiting search entries whenever woken, and once a second. The
/// connection opens on the installed WAL index. A private build completes
/// its own search installation before publishing that file.
fn search_thread(
    index_db: &std::path::Path,
    queue: &WriterQueue,
    ready: &std::sync::atomic::AtomicBool,
    wake: &Receiver<()>,
    cache_mib: i64,
) {
    let mut conn: Option<Connection> = None;
    loop {
        match wake.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        }
        if !ready.load(std::sync::atomic::Ordering::Acquire) {
            continue;
        }
        if conn.is_none() {
            match crate::archive::open_read(index_db).and_then(|c| {
                c.pragma_update(None, "cache_size", -(cache_mib * 1024))?;
                Ok(c)
            }) {
                Ok(c) => conn = Some(c),
                Err(e) => {
                    crate::log!(warning, "search thread: open the index: {e}");
                    continue;
                }
            }
        }
        if let Some(c) = &conn {
            match drain(c, queue) {
                Ok(()) | Err(StoreError::WriterGone) => {}
                Err(e) => crate::log!(warning, "search thread: {e}"),
            }
        }
    }
}

/// Computes and sends batches until the queue reads empty.
fn drain(conn: &Connection, queue: &WriterQueue) -> Result<()> {
    loop {
        crate::fault::check("search.read_queue", "")?;
        let entries = crate::text_rows::pending_entries(conn, crate::text_rows::SEARCH_BYTES)?;
        if entries.is_empty() {
            return Ok(());
        }
        // A batch the writer settled before applying it writes nothing; the
        // next read no longer sees its queue rows.
        queue.ask(|reply| Msg::SearchWrite(Box::new(entries), reply))?;
    }
}
