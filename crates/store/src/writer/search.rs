//! The search thread: it writes the full-text entries the writer queued
//! in `search_pending`, in its own transactions, shortly after their rows
//! commit (architecture 5.5c).

use super::*;

/// The search thread's handle: it writes the entries in `search_pending`
/// shortly after the writer commits them (architecture 5.5c), in its own
/// transactions on its own connection, so search indexing is off the
/// writer's thread.
pub(super) struct Search {
    wake: Option<SyncSender<()>>,
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Search {
    pub(super) fn start(index_db: std::path::PathBuf, ready: bool) -> Result<Self> {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(ready));
        let (wake, rx) = sync_channel::<()>(1);
        let seen = flag.clone();
        let handle = std::thread::Builder::new()
            .name("polylogue-search".into())
            .spawn(move || search_thread(&index_db, &seen, &rx))?;
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
}

impl Drop for Search {
    fn drop(&mut self) {
        self.wake.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Writes waiting search entries whenever woken, and once a second. The
/// connection opens only after a first build, whose index is not in WAL
/// mode and has one connection.
pub(super) fn search_thread(
    index_db: &std::path::Path,
    ready: &std::sync::atomic::AtomicBool,
    wake: &Receiver<()>,
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
            match open_write(index_db, crate::archive::cache::SEARCH) {
                Ok(c) => conn = Some(c),
                Err(e) => {
                    eprintln!("polylogue: search thread: open the index: {e}");
                    continue;
                }
            }
        }
        if let Some(c) = &conn
            && let Err(e) = crate::text_rows::index_pending(c)
        {
            eprintln!("polylogue: search thread: {e}");
        }
    }
}
