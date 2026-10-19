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
            && let Err(e) = write_pending_search(c, SEARCH_BATCH)
        {
            eprintln!("polylogue: search thread: {e}");
        }
    }
}

/// Search entries written per transaction by the search thread.
pub(super) const SEARCH_BATCH: i64 = 2_000;

/// Writes waiting search entries, at most `limit` per transaction, until
/// none wait; returns how many it wrote.
pub(super) fn write_pending_search(conn: &Connection, limit: i64) -> Result<u64> {
    let mut written = 0;
    loop {
        // Immediate: it writes, and a deferred transaction that has read
        // cannot wait for the writer's lock to upgrade.
        let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        let rows: Vec<(i64, i64, Vec<u8>)> = tx
            .prepare_cached(
                "SELECT p.text_id, p.fts_table, t.body FROM search_pending p JOIN text t ON t.id = p.text_id
                 ORDER BY p.text_id, p.fts_table LIMIT ?1",
            )?
            .query_map([limit], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        if rows.is_empty() {
            // Entries whose text was removed meanwhile (excision) are dropped.
            tx.execute(
                "DELETE FROM search_pending WHERE text_id NOT IN (SELECT id FROM text)",
                [],
            )?;
            tx.commit()?;
            return Ok(written);
        }
        for (id, bit, packed) in &rows {
            let folded = polylogue_model::text::fold(&unpack_text(packed)?);
            let sql = if *bit == 1 {
                "INSERT INTO text_fts(rowid, body) VALUES (?1, ?2)"
            } else {
                "INSERT INTO output_fts(rowid, body) VALUES (?1, ?2)"
            };
            tx.prepare_cached(sql)?.execute(params![id, folded])?;
            tx.prepare_cached("DELETE FROM search_pending WHERE text_id = ?1 AND fts_table = ?2")?
                .execute(params![id, bit])?;
        }
        written += rows.len() as u64;
        tx.commit()?;
    }
}
