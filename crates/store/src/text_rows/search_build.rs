//! Native first-build search runs beside the row writer. Only committed text
//! snapshots cross the channel; temporary postings have no saved progress.
use crate::{Result, StoreError};
use polylogue_model::LockExt;
use rusqlite::Connection;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

mod install;
mod removed;
mod snapshot;
pub(crate) use removed::RemovedTexts;
mod worker;

const BATCH_TEXTS: usize = 1024;
const BATCH_BYTES: usize = 4 << 20;

struct Text {
    id: i64,
    bits: i64,
    present: i64,
    start: i64,
    length: i64,
    body: Option<String>,
}
struct Group {
    body: Vec<u8>,
    members: Vec<usize>,
}
#[derive(Default)]
struct TextBatch {
    texts: Vec<Text>,
    groups: Vec<Group>,
}

/// A disposable native FTS build. SQLite notices are reread between
/// transactions, so a rollback cannot publish an uncommitted text snapshot.
/// Restart seeds another builder from committed text.
pub(crate) struct SearchBuild {
    changed: Arc<Mutex<BTreeMap<i64, bool>>>,
    seeded: bool,
    sender: Option<mpsc::SyncSender<TextBatch>>,
    handle: Option<JoinHandle<Result<tempfile::TempPath>>>,
    ended: Option<Result<tempfile::TempPath>>,
    stop: crate::cancellation::Cancellation,
}
impl SearchBuild {
    pub(crate) fn path(index: &std::path::Path) -> std::path::PathBuf {
        let mut path = index.as_os_str().to_owned();
        path.push(".search");
        path.into()
    }

    pub(crate) fn start(
        conn: &Connection,
        stop: &crate::cancellation::Cancellation,
        cache_mib: i64,
    ) -> Result<Self> {
        let stop = stop.child();
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle = worker::start(conn, receiver, stop.clone(), cache_mib)?;
        let changed = Arc::new(Mutex::new(BTreeMap::new()));
        let notices = Arc::clone(&changed);
        // IDs name immutable contents. Hook calls survive rollback; capture
        // rereads the current row only after its transaction has ended.
        let build = Self {
            changed,
            seeded: false,
            sender: Some(sender),
            handle: Some(handle),
            ended: None,
            stop,
        };
        conn.update_hook(Some(
            move |action: rusqlite::hooks::Action, database: &str, table: &str, id: i64| {
                if database == "main" && table == "text" {
                    *notices.locked().entry(id).or_default() |=
                        action != rusqlite::hooks::Action::SQLITE_INSERT;
                }
            },
        ))?;
        Ok(build)
    }

    /// The worker never reads the private index or acquires the row writer's
    /// leases. One bounded batch may wait while another is being indexed.
    pub(crate) fn capture(
        &mut self,
        conn: &Connection,
        removed: Option<RemovedTexts>,
    ) -> Result<()> {
        if !conn.is_autocommit() {
            return Err(StoreError::Invalid(
                "search snapshots require committed rows".into(),
            ));
        }
        if self.sender.is_none() {
            // The worker ended early. Keep its exact result for finalization;
            // ordinary row turns must not repeatedly rebuild the same prefix.
            self.changed.locked().clear();
            return Ok(());
        }
        crate::timing::time(crate::timing::Stage::SearchIndex, || {
            // A successful model commit transfers ownership of its old bodies.
            // Rollback never hands over this disposable transaction spool.
            if let Some(removed) = removed {
                removed.send(|batch| self.send(batch))?;
            }
            if !self.seeded {
                let mut after = 0;
                loop {
                    let ids = conn
                        .prepare_cached(
                            "SELECT id FROM text WHERE id>?1 AND indexed<>0 ORDER BY id LIMIT ?2",
                        )?
                        .query_map([after, BATCH_TEXTS as i64], |row| row.get::<_, i64>(0))?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    let Some(last) = ids.last().copied() else {
                        break;
                    };
                    snapshot::send(conn, &ids, None, |batch| self.send(batch))?;
                    after = last;
                }
                self.seeded = true;
                // The seed covered these same rows. No model writes can run
                // concurrently on this connection while it is being captured.
                self.changed.locked().clear();
            }
            let changed = std::mem::take(&mut *self.changed.locked());
            let ids: Vec<_> = changed.keys().copied().collect();
            for ids in ids.chunks(BATCH_TEXTS) {
                snapshot::send(conn, ids, Some(&changed), |batch| self.send(batch))?;
            }
            Ok(())
        })
    }

    fn send(&mut self, batch: TextBatch) -> Result<()> {
        self.stop.check()?;
        if self
            .sender
            .as_ref()
            .ok_or_else(|| StoreError::Invalid("search build already ended".into()))?
            .send(batch)
            .is_err()
        {
            self.sender.take();
            self.ended = Some(self.join());
            let cause = self
                .ended
                .as_ref()
                .and_then(|result| result.as_ref().err())
                .map(ToString::to_string)
                .unwrap_or_else(|| "search build ended before its input".into());
            return Err(StoreError::Invalid(cause));
        }
        Ok(())
    }

    fn join(&mut self) -> Result<tempfile::TempPath> {
        if let Some(result) = self.ended.take() {
            return result;
        }
        self.handle
            .take()
            .ok_or_else(|| StoreError::Invalid("search build already ended".into()))?
            .join()
            .map_err(|_| StoreError::Invalid("search build worker panicked".into()))?
    }

    /// Drain, flush native postings and replace both search tables atomically.
    /// Readiness still waits for the surrounding private index's installation.
    pub(crate) fn finish(mut self, conn: &Connection) -> Result<()> {
        let result = (|| {
            self.capture(conn, None)?;
            self.sender.take();
            let side = self.join()?;
            crate::timing::time(crate::timing::Stage::SearchIndex, || {
                install::copy(conn, &side, &self.stop)
            })
        })();
        conn.update_hook(None::<fn(rusqlite::hooks::Action, &str, &str, i64)>)?;
        result
    }
}
impl Drop for SearchBuild {
    fn drop(&mut self) {
        self.stop.cancel();
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
