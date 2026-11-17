//! Search tables are built on disk beside the private index, then copied in
//! together. Their page caches and queued text use fixed shares of the writer's
//! allowance; the tables themselves never live in process memory.
use super::{insert_entry, joined};
use crate::{ConnExt as _, InFlight, Result, StoreError};
use polylogue_model::LockExt;
use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use std::path::Path;
use std::sync::{
    Arc, Condvar, Mutex,
    mpsc::{SyncSender, sync_channel},
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const QUEUED_BYTES: usize = 64 << 20;

#[derive(Default)]
struct SearchBytes {
    used: Mutex<usize>,
    available: Condvar,
}
impl SearchBytes {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<InFlight> {
        let mut used = self.used.locked();
        // One text larger than the allowance is processed alone.
        while *used != 0 && used.saturating_add(bytes) > QUEUED_BYTES {
            crate::cancellation::check_current()?;
            used = self
                .available
                .wait(used)
                .unwrap_or_else(|error| error.into_inner());
        }
        crate::cancellation::check_current()?;
        *used += bytes;
        let shared = Arc::clone(self);
        Ok(InFlight::new(move || {
            *shared.used.locked() -= bytes;
            shared.available.notify_all();
        }))
    }
}

struct SearchText {
    id: i64,
    text: Arc<String>,
    _bytes: InFlight,
}
struct SearchDatabase {
    bit: i64,
    table: &'static str,
    path: tempfile::TempPath,
}
struct SearchBuild {
    sender: Option<SyncSender<SearchText>>,
    handle: Option<JoinHandle<Result<SearchDatabase>>>,
}
impl SearchBuild {
    fn start(conn: &Connection, bit: i64, table: &'static str) -> Result<Self> {
        let file = match conn.path().and_then(|path| Path::new(path).parent()) {
            Some(parent) if !parent.as_os_str().is_empty() => {
                tempfile::NamedTempFile::new_in(parent)?
            }
            _ => tempfile::NamedTempFile::new()?,
        };
        let path = file.into_temp_path();
        let sql = crate::schema::search_index_sql(table)?;
        let cache: i64 = conn.scalar("PRAGMA main.cache_size", [])?;
        let page: i64 = conn.scalar("PRAGMA main.page_size", [])?;
        let cache_kib = if cache < 0 {
            -cache
        } else {
            cache * page / 1024
        };
        let (sender, receiver) = sync_channel::<SearchText>(64);
        let stop = crate::cancellation::thread_stop();
        let name = if bit == 1 {
            "polylogue-search-prose"
        } else {
            "polylogue-search-output"
        };
        let handle = polylogue_model::spawn_named(name, move || {
            let run = || {
                let side = Connection::open(&path)?;
                side.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;")?;
                side.pragma_update(None, "cache_size", -(cache_kib / 8).max(1))?;
                if let Some(stop) = crate::cancellation::thread_stop() {
                    crate::cancellation::interrupt_on(&side, move || stop.check())?;
                }
                for statement in sql {
                    side.execute_batch(&statement)?;
                }
                let tx =
                    rusqlite::Transaction::new_unchecked(&side, TransactionBehavior::Immediate)?;
                for text in receiver {
                    crate::cancellation::check_current()?;
                    if text.text.is_ascii() {
                        insert_entry(&tx, text.id, bit, text.text.as_str())?;
                    } else {
                        let folded = polylogue_model::text::fold(text.text.as_str());
                        insert_entry(&tx, text.id, bit, &folded)?;
                    }
                }
                tx.commit()?;
                drop(side);
                Ok(SearchDatabase { bit, table, path })
            };
            match stop {
                Some(stop) => crate::cancellation::under(&stop, run),
                None => run(),
            }
        })?;
        Ok(Self {
            sender: Some(sender),
            handle: Some(handle),
        })
    }

    fn finish(mut self) -> Result<SearchDatabase> {
        self.sender.take();
        self.handle
            .take()
            .expect("search worker")
            .join()
            .map_err(|_| StoreError::Invalid("search worker panicked".into()))?
    }
}
impl Drop for SearchBuild {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Builds both search tables beside the index while `meanwhile` writes its
/// other indexes. Search replacement is one transaction on the main index.
pub(crate) fn build_search_indexes(
    conn: &Connection,
    meanwhile: impl FnOnce(&Connection) -> Result<()>,
) -> Result<()> {
    let started = Instant::now();
    let mut other_work = Duration::ZERO;
    let result = (|| {
        let bytes = Arc::new(SearchBytes::default());
        let stop = crate::cancellation::thread_stop();
        let _wake = stop.map(|stop| {
            let bytes = Arc::clone(&bytes);
            stop.on_cancel(move || {
                let _used = bytes.used.locked();
                bytes.available.notify_all();
            })
        });
        let mut builds = vec![
            SearchBuild::start(conn, 1, "text_fts")?,
            SearchBuild::start(conn, 2, "output_fts")?,
        ];
        let mut written = false;
        let prepared = (|| {
            let mut select = conn.prepare(
                "SELECT t.id, t.indexed, g.body, t.start, t.length FROM text t
                 JOIN text_group g ON g.id = t.group_id WHERE t.indexed <> 0 ORDER BY t.id",
            )?;
            let mut rows = select.query([])?;
            while let Some(row) = rows.next()? {
                crate::cancellation::check_current()?;
                let id: i64 = row.get(0)?;
                let indexed: i64 = row.get(1)?;
                let text = Arc::new(joined(row, 2)?);
                for (n, bit) in [1, 2].into_iter().enumerate() {
                    if indexed & bit != 0 {
                        let guard = bytes.reserve(text.capacity())?;
                        builds[n]
                            .sender
                            .as_ref()
                            .expect("search sender")
                            .send(SearchText {
                                id,
                                text: Arc::clone(&text),
                                _bytes: guard,
                            })
                            .map_err(|_| {
                                StoreError::Invalid("search worker ended before its input".into())
                            })?;
                    }
                }
                written = true;
                crate::fault::check("search.indexed", conn.path().unwrap_or_default())?;
            }
            drop(rows);
            drop(select);
            for build in &mut builds {
                build.sender.take();
            }
            let started = Instant::now();
            let result = meanwhile(conn);
            other_work = started.elapsed();
            result
        })();
        for build in &mut builds {
            build.sender.take();
        }
        let mut sides = Vec::new();
        let mut failed = None;
        for build in builds {
            match build.finish() {
                Ok(side) => sides.push(side),
                Err(error) if failed.is_none() => failed = Some(error),
                Err(_) => {}
            }
        }
        if let Some(error) = failed {
            return Err(error);
        }
        prepared?;
        install(conn, &sides, written)
    })();
    // This is the foreground search work: scan, final worker wait and copy.
    // Work done by `meanwhile` has its own stage timers.
    crate::timing::add(
        crate::timing::Stage::SearchIndex,
        started.elapsed().saturating_sub(other_work),
    );
    result
}

fn install(conn: &Connection, sides: &[SearchDatabase], written: bool) -> Result<()> {
    if !conn.is_autocommit() {
        return Err(StoreError::Invalid(
            "search installation requires the writer outside a transaction".into(),
        ));
    }
    let path = conn
        .path()
        .filter(|path| !path.is_empty())
        .ok_or_else(|| StoreError::Invalid("search installation requires the index file".into()))?;
    // FTS5 caches its segment structure by SQLite's data_version. Shadow writes
    // on the same connection do not change that version. A separate commit lets
    // the existing writer and readers discard their cached structures normally.
    let copy = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let journal: String = conn.scalar("PRAGMA main.journal_mode", [])?;
    copy.pragma_update(Some("main"), "journal_mode", journal)?;
    for pragma in ["synchronous", "busy_timeout", "wal_autocheckpoint"] {
        let value: i64 = conn.scalar(&format!("PRAGMA {pragma}"), [])?;
        copy.pragma_update(None, pragma, value)?;
    }
    // Both side writers have closed. Reuse one of their page-cache shares for
    // the final copy, without retaining another full writer-sized cache.
    let cache: i64 = conn.scalar("PRAGMA main.cache_size", [])?;
    let cache = if cache < 0 {
        -(-cache / 8).max(1)
    } else {
        (cache / 8).max(1)
    };
    copy.pragma_update(None, "cache_size", cache)?;
    copy.execute_batch("PRAGMA mmap_size=0; PRAGMA temp_store=FILE;")?;
    copy.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        true,
    )?;
    crate::read::limit::ReadLimit::observe_current(&copy)?;
    let conn = &copy;
    let mut attached = Vec::new();
    let result = (|| {
        for side in sides {
            let alias = if side.bit == 1 {
                "search_prose"
            } else {
                "search_output"
            };
            conn.execute(
                &format!("ATTACH DATABASE ?1 AS {alias}"),
                [side.path.to_string_lossy()],
            )?;
            attached.push(alias);
        }
        let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        for (side, alias) in sides.iter().zip(&attached) {
            let config = |database: &str| -> Result<Vec<(String, rusqlite::types::Value)>> {
                Ok(tx
                    .prepare(&format!(
                        "SELECT k,v FROM {database}.{}_config ORDER BY k",
                        side.table
                    ))?
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?)
            };
            if config("main")? != config(alias)? {
                return Err(StoreError::Invalid(
                    "side search configuration differs from the index".into(),
                ));
            }
        }
        for (side, alias) in sides.iter().zip(&attached) {
            let table = side.table;
            for suffix in ["data", "idx", "docsize"] {
                tx.execute(&format!("DELETE FROM main.{table}_{suffix}"), [])?;
                tx.execute(
                    &format!(
                        "INSERT INTO main.{table}_{suffix} SELECT * FROM {alias}.{table}_{suffix}"
                    ),
                    [],
                )?;
            }
            crate::fault::check("search.copied_table", conn.path().unwrap_or_default())?;
        }
        crate::fault::check("search.indexed", conn.path().unwrap_or_default())?;
        crate::cancellation::check_current()?;
        if written {
            crate::archive::bump_counter(&tx)?;
        }
        tx.commit()?;
        Ok(())
    })();
    let mut detached = Ok(());
    for alias in attached {
        if let Err(error) = conn.execute(&format!("DETACH DATABASE {alias}"), []) {
            detached = Err(error.into());
        }
    }
    result.and(detached)
}
