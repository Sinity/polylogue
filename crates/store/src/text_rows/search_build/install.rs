//! One separate commit invalidates cached native FTS structures by data_version.
use crate::{ConnExt as _, Result, StoreError};
use rusqlite::{Connection, OpenFlags, TransactionBehavior};
use std::path::Path;

pub(super) fn copy(
    conn: &Connection,
    side: &Path,
    stop: &crate::cancellation::Cancellation,
) -> Result<()> {
    if !conn.is_autocommit() {
        return Err(StoreError::Invalid(
            "search installation requires committed rows".into(),
        ));
    }
    let path = conn
        .path()
        .filter(|path| !path.is_empty())
        .ok_or_else(|| StoreError::Invalid("search installation requires an index file".into()))?;
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
    let cache: i64 = conn.scalar("PRAGMA main.cache_size", [])?;
    copy.pragma_update(
        None,
        "cache_size",
        if cache < 0 {
            -(-cache / 8).max(1)
        } else {
            (cache / 8).max(1)
        },
    )?;
    copy.execute_batch("PRAGMA mmap_size=0; PRAGMA temp_store=FILE;")?;
    copy.set_db_config(
        rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
        true,
    )?;
    crate::read::limit::ReadLimit::observe_current(&copy)?;
    let interrupt = stop.clone();
    crate::cancellation::interrupt_on(&copy, move || interrupt.check())?;
    copy.execute(
        "ATTACH DATABASE ?1 AS search_build",
        [side.to_string_lossy()],
    )?;
    let result = (|| {
        let tx = rusqlite::Transaction::new_unchecked(&copy, TransactionBehavior::Immediate)?;
        for table in ["text_fts", "output_fts"] {
            let config = |database: &str| -> Result<Vec<(String, rusqlite::types::Value)>> {
                Ok(tx
                    .prepare(&format!(
                        "SELECT k,v FROM {database}.{table}_config ORDER BY k"
                    ))?
                    .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<rusqlite::Result<_>>()?)
            };
            if config("main")? != config("search_build")? {
                return Err(StoreError::Invalid(
                    "search build configuration differs from the index".into(),
                ));
            }
            for suffix in ["data", "idx", "docsize"] {
                tx.execute(&format!("DELETE FROM main.{table}_{suffix}"), [])?;
                tx.execute(&format!("INSERT INTO main.{table}_{suffix} SELECT * FROM search_build.{table}_{suffix}"), [])?;
            }
            crate::fault::check("search.copied_table", path)?;
        }
        crate::fault::check("search.before_commit", path)?;
        stop.check()?;
        crate::archive::bump_counter(&tx)?;
        tx.commit()?;
        Ok(())
    })();
    let detached = copy
        .execute("DETACH DATABASE search_build", [])
        .map(|_| ())
        .map_err(StoreError::from);
    result.and(detached)
}
