//! The write lock: an OS file lock on the register (`archive.db`, or
//! `sinex.db` in primary mode). A reader holds it shared while one statement
//! starts its read, first on `index.db`, then on the register. A register
//! commit between those two starts is the only way the reader could see a
//! pair of states that never existed together; an index commit there is
//! harmless, because the register then reads as it was when the index read
//! started. So writers hold the lock exclusively only while the register
//! commits ([`commit`]) and through schema setup. Index transactions take no lock, so a
//! reader waits at most for one register commit. The file is stable while
//! open, so readers need no writable directory or extra lock file.
use crate::{Archive, Result, StoreError};
use rusqlite::Connection;
use std::cell::Cell;
use std::fs::{File, OpenOptions, TryLockError};
use std::os::unix::fs::MetadataExt;
use std::path::Path;

type FileIdentity = (u64, u64);
thread_local! {
    static HELD: Cell<Option<FileIdentity>> = const { Cell::new(None) };
}
fn identity(file: &File) -> std::io::Result<FileIdentity> {
    let metadata = file.metadata()?;
    Ok((metadata.dev(), metadata.ino()))
}

/// Held exclusively, as the module describes. Take it before SQLite's write
/// locks and drop it before file cleanup after the commit. A second lock on
/// the same thread refuses.
#[must_use]
pub(crate) struct WriteLock {
    file: Option<(FileIdentity, File)>,
}
impl Drop for WriteLock {
    fn drop(&mut self) {
        if let Some((key, _)) = &self.file {
            HELD.with(|held| {
                debug_assert_eq!(held.get(), Some(*key));
                held.set(None);
            });
        }
        // Closing the independently opened descriptor releases its OS flock.
    }
}

pub(crate) fn hold(conn: &Connection) -> Result<WriteLock> {
    let Some(path) = conn.path().filter(|path| !path.is_empty()) else {
        return Ok(WriteLock { file: None });
    };
    let path = Path::new(path);
    if path.file_name().and_then(|name| name.to_str()) == Some("index.db")
        && conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='primary_fact')",
            [],
            |row| row.get::<_, bool>(0),
        )?
    {
        return Ok(WriteLock { file: None });
    }
    let archive_db = match path.file_name().and_then(|name| name.to_str()) {
        Some("archive.db") => path.to_path_buf(),
        Some("index.db") => path.with_file_name("archive.db"),
        _ => {
            return Err(StoreError::Invalid(
                "the write lock needs a standalone archive or index connection".into(),
            ));
        }
    };
    let file = OpenOptions::new().read(true).open(archive_db)?;
    let key = identity(&file)?;
    if HELD.with(|held| held.get().is_some()) {
        return Err(StoreError::Invalid(
            "this thread already holds the write lock".into(),
        ));
    }
    file.lock()?;
    HELD.with(|held| held.set(Some(key)));
    Ok(WriteLock {
        file: Some((key, file)),
    })
}

fn probe_cpu() -> std::time::Duration {
    let text = std::fs::read_to_string("/proc/thread-self/schedstat").unwrap_or_default();
    std::time::Duration::from_nanos(
        text.split_whitespace()
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0),
    )
}
/// Commits a register transaction under the lock.
#[track_caller]
pub(crate) fn commit(tx: rusqlite::Transaction<'_>) -> Result<()> {
    let probe = std::time::Instant::now();
    let lock = hold(&tx)?;
    let probe_held = std::time::Instant::now();
    let probe_cpu0 = probe_cpu();
    let caller = std::panic::Location::caller();
    let db = tx
        .path()
        .map(|p| p.rsplit('/').next().unwrap_or("").to_owned())
        .unwrap_or_default();
    crate::probe_tmp::log("writelock.acquire", probe.elapsed());
    let committed = tx.commit();
    let probe_cpu1 = probe_cpu();
    drop(lock);
    crate::probe_tmp::log(
        &format!(
            "writelock.held_commit[{db} {}:{} cpu={:.1}ms]",
            caller.file().rsplit('/').next().unwrap_or(""),
            caller.line(),
            (probe_cpu1 - probe_cpu0).as_secs_f64() * 1000.0
        ),
        probe_held.elapsed(),
    );
    committed?;
    Ok(())
}

/// Takes the lock shared for a reader's first read; the caller drops the
/// returned file once its read transaction has started. `wait` runs the
/// reader's stop and interrupt checks while a writer holds the lock.
pub(crate) fn wait_shared(
    archive: &Archive,
    mut wait: impl FnMut() -> Result<()>,
) -> Result<Option<File>> {
    if archive.is_primary() {
        return Ok(None);
    }
    let file = OpenOptions::new().read(true).open(archive.archive_db())?;
    let key = identity(&file)?;
    if HELD.with(|held| held.get() == Some(key)) {
        return Err(StoreError::Invalid(
            "cannot start a read while this thread holds the write lock".into(),
        ));
    }
    loop {
        match file.try_lock_shared() {
            Ok(()) => return Ok(Some(file)),
            Err(TryLockError::WouldBlock) => wait()?,
            Err(TryLockError::Error(error)) if error.kind() == std::io::ErrorKind::Interrupted => {
                wait()?
            }
            Err(TryLockError::Error(error)) => return Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests;
