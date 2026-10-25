//! The write lock: an OS file lock on `archive.db`. Writers hold it
//! exclusively for each transaction that writes `archive.db` or `index.db`;
//! a reader holds it shared only while it starts its first read, so the two
//! databases it reads are a committed pair. The file is stable while open,
//! so readers need no writable directory or extra lock file.
use crate::{Archive, Result, StoreError};
use rusqlite::{Connection, TransactionBehavior};
use std::cell::Cell;
use std::fs::{File, OpenOptions, TryLockError};
use std::ops::Deref;
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

/// Held for paired transactions and DDL only. Take it before SQLite's write
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
    let archive_db = match path.file_name().and_then(|name| name.to_str()) {
        Some("archive.db") => path.to_path_buf(),
        Some("index.db") => path.with_file_name("archive.db"),
        _ => {
            return Err(StoreError::Invalid(
                "the write lock needs an archive or index connection".into(),
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

/// A transaction that holds the write lock. SQLite rolls back before the lock
/// is released, also on an error or early return. In-memory test databases
/// have no shared pair.
#[must_use]
pub(crate) struct Transaction<'a> {
    tx: rusqlite::Transaction<'a>,
    lock: WriteLock,
}
pub(crate) fn transaction(
    conn: &Connection,
    behavior: TransactionBehavior,
) -> Result<Transaction<'_>> {
    let lock = hold(conn)?;
    let tx = rusqlite::Transaction::new_unchecked(conn, behavior)?;
    Ok(Transaction { tx, lock })
}
impl<'a> Deref for Transaction<'a> {
    type Target = rusqlite::Transaction<'a>;
    fn deref(&self) -> &Self::Target {
        &self.tx
    }
}
impl Transaction<'_> {
    pub(crate) fn commit(self) -> Result<()> {
        let Self { tx, lock } = self;
        let result = tx.commit();
        drop(lock);
        Ok(result?)
    }
    pub(crate) fn rollback(self) -> Result<()> {
        let Self { tx, lock } = self;
        let result = tx.rollback();
        drop(lock);
        Ok(result?)
    }
}

/// Takes the lock shared for a reader's first read; the caller drops the
/// returned file once its read transaction has started. `wait` runs the
/// reader's stop and interrupt checks while a writer holds the lock.
pub(crate) fn wait_shared(archive: &Archive, mut wait: impl FnMut() -> Result<()>) -> Result<File> {
    let file = OpenOptions::new().read(true).open(archive.archive_db())?;
    let key = identity(&file)?;
    if HELD.with(|held| held.get() == Some(key)) {
        return Err(StoreError::Invalid(
            "cannot start a read while this thread holds the write lock".into(),
        ));
    }
    loop {
        match file.try_lock_shared() {
            Ok(()) => return Ok(file),
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
