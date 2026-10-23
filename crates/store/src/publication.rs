//! First-read admission and bounded database publications share one OS lock.
//! The canonical file is stable while open; readers need no writable directory
//! or additional lock file. A transaction releases exclusion on commit/rollback.
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

/// For paired transactions and DDL only. Acquire before SQLite write locks and
/// release before postcommit physical cleanup. Nested guards refuse.
#[must_use]
pub(crate) struct Publication {
    file: Option<(FileIdentity, File)>,
}
impl Drop for Publication {
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

pub(crate) fn write(conn: &Connection) -> Result<Publication> {
    let Some(path) = conn.path().filter(|path| !path.is_empty()) else {
        return Ok(Publication { file: None });
    };
    let path = Path::new(path);
    let canonical = match path.file_name().and_then(|name| name.to_str()) {
        Some("archive.db") => path.to_path_buf(),
        Some("index.db") => path.with_file_name("archive.db"),
        _ => {
            return Err(StoreError::Invalid(
                "publication requires an archive or index connection".into(),
            ));
        }
    };
    let file = OpenOptions::new().read(true).open(canonical)?;
    let key = identity(&file)?;
    if HELD.with(|held| held.get().is_some()) {
        return Err(StoreError::Invalid(
            "a publication is already active on this thread".into(),
        ));
    }
    file.lock()?;
    HELD.with(|held| held.set(Some(key)));
    Ok(Publication {
        file: Some((key, file)),
    })
}

/// SQLite rollback happens before its publication guard is released, including
/// on error or an early return. In-memory test databases have no shared pair.
#[must_use]
pub(crate) struct Transaction<'a> {
    tx: rusqlite::Transaction<'a>,
    publication: Publication,
}
pub(crate) fn transaction(
    conn: &Connection,
    behavior: TransactionBehavior,
) -> Result<Transaction<'_>> {
    let publication = write(conn)?;
    let tx = rusqlite::Transaction::new_unchecked(conn, behavior)?;
    Ok(Transaction { tx, publication })
}
impl<'a> Deref for Transaction<'a> {
    type Target = rusqlite::Transaction<'a>;
    fn deref(&self) -> &Self::Target {
        &self.tx
    }
}
impl Transaction<'_> {
    pub(crate) fn commit(self) -> Result<()> {
        let Self { tx, publication } = self;
        let result = tx.commit();
        drop(publication);
        Ok(result?)
    }
    pub(crate) fn rollback(self) -> Result<()> {
        let Self { tx, publication } = self;
        let result = tx.rollback();
        drop(publication);
        Ok(result?)
    }
}

/// The successful guard lasts through first pinning, never the query lifetime.
/// `wait` supplies the reader's stop/interrupt checks while another publisher runs.
pub(crate) fn admit(archive: &Archive, mut wait: impl FnMut() -> Result<()>) -> Result<File> {
    let file = OpenOptions::new().read(true).open(archive.archive_db())?;
    let key = identity(&file)?;
    if HELD.with(|held| held.get() == Some(key)) {
        return Err(StoreError::Invalid(
            "cannot start a read snapshot during publication".into(),
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
