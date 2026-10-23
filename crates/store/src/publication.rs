//! First-read admission and bounded database publications share one OS lock.
//! The existing canonical file is stable while the archive is open, and a
//! reader needs no writable directory or new lock file.
use crate::{Archive, Result, StoreError};
use rusqlite::Connection;
use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

/// Keep this guard through the bounded SQLite transaction(s), starting before
/// any write lock. Never nest guards or retain one for capture/parse work.
/// Closing its independently opened descriptor releases its flock.
pub(crate) struct Publication {
    _file: Option<File>,
}

pub(crate) fn write(conn: &Connection) -> Result<Publication> {
    let Some(path) = conn.path().filter(|path| !path.is_empty()) else {
        // Isolated in-memory databases cannot be attached by a ReadPool.
        return Ok(Publication { _file: None });
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
    file.lock()?;
    Ok(Publication { _file: Some(file) })
}

/// No SQLite reads may happen between admission and the first pin. `wait`
/// supplies the pool's stop/interrupt checks without making this lock own a
/// request lifecycle. A successful guard lasts only through snapshot pinning.
pub(crate) fn admit(archive: &Archive, mut wait: impl FnMut() -> Result<()>) -> Result<File> {
    let file = OpenOptions::new().read(true).open(archive.archive_db())?;
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
