//! Disposal of this archive's obsolete derived indexes, never source archives.
use super::{Archive, open_read};
use crate::{Result, StoreError};
use rusqlite::{Connection, OpenFlags};
use std::{collections::BTreeSet, fs, path::PathBuf};

pub(crate) fn validate(archive: &Archive, canonical: &Connection) -> Result<()> {
    checked(archive, canonical).map(|_| ())
}

pub(crate) fn remove(archive: &Archive, canonical: &Connection) -> Result<()> {
    let files = checked(archive, canonical)?;
    // Consolidate WAL state before any unlink. On interruption the remaining
    // database still holds all the removal receipts checked above.
    for path in &files {
        checkpoint(path)?;
    }
    for path in files {
        for suffix in ["-wal", "-shm"] {
            let sidecar = path.with_file_name(format!(
                "{}{suffix}",
                path.file_name().unwrap().to_string_lossy()
            ));
            match fs::remove_file(sidecar) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        fs::File::open(&archive.root)?.sync_all()?;
        crate::fault::check("index_cleanup.after_sidecars", &path.to_string_lossy())?;
        fs::remove_file(&path)?;
        crate::fault::check("index_cleanup.after_unlink", &path.to_string_lossy())?;
        fs::File::open(&archive.root)?.sync_all()?;
    }
    // Also sync an empty retry after a previous unlink failed before its sync.
    fs::File::open(&archive.root)?.sync_all()?;
    Ok(())
}

pub(super) fn checkpoint(path: &std::path::Path) -> Result<()> {
    let db = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    db.busy_timeout(std::time::Duration::ZERO)?;
    let busy: i64 = db.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
    if busy != 0 {
        return Err(StoreError::Invalid(format!(
            "obsolete index is still in use: {}",
            path.display()
        )));
    }
    Ok(())
}

fn checked(archive: &Archive, canonical: &Connection) -> Result<Vec<PathBuf>> {
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(&archive.root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let base = name
            .strip_suffix("-wal")
            .or_else(|| name.strip_suffix("-shm"))
            .unwrap_or(name);
        if !owned_name(base) {
            continue;
        }
        if !entry.file_type()?.is_file() {
            return Err(StoreError::Invalid(format!(
                "obsolete index is not a regular file: {name}"
            )));
        }
        files.insert(archive.root.join(base));
    }
    for path in &files {
        // A sidecar alone cannot prove the removal history of its missing DB.
        if !path.is_file() {
            return Err(StoreError::Invalid(format!(
                "obsolete index sidecar has no database: {}",
                path.display()
            )));
        }
        let db = open_read(path)?;
        let known: bool = db.query_row("SELECT count(*)=3 FROM sqlite_schema WHERE type='table' AND name IN ('meta','input_ref','message')", [], |r| r.get(0))?;
        if !known {
            return Err(StoreError::Invalid(format!(
                "unrecognized obsolete index: {}",
                path.display()
            )));
        }
        crate::excise::recovery::needs_rebuild(canonical, &db)?;
    }
    Ok(files.into_iter().collect())
}

fn owned_name(name: &str) -> bool {
    name == "index.db.unsynced"
        || name
            .strip_prefix("index.db.v")
            .is_some_and(|v| v.parse::<u32>().is_ok_and(|n| n.to_string() == v))
        || name
            .strip_prefix("index.db.excision-")
            .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok_and(|u| u.to_string() == id))
}

#[cfg(test)]
mod tests;
