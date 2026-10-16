//! The one-shot owner recovers a verified, separately owned copy only.
use crate::{Error, ReadHandle, Result, db, native, open, owned_metadata, sibling, wal};
use polylogue_model::{
    BlobHash, DurableTier, HeadCoverage, RecoveredWal, RecoveryQualification, RecoveryReason,
};
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Component, Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct Entry {
    path: String,
    length: u64,
    blake3: BlobHash,
    device: u64,
    inode: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}
fn refused(reason: impl Into<String>) -> Error {
    Error::RecoveryRefused(reason.into())
}
fn regular(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(refused(format!("not a regular file: {}", path.display())));
    }
    Ok(file)
}
fn digest(path: &Path, cancelled: &AtomicBool) -> Result<Entry> {
    let mut file = regular(path)?;
    let before = file.metadata()?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = [0; 65536];
    loop {
        wal::check_cancel(cancelled)?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    let stamp = |m: &fs::Metadata| {
        (
            m.dev(),
            m.ino(),
            m.len(),
            m.mtime(),
            m.mtime_nsec(),
            m.ctime(),
            m.ctime_nsec(),
        )
    };
    if stamp(&before) != stamp(&file.metadata()?)
        || stamp(&before) != stamp(&fs::symlink_metadata(path)?)
    {
        return Err(refused(format!("source changed: {}", path.display())));
    }
    Ok(Entry {
        path: String::new(),
        length: before.len(),
        blake3: BlobHash::from_bytes(*hash.finalize().as_bytes()),
        device: before.dev(),
        inode: before.ino(),
        modified_seconds: before.mtime(),
        modified_nanoseconds: before.mtime_nsec(),
        changed_seconds: before.ctime(),
        changed_nanoseconds: before.ctime_nsec(),
    })
}
fn inventory(root: &Path, cancelled: &AtomicBool) -> Result<Vec<Entry>> {
    fn walk(
        root: &Path,
        dir: &Path,
        depth: usize,
        out: &mut Vec<Entry>,
        cancelled: &AtomicBool,
    ) -> Result<()> {
        wal::check_cancel(cancelled)?;
        if depth > 32 {
            return Err(refused(
                "archive directory nesting exceeds recovery admission",
            ));
        }
        let mut children = fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
        children.sort_by_key(|e| e.file_name());
        for child in children {
            let path = child.path();
            let meta = fs::symlink_metadata(&path)?;
            if meta.is_dir() {
                walk(root, &path, depth + 1, out, cancelled)?;
            } else if meta.is_file() {
                let mut entry = digest(&path, cancelled)?;
                entry.path = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .ok_or_else(|| refused("recovery requires UTF-8 relative paths"))?
                    .to_owned();
                out.push(entry);
            } else {
                return Err(refused(format!("unsafe file type: {}", path.display())));
            }
        }
        Ok(())
    }
    let mut entries = vec![];
    walk(root, root, 0, &mut entries, cancelled)?;
    Ok(entries)
}
fn copy_file(
    source: &Path,
    destination: &Path,
    expected: &Entry,
    cancelled: &AtomicBool,
) -> Result<()> {
    let mut input = regular(source)?;
    if let Some(parent) = destination.parent() {
        crate::private_dir(parent)?;
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(destination)?;
    if input.metadata()?.dev() == output.metadata()?.dev()
        && input.metadata()?.ino() == output.metadata()?.ino()
    {
        return Err(refused("source and destination share an inode"));
    }
    let mut hash = blake3::Hasher::new();
    let mut length = 0;
    let mut buffer = [0; 65536];
    loop {
        wal::check_cancel(cancelled)?;
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        output.write_all(&buffer[..n])?;
        hash.update(&buffer[..n]);
        length += n as u64;
    }
    output.sync_all()?;
    if length != expected.length || hash.finalize().as_bytes() != expected.blake3.bytes() {
        return Err(refused(
            "copy digest differs from the preserved source manifest",
        ));
    }
    let actual = digest(destination, cancelled)?;
    if actual.blake3 != expected.blake3 || actual.length != expected.length {
        return Err(refused("destination readback mismatch"));
    }
    Ok(())
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn json_bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| refused(e.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}
fn sync_tree(root: &Path) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        let meta = fs::symlink_metadata(&path)?;
        if meta.is_dir() {
            sync_tree(&path)?;
        } else if meta.is_file() {
            File::open(&path)?.sync_all()?;
        } else {
            return Err(refused("unsafe staging file type"));
        }
    }
    File::open(root)?.sync_all()?;
    Ok(())
}
pub(crate) fn qualification(
    conn: &Connection,
    path: &Path,
) -> Result<Option<RecoveryQualification>> {
    let value: Option<Option<String>> = db(path, || {
        conn.query_row("SELECT CASE WHEN length(CAST(value AS BLOB))<=65536 THEN value END FROM meta WHERE key='recovery_qualification'",[],|r|r.get(0)).optional()
    })?;
    value
        .map(|value| {
            let value = value.ok_or_else(|| Error::Schema(path.into()))?;
            let qualification: RecoveryQualification =
                serde_json::from_str(&value).map_err(|_| Error::Schema(path.into()))?;
            if !qualification.valid() {
                return Err(Error::Schema(path.into()));
            }
            Ok(qualification)
        })
        .transpose()
}
pub(crate) fn raw_versions(root: &Path) -> Result<Option<u32>> {
    let mut archive = None;
    for (name, supported) in [
        ("archive.db", crate::ARCHIVE_VERSION),
        ("embeddings.db", crate::EMBEDDINGS_VERSION),
    ] {
        if owned_metadata(&root.join(name))?.is_some_and(|m| m.len() > 0) {
            let found = wal::schema_version(&root.join(name))?;
            if found > supported {
                return Err(Error::NewerSchema {
                    file: root.join(name),
                    found,
                    supported,
                });
            }
            if name == "archive.db" {
                archive = Some(found);
            }
        }
    }
    Ok(archive)
}

/// One concrete v1 -> v2 migration. Preserve the quiescent sibling set before
/// even a read-only SQLite open can rebuild shared memory. Each attempt has its
/// own completed backup; a failed previous attempt never substitutes older bytes.
pub(crate) fn migrate(root: &Path, cancelled: &AtomicBool) -> Result<()> {
    if raw_versions(root)? != Some(1) {
        return Ok(());
    }
    let backups = root.join("migrations");
    crate::private_dir(&backups)?;
    let backup = tempfile::Builder::new()
        .prefix("v1-to-v2-")
        .tempdir_in(&backups)?;
    let mut entries = vec![];
    for name in ["archive.db", "index.db", "embeddings.db"] {
        for suffix in ["", "-wal", "-shm"] {
            let name = format!("{name}{suffix}");
            let source = root.join(&name);
            if owned_metadata(&source)?.is_some() {
                let mut entry = digest(&source, cancelled)?;
                entry.path = name.clone();
                copy_file(&source, &backup.path().join(&name), &entry, cancelled)?;
                entries.push(entry);
            }
        }
    }
    write_new(
        &backup.path().join("source-manifest.json"),
        &json_bytes(&entries)?,
    )?;
    for entry in &entries {
        let mut current = digest(&root.join(&entry.path), cancelled)?;
        current.path = entry.path.clone();
        if current != *entry {
            return Err(refused(
                "source changed while preserving migration siblings",
            ));
        }
    }
    sync_tree(backup.path())?;
    let _preserved = backup.keep();
    File::open(&backups)?.sync_all()?;
    crate::preflight_inner(root, true)?;
    wal::check_cancel(cancelled)?;
    let path = root.join("archive.db");
    let mut conn = open(&path, false, true)?;
    set_format(&mut conn, &path, None)?;
    Ok(())
}
fn set_format(
    conn: &mut Connection,
    path: &Path,
    recovery: Option<&RecoveryQualification>,
) -> Result<()> {
    let next = crate::counter(conn, path)?
        .checked_add(1)
        .filter(|n| *n <= i64::MAX as u64)
        .ok_or(Error::Range)?;
    db(path, || {
        conn.pragma_update(None, "synchronous", "FULL")?;
        let tx = conn.transaction()?;
        if let Some(recovery) = recovery {
            let text = serde_json::to_string(recovery).expect("typed recovery JSON");
            tx.execute("INSERT INTO meta(key,value) VALUES('recovery_qualification',?1) ON CONFLICT(key) DO UPDATE SET value=excluded.value",[text])?;
        }
        tx.execute(
            "UPDATE meta SET value=?1 WHERE key='write_counter'",
            [next.to_string()],
        )?;
        tx.pragma_update(None, "user_version", crate::ARCHIVE_VERSION)?;
        tx.commit()
    })
}
fn integrity(conn: &Connection, path: &Path, cancelled: &AtomicBool) -> Result<()> {
    wal::check_cancel(cancelled)?;
    let _progress = native::cancellation(conn, cancelled);
    let result = db(path, || {
        let mut statement = conn.prepare("PRAGMA integrity_check")?;
        let mut rows = statement.query([])?;
        let row = rows.next()?.ok_or(rusqlite::Error::InvalidQuery)?;
        if row.get::<_, String>(0)? != "ok" || rows.next()?.is_some() {
            return Err(rusqlite::Error::InvalidQuery);
        }
        if conn
            .prepare("PRAGMA foreign_key_check")?
            .query([])?
            .next()?
            .is_some()
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
        Ok(())
    });
    wal::check_cancel(cancelled)?;
    result
}
fn blobs(reader: &ReadHandle, cancelled: &AtomicBool) -> Result<(u64, u64)> {
    let path = reader.root.join("archive.db");
    let mut count = 0;
    let mut bytes = 0;
    let mut imports = db(&path, || {
        reader.archive.prepare("SELECT i.id,i.format,r.length,r.blake3,i.original_hash,i.context_json FROM input i JOIN revision r ON r.input=i.id AND r.number=1 WHERE i.original_hash IS NOT NULL")
    })?;
    let mut imported = db(&path, || imports.query([]))?;
    while let Some(row) = db(&path, || imported.next())? {
        wal::check_cancel(cancelled)?;
        let (id, format, length, hash, original, context) = db(&path, || {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
            ))
        })?;
        crate::retained::decode_import_identity(id, format, length, hash, original, context)?;
    }
    let mut stmt = db(&path, || {
        reader
            .archive
            .prepare("SELECT hash,size,place,encoding,state FROM blob ORDER BY hash")
    })?;
    let mut rows = db(&path, || stmt.query([]))?;
    while let Some(row) = db(&path, || rows.next())? {
        wal::check_cancel(cancelled)?;
        let (hash, size, place, encoding, state): (Vec<u8>, i64, String, String, String) =
            db(&path, || {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })?;
        if place != "file" || encoding != "plain" || state != "present" {
            return Err(refused("recovery only supports present plain file blobs"));
        }
        let hash = BlobHash::from_bytes(hash.try_into().map_err(|_| Error::Schema(path.clone()))?);
        let size = u64::try_from(size).map_err(|_| Error::Schema(path.clone()))?;
        let mut actual = blake3::Hasher::new();
        let mut offset = 0;
        while offset < size {
            wal::check_cancel(cancelled)?;
            let data = reader.read_blob_range(hash, offset, (size - offset).min(65536))?;
            actual.update(&data);
            offset += data.len() as u64;
        }
        if actual.finalize().as_bytes() != hash.bytes() {
            return Err(refused(format!("blob content digest mismatch: {hash}")));
        }
        count += 1;
        bytes += size;
    }
    // Retained revisions, not only physical blobs, own cited chain byte ranges.
    let mut stmt = db(&path, || {
        reader
            .archive
            .prepare("SELECT input,chain,length,blake3 FROM revision ORDER BY input,number")
    })?;
    let mut rows = db(&path, || stmt.query([]))?;
    while let Some(row) = db(&path, || rows.next())? {
        let (input, chain, length, expected): (String, i64, i64, Option<Vec<u8>>) =
            db(&path, || {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
        let length = u64::try_from(length).map_err(|_| Error::Schema(path.clone()))?;
        let mut extents = db(&path, || {
            reader.archive.prepare("SELECT offset,length,blob,blob_offset FROM extent WHERE input=?1 AND chain=?2 AND offset<?3 ORDER BY offset")
        })?;
        let mut parts = db(&path, || {
            extents.query(rusqlite::params![
                input,
                chain,
                i64::try_from(length).map_err(|_| rusqlite::Error::InvalidQuery)?
            ])
        })?;
        let mut offset = 0;
        let mut hash = blake3::Hasher::new();
        while let Some(part) = db(&path, || parts.next())? {
            let (start, len, blob, blob_offset): (i64, i64, Vec<u8>, i64) = db(&path, || {
                Ok((part.get(0)?, part.get(1)?, part.get(2)?, part.get(3)?))
            })?;
            if start < 0 || start as u64 != offset || len <= 0 || blob_offset < 0 {
                return Err(refused("inconsistent revision extents"));
            }
            let blob =
                BlobHash::from_bytes(blob.try_into().map_err(|_| Error::Schema(path.clone()))?);
            let end = length.min(offset.checked_add(len as u64).ok_or(Error::Range)?);
            let mut local = 0;
            while offset < end {
                wal::check_cancel(cancelled)?;
                let data = reader.read_blob_range(
                    blob,
                    (blob_offset as u64)
                        .checked_add(local)
                        .ok_or(Error::Range)?,
                    (end - offset).min(65536),
                )?;
                hash.update(&data);
                offset += data.len() as u64;
                local += data.len() as u64;
            }
        }
        if offset != length
            || expected.is_some_and(|expected| expected.as_slice() != hash.finalize().as_bytes())
        {
            return Err(refused("retained revision content digest mismatch"));
        }
    }
    Ok((count, bytes))
}

#[derive(Debug, Serialize)]
pub struct RecoveryReport {
    pub format: &'static str,
    pub version: u32,
    pub destination: PathBuf,
    pub recovery: RecoveryQualification,
    pub receipt: PathBuf,
    pub checked_blobs: u64,
    pub checked_blob_bytes: u64,
    pub checks: Vec<&'static str>,
}
/// Called only by the one-shot process owner. No SQLite capability escapes.
pub fn recover_copy(
    source: &Path,
    destination: &Path,
    cancelled: &AtomicBool,
) -> Result<RecoveryReport> {
    recover_inner(source, destination, cancelled, &mut |_| Ok(()))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    Copied,
    Qualified,
    BeforePublish,
    Published,
}
fn recover_inner(
    source: &Path,
    destination: &Path,
    cancelled: &AtomicBool,
    boundary: &mut impl FnMut(Boundary) -> Result<()>,
) -> Result<RecoveryReport> {
    wal::check_cancel(cancelled)?;
    if !fs::symlink_metadata(source)?.is_dir() {
        return Err(refused("source must be a real directory"));
    }
    let source = fs::canonicalize(source)?;
    let parent = fs::canonicalize(
        destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    let leaf = destination
        .file_name()
        .ok_or_else(|| refused("destination needs a new directory name"))?;
    if destination
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(refused("parent traversal in destination"));
    }
    let destination = parent.join(leaf);
    if destination.starts_with(&source) || source.starts_with(&destination) {
        return Err(refused("source and destination overlap"));
    }
    if fs::symlink_metadata(&destination).is_ok() {
        return Err(refused("destination already exists"));
    }
    let source_lock = crate::startup_lock(&source, false, false)?;
    let entries = inventory(&source, cancelled)?;
    let locked = source_lock.metadata()?;
    if !entries
        .iter()
        .any(|e| e.path == "writer.lock" && e.device == locked.dev() && e.inode == locked.ino())
    {
        return Err(refused("source coordination inode changed"));
    }
    let mut affected: Vec<RecoveredWal> = vec![];
    for tier in [DurableTier::Archive, DurableTier::Embeddings] {
        let main = source.join(tier.file());
        if !owned_metadata(&main)?.is_some_and(|m| m.is_file() && m.len() > 0) {
            return Err(refused("missing durable main database"));
        }
        let path = sibling(&main, "-wal");
        if owned_metadata(&path)?.is_some_and(|m| m.len() > 0) {
            if owned_metadata(&sibling(&main, "-shm"))?.is_none() {
                return Err(refused("orphan durable WAL"));
            }
            if let Some(tail) = wal::recovery_tail(
                &path,
                &mut || cancelled.load(std::sync::atomic::Ordering::Acquire),
                tier,
            )? {
                affected.push(tail);
            }
        }
    }
    if affected.len() != 1 {
        return Err(refused(
            "recovery requires exactly one complete previous-generation durable WAL tail",
        ));
    }
    let version = raw_versions(&source)?.ok_or_else(|| refused("missing archive format"))?;
    if !matches!(version, 1 | 2) {
        return Err(refused("unsupported archive format"));
    }
    let stage = tempfile::Builder::new()
        .prefix(".polylogue-recovery-")
        .tempdir_in(&parent)?;
    crate::private_dir(stage.path())?;
    native::no_cow(stage.path())?;
    let stage_path = stage.path().to_owned();
    let result = (|| {
        let destination_lock = Arc::new(crate::startup_lock(&stage_path, true, true)?);
        for entry in &entries {
            if entry.path != "writer.lock" {
                copy_file(
                    &source.join(&entry.path),
                    &stage_path.join(&entry.path),
                    entry,
                    cancelled,
                )?;
            }
        }
        if inventory(&source, cancelled)? != entries {
            return Err(refused("source changed during copy"));
        }
        boundary(Boundary::Copied)?;
        wal::check_cancel(cancelled)?;
        crate::initialize_sqlite()?;
        crate::preflight_inner(&stage_path, true)?;
        let manifest = json_bytes(&entries)?;
        let recovery = RecoveryQualification {
            version: 1,
            latest_acknowledged_head: HeadCoverage::Unknown,
            reason: RecoveryReason::UncertainGenerationTail,
            affected,
            source_manifest_blake3: BlobHash::of(&manifest),
        };
        let receipt_relative =
            PathBuf::from("recovery-receipts").join(recovery.source_manifest_blake3.to_string());
        let receipts = stage_path.join(&receipt_relative);
        crate::private_dir(&receipts)?;
        write_new(&receipts.join("source-manifest.json"), &manifest)?;
        let path = stage_path.join("archive.db");
        let mut archive = open(&path, false, true)?;
        integrity(&archive, &path, cancelled)?;
        set_format(&mut archive, &path, Some(&recovery))?;
        for name in ["archive.db", "embeddings.db", "index.db"] {
            wal::check_cancel(cancelled)?;
            let file = stage_path.join(name);
            let conn = open(&file, false, name != "index.db")?;
            integrity(&conn, &file, cancelled)?;
            db(&file, || {
                let (busy, log, checkpointed): (i64, i64, i64) =
                    conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
                        Ok((r.get(0)?, r.get(1)?, r.get(2)?))
                    })?;
                if busy != 0 || log != checkpointed {
                    return Err(rusqlite::Error::InvalidQuery);
                }
                Ok(())
            })?;
        }
        drop(archive);
        wal::validate(&stage_path, cancelled)?;
        crate::preflight(&stage_path)?;
        let reader = ReadHandle::owned(stage_path.clone(), Arc::clone(&destination_lock))?;
        let (checked_blobs, checked_blob_bytes) = blobs(&reader, cancelled)?;
        let status = reader.status()?;
        if status.recovery.as_ref() != Some(&recovery) {
            return Err(refused("reopened qualification mismatch"));
        }
        drop(reader);
        boundary(Boundary::Qualified)?;
        let report = RecoveryReport {
            format: "polylogue-recovery",
            version: 1,
            destination: destination.clone(),
            recovery,
            receipt: destination.join(&receipt_relative).join("report.json"),
            checked_blobs,
            checked_blob_bytes,
            checks: vec![
                "source_copy_hash_readback_and_independent_inodes",
                "committed_current_generation_wal_checksums",
                "one_complete_previous_generation_tail",
                "pinned_sqlite_production_checksum_vfs",
                "database_integrity_and_foreign_keys",
                "immutable_import_identities",
                "blob_and_retained_revision_content_blake3",
                "production_reader_reopen",
                "persistent_unknown_historical_head",
            ],
        };
        write_new(&receipts.join("report.json"), &json_bytes(&report)?)?;
        sync_tree(&stage_path)?;
        if inventory(&source, cancelled)? != entries {
            return Err(refused("source changed before publication"));
        }
        boundary(Boundary::BeforePublish)?;
        wal::check_cancel(cancelled)?;
        native::publish(&stage_path, &destination)?;
        // The rename is the commit point. Cancellation afterwards reports the
        // completed qualified destination; it never deletes a published copy.
        File::open(&parent)?.sync_all()?;
        boundary(Boundary::Published)?;
        Ok(report)
    })();
    match result {
        Ok(report) => Ok(report),
        Err(error) => {
            if stage_path.exists() {
                let path = stage.keep();
                Err(Error::RecoveryStaging {
                    path,
                    error: Box::new(error),
                })
            } else {
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests;
