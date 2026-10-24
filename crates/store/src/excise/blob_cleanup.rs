//! Replay only the local file deletions recorded by a canonical excision.
use crate::{Archive, Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior};

pub(crate) fn pending(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM pending_blob_excision)",
        [],
        |r| r.get(0),
    )?)
}

/// Records an authorized local retirement in the caller's archive transaction.
/// File removal belongs to replay after commit.
pub(super) fn record(tx: &Transaction<'_>, excision_id: &str, hash: &BlobHash) -> Result<()> {
    tx.execute(
        "INSERT INTO pending_blob_excision(hash,excision_id) VALUES (?1,?2)
         ON CONFLICT(hash) DO UPDATE SET excision_id=excluded.excision_id",
        rusqlite::params![&hash.0[..], excision_id],
    )?;
    Ok(())
}

pub(super) fn require_no_delta_base(conn: &Connection, hash: &BlobHash) -> Result<()> {
    let retained: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM blob WHERE base=?1)",
        [&hash.0[..]],
        |r| r.get(0),
    )?;
    if retained {
        return Err(StoreError::ExcisionBlobReferenced(*hash));
    }
    Ok(())
}

/// Caller owns the archive's writer lock. No external read roots are opened.
pub(crate) fn recover(archive: &Archive, conn: &Connection) -> Result<()> {
    if !pending(conn)? {
        return Ok(());
    }
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600)?;
    delete_pending(conn, |hash| {
        crate::fault::check("excision.before_blob_delete", &hash.to_string())?;
        match store.delete(&hashed_blobs::Hash(hash.0)) {
            Ok(()) | Err(hashed_blobs::Error::Missing(_)) => {}
            Err(error) => return Err(error.into()),
        }
        crate::fault::check("excision.after_blob_delete", &hash.to_string())
    })
}

fn require_unreferenced(
    conn: &Connection,
    current: &crate::sinex_rows::CurrentBatches,
    hash: &BlobHash,
) -> Result<()> {
    let retained: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM blob WHERE hash=?1 OR base=?1) OR EXISTS(SELECT 1 FROM extent WHERE blob=?1) OR EXISTS(SELECT 1 FROM extracted_blob WHERE blob=?1)", [&hash.0[..]], |row| row.get(0))?;
    if retained || current.contains(hash)? {
        return Err(StoreError::ExcisionBlobReferenced(*hash));
    }
    Ok(())
}

// The archive-writer lock keeps references stable during deletion. Pending
// intent remains visible until a fresh short transaction validates and acks it.
fn delete_pending(
    conn: &Connection,
    mut delete: impl FnMut(&BlobHash) -> Result<()>,
) -> Result<()> {
    // Startup owns ArchiveLock; normal cleanup runs synchronously inside the
    // serial Writer's excision. Search writes only index.db and paid-vector
    // writers only embeddings.db, so no replacement manifest changes while
    // this caller retains canonical writer custody.
    let current = crate::sinex_rows::CurrentBatches::new(conn)?;
    loop {
        let pending: Vec<(Vec<u8>, String)> = conn
            .prepare("SELECT hash,excision_id FROM pending_blob_excision ORDER BY hash LIMIT 500")?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        if pending.is_empty() {
            return Ok(());
        }
        let mut deleted = Vec::with_capacity(pending.len());
        for (bytes, operation) in pending {
            let hash = BlobHash(
                bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| StoreError::Invalid("invalid pending blob hash".into()))?,
            );
            require_unreferenced(conn, &current, &hash)?;
            delete(&hash)?;
            deleted.push((hash, operation));
        }
        let tx = crate::publication::transaction(conn, TransactionBehavior::Immediate)?;
        for (hash, operation) in deleted {
            let now: Option<String> = tx
                .query_row(
                    "SELECT excision_id FROM pending_blob_excision WHERE hash=?1",
                    [&hash.0[..]],
                    |row| row.get(0),
                )
                .optional()?;
            if now.as_deref() != Some(operation.as_str()) {
                return Err(StoreError::Invalid(
                    "pending blob cleanup changed during deletion".into(),
                ));
            }
            require_unreferenced(&tx, &current, &hash)?;
            tx.execute(
                "DELETE FROM pending_blob_excision WHERE hash=?1",
                [&hash.0[..]],
            )?;
        }
        tx.commit()?;
    }
}

#[cfg(test)]
mod tests;
