//! Replay only the local file deletions recorded by a canonical excision.
use crate::{Archive, Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, Transaction, TransactionBehavior};

pub(crate) fn pending(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM pending_blob_excision)",
        [],
        |r| r.get(0),
    )?)
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
    let current = crate::sinex_rows::CurrentBatches::new(conn)?;
    loop {
        let tx = Transaction::new_unchecked(conn, TransactionBehavior::Immediate)?;
        let hashes: Vec<Vec<u8>> = tx
            .prepare("SELECT hash FROM pending_blob_excision ORDER BY hash LIMIT 500")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        if hashes.is_empty() {
            return Ok(());
        }
        for bytes in hashes {
            let hash = BlobHash(
                bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| StoreError::Invalid("invalid pending blob hash".into()))?,
            );
            let retained: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM blob WHERE hash=?1 OR base=?1) OR EXISTS(SELECT 1 FROM extent WHERE blob=?1)", [&hash.0[..]], |r| r.get(0))?;
            if retained || current.contains(&hash)? {
                return Err(StoreError::ExcisionBlobReferenced(hash));
            }
            crate::fault::check("excision.before_blob_delete", &hash.to_string())?;
            match store.delete(&hashed_blobs::Hash(hash.0)) {
                Ok(()) | Err(hashed_blobs::Error::Missing(_)) => {}
                Err(e) => return Err(e.into()),
            }
            crate::fault::check("excision.after_blob_delete", &hash.to_string())?;
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
