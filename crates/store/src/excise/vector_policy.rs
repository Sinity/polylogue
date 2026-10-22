//! Immutable deletion sets and each embeddings file's applied-policy evidence.
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};

mod hashes;

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

pub(crate) fn exists(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_deletion_policy')", [], |r|r.get(0))?)
}

pub(crate) fn digest(version: i64, json: &str) -> Result<blake3::Hash> {
    hashes::visit(version, json, |_| Ok(()))?;
    let mut h = blake3::Hasher::new();
    h.update(b"polylogue-vector-deletion-v1\0");
    h.update(json.as_bytes());
    Ok(h.finalize())
}

pub(crate) fn policy(conn: &Connection, id: &str) -> Result<Option<blake3::Hash>> {
    if !exists(conn)? {
        return Ok(None);
    }
    let row: Option<(i64, String)> = conn
        .query_row(
            "SELECT version,hashes_json FROM vector_deletion_policy WHERE excision_id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    row.map(|(v, j)| digest(v, &j)).transpose()
}

pub(crate) fn revision(conn: &Connection) -> Result<String> {
    Ok(conn.query_row(
        "SELECT COALESCE((SELECT value FROM meta WHERE key='vector_policy_revision'),'')",
        [],
        |r| r.get(0),
    )?)
}

fn identity(conn: &Connection) -> Result<String> {
    conn.query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
        r.get(0)
    })
    .map_err(Into::into)
}

fn receipt_hash(archive: &str, id: &str, policy: blake3::Hash) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    h.update(b"polylogue-vector-receipt-v1\0");
    for part in [
        archive.as_bytes(),
        id.as_bytes(),
        policy.as_bytes().as_slice(),
    ] {
        h.update(&(part.len() as u64).to_be_bytes());
        h.update(part);
    }
    h.finalize()
}

pub(crate) fn acknowledge(vectors: &Connection, canonical: &Connection, id: &str) -> Result<()> {
    let hash = policy(canonical, id)?.ok_or_else(|| invalid("missing vector deletion policy"))?;
    let archive = identity(canonical)?;
    let receipt = receipt_hash(&archive, id, hash);
    vectors.execute(
        "INSERT INTO vector_policy_receipt VALUES (?1,?2,?3)",
        params![id, archive, receipt.as_bytes().as_slice()],
    )?;
    stamp(vectors, canonical)
}

fn stamp(vectors: &Connection, canonical: &Connection) -> Result<()> {
    for (key, value) in [
        ("archive_id", identity(canonical)?),
        ("policy_revision", revision(canonical)?),
    ] {
        vectors.execute("INSERT INTO vector_meta VALUES (?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",params![key,value])?;
    }
    Ok(())
}

/// Caller holds the embeddings write transaction before any canonical snapshot.
/// All receipts are checked before any deletion. Matching policies never replay.
pub(crate) fn reconcile(vectors: &Connection, canonical: &Connection) -> Result<()> {
    if !exists(canonical)? {
        return Ok(());
    }
    let archive = identity(canonical)?;
    let bound: Option<String> = vectors
        .query_row(
            "SELECT value FROM vector_meta WHERE key='archive_id'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    if bound.is_some_and(|id| id != archive) {
        let populated:bool=vectors.query_row("SELECT EXISTS(SELECT 1 FROM embedded) OR EXISTS(SELECT 1 FROM embed_failure) OR EXISTS(SELECT 1 FROM vector_policy_receipt)",[],|r|r.get(0))?;
        if populated {
            return Err(invalid("embeddings belong to a different archive"));
        }
    }
    {
        let mut stmt = vectors
            .prepare("SELECT excision_id,archive_id,policy_hash FROM vector_policy_receipt")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let recorded: Vec<u8> = row.get(2)?;
            if row.get::<_, String>(1)? != archive
                || policy(canonical, &id)?.is_none_or(|hash| {
                    receipt_hash(&archive, &id, hash).as_bytes().as_slice() != recorded
                })
            {
                return Err(invalid(
                    "embeddings vector deletion receipt conflicts with canonical policy",
                ));
            }
        }
    }
    let mut stmt=canonical.prepare("SELECT p.excision_id,p.version,p.hashes_json FROM vector_deletion_policy p LEFT JOIN excision x ON x.id=p.excision_id WHERE x.id IS NOT NULL ORDER BY p.excision_id")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let version = row.get(1)?;
        let json = row.get_ref(2)?;
        let json = json.as_str().map_err(|e| invalid(&e.to_string()))?;
        let hash = digest(version, json)?;
        let applied: bool = vectors.query_row(
            "SELECT EXISTS(SELECT 1 FROM vector_policy_receipt WHERE excision_id=?1)",
            [&id],
            |r| r.get(0),
        )?;
        if !applied {
            // The completed digest pass above validates the entire array before deletion.
            hashes::visit(version, json, |hash| {
                vectors.execute("DELETE FROM embedded WHERE text_hash=?1", [hash])?;
                vectors.execute("DELETE FROM embed_failure WHERE text_hash=?1", [hash])?;
                Ok(())
            })?;
            let receipt = receipt_hash(&archive, &id, hash);
            vectors.execute(
                "INSERT INTO vector_policy_receipt VALUES (?1,?2,?3)",
                params![id, archive, receipt.as_bytes().as_slice()],
            )?;
        }
    }
    let orphan:bool=canonical.query_row("SELECT EXISTS(SELECT 1 FROM vector_deletion_policy p WHERE NOT EXISTS(SELECT 1 FROM excision x WHERE x.id=p.excision_id))",[],|r|r.get(0))?;
    if orphan {
        return Err(invalid("vector deletion policy has no owning excision"));
    }
    stamp(vectors, canonical)
}

/// Both schemas are pinned by this statement. Receipt edits invalidate the stamp.
pub(crate) fn check(conn: &Connection) -> Result<()> {
    let canonical: String = conn.query_row(
        "SELECT COALESCE((SELECT value FROM canonical.meta WHERE key='vector_policy_revision'),'')",
        [],
        |r| r.get(0),
    )?;
    let has_meta: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM main.sqlite_master WHERE type='table' AND name='vector_meta')",
        [],
        |r| r.get(0),
    )?;
    if !has_meta {
        return if canonical.is_empty() {
            Ok(())
        } else {
            Err(StoreError::ExcisionPending)
        };
    }
    let ready:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM main.vector_meta WHERE key='policy_revision' AND value=COALESCE((SELECT value FROM canonical.meta WHERE key='vector_policy_revision'),'')) AND EXISTS(SELECT 1 FROM main.vector_meta WHERE key='archive_id' AND value=(SELECT value FROM canonical.meta WHERE key='archive_id'))",[],|r|r.get(0))?;
    if !ready {
        return Err(StoreError::ExcisionPending);
    }
    Ok(())
}

/// Stable content fingerprint, independent of restore's delete/insert triggers.
pub(crate) fn fingerprint(conn: &Connection) -> Result<blake3::Hash> {
    let mut h = blake3::Hasher::new();
    if exists(conn)? {
        let mut stmt=conn.prepare("SELECT excision_id,version,hashes_json FROM vector_deletion_policy ORDER BY excision_id")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let json: String = row.get(2)?;
            let hash = digest(row.get(1)?, &json)?;
            h.update(&(id.len() as u64).to_be_bytes());
            h.update(id.as_bytes());
            h.update(hash.as_bytes());
        }
    }
    Ok(h.finalize())
}

#[cfg(test)]
mod tests;
