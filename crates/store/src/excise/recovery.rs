//! Compare excision revisions inside a read snapshot; index data is derived.
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Revision {
    pub version: u64,
    pub id: String,
}
impl Revision {
    pub(super) fn decode(version: String, id: String) -> Result<Self> {
        let version = version
            .parse()
            .map_err(|_| StoreError::Invalid("invalid excision revision metadata".into()))?;
        if (version == 0) != id.is_empty() {
            return Err(StoreError::Invalid(
                "invalid excision revision metadata".into(),
            ));
        }
        Ok(Self { version, id })
    }
}

pub(crate) fn current(conn: &Connection) -> Result<Revision> {
    let (version, id) = conn.query_row(
        "SELECT
        COALESCE((SELECT value FROM meta WHERE key='excision_version'),'0'),
        COALESCE((SELECT value FROM meta WHERE key='excision_revision'),'')",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Revision::decode(version, id)
}

pub(crate) fn set(conn: &Connection, revision: &Revision) -> Result<()> {
    conn.execute("INSERT INTO meta(key,value) VALUES ('excision_revision',?1),('excision_version',?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value",params![&revision.id,revision.version.to_string()])?;
    Ok(())
}

// A revision counter detects stale snapshots. These receipts additionally
// prevent a newer fork from silently dropping an already applied removal.
fn policy_hash(conn: &Connection, row: &rusqlite::Row<'_>) -> Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    for i in 0..3 {
        let value: String = row.get(i)?;
        hash.update(&(value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    }
    if let Some(policy) = super::vector_policy::policy(conn, &row.get::<_, String>(3)?)? {
        hash.update(b"vector_deletion_policy");
        hash.update(policy.as_bytes());
    }
    let has_documents:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='excision_document')",[],|r|r.get(0))?;
    if has_documents {
        let mut statement = conn.prepare(
            "SELECT document_id FROM excision_document WHERE excision_id=?1 ORDER BY document_id",
        )?;
        let mut rows = statement.query([row.get::<_, String>(3)?])?;
        let mut first = true;
        while let Some(row) = rows.next()? {
            if first {
                hash.update(b"document_exclusion_v1");
                first = false;
            }
            let id: String = row.get(0)?;
            hash.update(&(id.len() as u64).to_be_bytes());
            hash.update(id.as_bytes());
        }
    }
    Ok(hash.finalize())
}

/// The caller includes receipts and revision in the index transaction.
pub(crate) fn acknowledge(archive: &Connection, index: &Connection) -> Result<()> {
    remember_policies(archive, index)?;
    set(index, &current(archive)?)
}

/// Snapshot only stable policy hashes in an existing receipt table.
pub(crate) fn remember_policies(archive: &Connection, receipts: &Connection) -> Result<()> {
    receipts.execute("DELETE FROM excision_applied", [])?;
    let mut select = archive
        .prepare("SELECT scope,refused_json,message_ids_json,id FROM excision ORDER BY id")?;
    let mut insert =
        receipts.prepare("INSERT INTO excision_applied(id_hash,policy_hash) VALUES (?1,?2)")?;
    let mut rows = select.query([])?;
    while let Some(row) = rows.next()? {
        insert.execute(params![
            blake3::hash(row.get::<_, String>(3)?.as_bytes())
                .as_bytes()
                .as_slice(),
            policy_hash(archive, row)?.as_bytes().as_slice()
        ])?;
    }
    Ok(())
}

pub(crate) fn preserves_policies(archive: &Connection, receipts: &Connection) -> Result<bool> {
    let count: i64 =
        receipts.query_row("SELECT count(*) FROM excision_applied", [], |r| r.get(0))?;
    let mut matched = 0i64;
    let mut policies = archive
        .prepare("SELECT scope,refused_json,message_ids_json,id FROM excision ORDER BY id")?;
    let mut receipt =
        receipts.prepare("SELECT policy_hash FROM excision_applied WHERE id_hash=?1")?;
    let mut rows = policies.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(3)?;
        let id_hash = blake3::hash(id.as_bytes());
        let expected: Option<Vec<u8>> = receipt
            .query_row([id_hash.as_bytes().as_slice()], |r| r.get(0))
            .optional()?;
        if let Some(expected) = expected {
            if policy_hash(archive, row)?.as_bytes().as_slice() != expected {
                return Ok(false);
            }
            matched += 1;
        }
    }
    Ok(matched == count)
}

pub(crate) fn needs_rebuild(archive: &Connection, index: &Connection) -> Result<bool> {
    let applied = current(index)?;
    let current = current(archive)?;
    let conflict = || StoreError::ExcisionConflict {
        archive: current.version,
        index: applied.version,
    };
    let has_receipts: bool = index.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='excision_applied')",
        [],
        |r| r.get(0),
    )?;
    if has_receipts {
        if !preserves_policies(archive, index)? {
            return Err(conflict());
        }
    } else if applied.version != 0 {
        // Pre-receipt indexes have no known policy history. A marked index
        // missing its receipt table cannot be treated as such a predecessor.
        return Err(conflict());
    }
    if applied == current {
        return Ok(false);
    }
    if current.version > applied.version {
        return Ok(true);
    }
    Err(conflict())
}

pub(crate) fn check_snapshot(conn: &Connection) -> Result<()> {
    // The caller excludes publication while this statement pins both snapshots.
    let (applied_version, applied_id, current_version, current_id) = conn
        .prepare_cached(
            "SELECT
        COALESCE((SELECT value FROM main.meta WHERE key='excision_version'),'0'),
        COALESCE((SELECT value FROM main.meta WHERE key='excision_revision'),''),
        COALESCE((SELECT value FROM a.meta WHERE key='excision_version'),
            CASE WHEN EXISTS(SELECT 1 FROM a.excision) THEN '1' ELSE '0' END),
        COALESCE((SELECT value FROM a.meta WHERE key='excision_revision'),
            CASE WHEN EXISTS(SELECT 1 FROM a.excision) THEN 'legacy' ELSE '' END)",
        )?
        .query_row([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
    let applied = Revision::decode(applied_version, applied_id)?;
    let current = Revision::decode(current_version, current_id)?;
    if applied != current {
        return Err(StoreError::ExcisionPending);
    }
    // The statement above pins both databases. Older read-only archives may
    // predate the cleanup table; the daemon migrates before serving writes.
    for table in ["pending_vector_excision", "pending_blob_excision"] {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM a.sqlite_master WHERE type='table' AND name=?1)",
            [table],
            |r| r.get(0),
        )?;
        if exists
            && conn.query_row(
                &format!("SELECT EXISTS(SELECT 1 FROM a.{table})"),
                [],
                |r| r.get::<_, bool>(0),
            )?
        {
            return Err(StoreError::ExcisionPending);
        }
    }
    Ok(())
}

/// Restore may delete and reinsert identical rows. Only the resulting excision
/// state should invalidate an index, not those intermediate trigger executions.
pub(crate) fn fingerprint(conn: &Connection) -> Result<blake3::Hash> {
    use rusqlite::types::ValueRef;
    let mut hash = blake3::Hasher::new();
    let mut stmt = conn.prepare("SELECT * FROM excision ORDER BY id")?;
    let columns = stmt.column_count();
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        hash.update(b"row");
        for i in 0..columns {
            match row.get_ref(i)? {
                ValueRef::Null => {
                    hash.update(&[0]);
                }
                ValueRef::Integer(n) => {
                    hash.update(&[1]);
                    hash.update(&n.to_be_bytes());
                }
                ValueRef::Real(n) => {
                    hash.update(&[2]);
                    hash.update(&n.to_bits().to_be_bytes());
                }
                ValueRef::Text(bytes) | ValueRef::Blob(bytes) => {
                    hash.update(&[if matches!(row.get_ref(i)?, ValueRef::Text(_)) {
                        3
                    } else {
                        4
                    }]);
                    hash.update(&(bytes.len() as u64).to_be_bytes());
                    hash.update(bytes);
                }
            }
        }
    }
    hash.update(super::vector_policy::fingerprint(conn)?.as_bytes());
    let has_documents:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='excision_document' AND type='table')",[],|r|r.get(0))?;
    if has_documents {
        let mut stmt=conn.prepare("SELECT excision_id,document_id FROM excision_document ORDER BY excision_id,document_id")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            hash.update(b"document_exclusion_v1");
            for column in 0..2 {
                let text: String = row.get(column)?;
                hash.update(&(text.len() as u64).to_be_bytes());
                hash.update(text.as_bytes());
            }
        }
    }
    Ok(hash.finalize())
}

#[cfg(test)]
mod tests;
