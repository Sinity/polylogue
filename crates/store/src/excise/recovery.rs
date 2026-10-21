//! Compare excision revisions inside a read snapshot; index data is derived.
use crate::{Result, StoreError};
use rusqlite::{Connection, OptionalExtension, params};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Revision {
    pub version: u64,
    pub id: String,
}
impl Revision {
    fn decode(version: String, id: String) -> Result<Self> {
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
fn policy_hash(row: &rusqlite::Row<'_>) -> rusqlite::Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    for i in 0..3 {
        let value = row.get_ref(i)?.as_str()?;
        hash.update(&(value.len() as u64).to_be_bytes());
        hash.update(value.as_bytes());
    }
    Ok(hash.finalize())
}

/// The caller includes receipts and revision in the index transaction.
pub(crate) fn acknowledge(archive: &Connection, index: &Connection) -> Result<()> {
    index.execute("DELETE FROM excision_applied", [])?;
    let mut select = archive
        .prepare("SELECT scope,refused_json,message_ids_json,id FROM excision ORDER BY id")?;
    let mut insert =
        index.prepare("INSERT INTO excision_applied(id_hash,policy_hash) VALUES (?1,?2)")?;
    let mut rows = select.query([])?;
    while let Some(row) = rows.next()? {
        insert.execute(params![
            blake3::hash(row.get::<_, String>(3)?.as_bytes())
                .as_bytes()
                .as_slice(),
            policy_hash(row)?.as_bytes().as_slice()
        ])?;
    }
    set(index, &current(archive)?)
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
        let count: i64 =
            index.query_row("SELECT count(*) FROM excision_applied", [], |r| r.get(0))?;
        let mut matched = 0i64;
        let mut policies = archive
            .prepare("SELECT scope,refused_json,message_ids_json,id FROM excision ORDER BY id")?;
        let mut receipt =
            index.prepare("SELECT policy_hash FROM excision_applied WHERE id_hash=?1")?;
        let mut rows = policies.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(3)?;
            let id_hash = blake3::hash(id.as_bytes());
            let expected: Option<Vec<u8>> = receipt
                .query_row([id_hash.as_bytes().as_slice()], |r| r.get(0))
                .optional()?;
            if let Some(expected) = expected {
                if policy_hash(row)?.as_bytes().as_slice() != expected {
                    return Err(conflict());
                }
                matched += 1;
            }
        }
        if matched != count {
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
    // This statement pins both snapshots before the first database read.
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
    Ok(hash.finalize())
}

#[cfg(test)]
mod tests;
