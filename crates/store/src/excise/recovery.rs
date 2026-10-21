//! Compare excision revisions inside a read snapshot; index data is derived.
use crate::{Result, StoreError};
use rusqlite::{Connection, params};

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

pub(crate) fn acknowledge(archive: &Connection, index: &Connection) -> Result<()> {
    set(index, &current(archive)?)
}

pub(crate) fn needs_rebuild(applied: &Revision, current: &Revision) -> Result<bool> {
    if applied == current {
        return Ok(false);
    }
    if current.version > applied.version {
        return Ok(true);
    }
    Err(StoreError::ExcisionConflict {
        archive: current.version,
        index: applied.version,
    })
}

pub(crate) fn check_snapshot(conn: &Connection) -> Result<()> {
    // This statement pins both snapshots before the Reader is admitted.
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
