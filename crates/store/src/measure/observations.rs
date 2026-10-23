//! Typed normalized observations, independent of message ownership.
use super::Scope;
use crate::{Reader, Result};
use serde::de::DeserializeOwned;

/// SQLite keeps deduplication bounded in memory across copied inputs. The
/// original observation identity wins over an enclosing event's placement.
pub(super) fn visit<T: DeserializeOwned>(
    reader: &Reader<'_>,
    scope: &Scope,
    kind: &str,
    mut visit: impl FnMut(&str, Option<i64>, T) -> Result<()>,
) -> Result<()> {
    let conn = reader.conn()?;
    let scratch = tempfile::NamedTempFile::new()?;
    let seen = rusqlite::Connection::open(scratch.path())?;
    seen.execute_batch(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        PRAGMA mmap_size=0; PRAGMA cache_size=-1024; PRAGMA temp_store=FILE;
        CREATE TABLE measurement_observation(
        origin TEXT NOT NULL, id TEXT NOT NULL, at INTEGER, data BLOB NOT NULL,
        confirmed INTEGER NOT NULL, PRIMARY KEY(origin,id)) WITHOUT ROWID;",
    )?;
    let (cond, mut params) = scope.condition();
    params.push(kind.to_owned().into());
    let mut stmt = conn.prepare(&format!(
        "SELECT s.origin,m.at,m.data FROM current_event m
        JOIN session s ON s.pk=m.session_pk WHERE {cond} AND m.kind=?{} ORDER BY m.at,m.id",
        params.len()
    ))?;
    let mut rows = stmt.query(rusqlite::params_from_iter(params.iter()))?;
    while let Some(row) = rows.next()? {
        let packed: Vec<u8> = row.get(2)?;
        let value: serde_json::Value = crate::encoding::unpack_json(&packed)?;
        let Some(id) = value.get("observation").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let confirmed = value
            .get("completion_observed")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        seen.execute("INSERT INTO measurement_observation(id,origin,at,data,confirmed) VALUES (?1,?2,?3,?4,?5)
            ON CONFLICT(origin,id) DO UPDATE SET origin=excluded.origin,at=excluded.at,data=excluded.data,confirmed=excluded.confirmed
            WHERE excluded.confirmed > measurement_observation.confirmed",
            rusqlite::params![id, row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?, packed, confirmed])?;
    }
    let mut stmt =
        seen.prepare("SELECT origin,at,data FROM measurement_observation ORDER BY at,id")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        visit(
            &row.get::<_, String>(0)?,
            row.get(1)?,
            crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(2)?)?,
        )?;
    }
    Ok(())
}
