//! Current completed discovery snapshots. A partial incremental walk never
//! replaces these full-walk facts. Confirmation order chooses the current fact.
use crate::{Reader, Result, StoreError};
use polylogue_model::capture::Coverage;
use rusqlite::{Connection, OptionalExtension, params};

pub const SCHEMA: &str = "CREATE TABLE capture_coverage (machine TEXT PRIMARY KEY, observed_at INTEGER NOT NULL, payload_json TEXT NOT NULL) STRICT, WITHOUT ROWID;";

pub fn apply(tx: &Connection, coverage: &Coverage) -> Result<()> {
    coverage
        .validate()
        .map_err(|error| StoreError::Invalid(error.into()))?;
    tx.execute("INSERT INTO capture_coverage(machine,observed_at,payload_json) VALUES (?1,?2,?3) ON CONFLICT(machine) DO UPDATE SET observed_at=excluded.observed_at,payload_json=excluded.payload_json",
        params![coverage.machine,coverage.observed_at.unix_ms(),serde_json::to_string(coverage)?])?;
    Ok(())
}

pub fn clear(tx: &Connection) -> Result<()> {
    tx.execute("DELETE FROM capture_coverage", [])?;
    Ok(())
}

impl Reader<'_> {
    pub fn capture_coverage(&self, machine: &str) -> Result<Option<Coverage>> {
        let payload: Option<String> = self
            .conn()?
            .query_row(
                "SELECT payload_json FROM capture_coverage WHERE machine=?1",
                [machine],
                |row| row.get(0),
            )
            .optional()?;
        payload
            .map(|payload| serde_json::from_str(&payload).map_err(StoreError::from))
            .transpose()
    }
    pub fn all_capture_coverage(&self) -> Result<Vec<Coverage>> {
        let conn = self.conn()?;
        let mut statement =
            conn.prepare("SELECT payload_json FROM capture_coverage ORDER BY machine")?;
        let payloads = statement.query_map([], |row| row.get::<_, String>(0))?;
        payloads
            .map(|payload| serde_json::from_str(&payload?).map_err(StoreError::from))
            .collect()
    }
}

#[cfg(test)]
mod tests;
