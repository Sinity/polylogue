//! Disk-backed key lookups while a first build defers natural-key indexes.
//! Rows share their parsed rows' transaction, including rollback and restart.

use crate::Result;
use rusqlite::{Connection, params};

#[derive(Default)]
pub(crate) struct FirstBuild;

impl FirstBuild {
    fn find(&self, conn: &Connection, kind: &str, key: &[u8], sql: &str) -> Result<Option<i64>> {
        let hash = blake3::hash(key);
        let mut statement = conn.prepare_cached(sql)?;
        let mut rows = statement.query(params![kind, &hash.as_bytes()[..8], key])?;
        Ok(rows.next()?.map(|r| r.get(0)).transpose()?)
    }

    fn add(&self, conn: &Connection, kind: &str, key: &[u8], pk: i64) -> Result<()> {
        let hash = blake3::hash(key);
        conn.prepare_cached("INSERT OR IGNORE INTO build_key(kind, hash, row) VALUES (?1, ?2, ?3)")?
            .execute(params![kind, &hash.as_bytes()[..8], pk])?;
        Ok(())
    }

    pub(crate) fn message(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.find(conn, "message", id.as_bytes(),
            "SELECT m.pk FROM build_key k JOIN message m ON m.pk=k.row
             WHERE k.kind=?1 AND k.hash=?2 AND CAST(m.id AS BLOB)=?3")
    }

    pub(crate) fn add_message(&self, conn: &Connection, id: &str, pk: i64) -> Result<()> {
        self.add(conn, "message", id.as_bytes(), pk)
    }

    pub(crate) fn event(&self, conn: &Connection, id: &str) -> Result<Option<i64>> {
        self.find(conn, "event", id.as_bytes(),
            "SELECT e.pk FROM build_key k JOIN event e ON e.pk=k.row
             WHERE k.kind=?1 AND k.hash=?2 AND CAST(e.id AS BLOB)=?3")
    }

    pub(crate) fn add_event(&self, conn: &Connection, id: &str, pk: i64) -> Result<()> {
        self.add(conn, "event", id.as_bytes(), pk)
    }

    pub(crate) fn text(&self, conn: &Connection, hash: &[u8; 32]) -> Result<Option<i64>> {
        self.find(conn, "text", hash,
            "SELECT t.id FROM build_key k JOIN text t ON t.id=k.row
             WHERE k.kind=?1 AND k.hash=?2 AND t.hash=?3")
    }

    pub(crate) fn add_text(&self, conn: &Connection, hash: &[u8; 32], id: i64) -> Result<()> {
        self.add(conn, "text", hash, id)
    }
}

#[cfg(test)]
mod tests;
