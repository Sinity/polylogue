use super::super::record_ranges::Erasure;
use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};

/// Private computation for one shared-input rewrite. All pieces are checked
/// before any replacement is installed; the archive owns durable recovery.
pub(super) struct Erasures {
    conn: Connection,
    _file: tempfile::NamedTempFile,
}

impl Erasures {
    pub fn new() -> Result<Self> {
        let file = tempfile::NamedTempFile::new()?;
        let conn = Connection::open(file.path())?;
        conn.execute_batch(
            "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
             PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
             CREATE TABLE requested(hash BLOB, start INTEGER, end INTEGER,
                 PRIMARY KEY(hash,start,end)) WITHOUT ROWID;
             CREATE TABLE checked(hash BLOB PRIMARY KEY, documents INTEGER,
                 erased INTEGER, changed INTEGER) WITHOUT ROWID;
             CREATE TABLE ranges(hash BLOB, start INTEGER, end INTEGER,
                 PRIMARY KEY(hash,start,end)) WITHOUT ROWID;
             BEGIN;",
        )?;
        Ok(Self { conn, _file: file })
    }

    pub fn request(&self, hash: &[u8], start: usize, end: usize) -> Result<()> {
        self.conn
            .prepare_cached("INSERT OR IGNORE INTO requested VALUES(?1,?2,?3)")?
            .execute(params![hash, start as i64, end as i64])?;
        Ok(())
    }

    pub fn requested(&self, hash: &[u8]) -> Result<Vec<(usize, usize)>> {
        self.read_ranges("requested", hash)
    }

    pub fn counts(&self, hash: &[u8]) -> Result<Option<(usize, usize)>> {
        Ok(self
            .conn
            .query_row(
                "SELECT documents,erased FROM checked WHERE hash=?1",
                [hash],
                |r| Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as usize)),
            )
            .optional()?)
    }

    pub fn checked(&self, hash: &[u8], plan: &Erasure) -> Result<()> {
        let changed = !plan.ranges.is_empty();
        self.conn
            .prepare_cached("INSERT INTO checked VALUES(?1,?2,?3,?4)")?
            .execute(params![
                hash,
                plan.documents as i64,
                plan.erased_documents as i64,
                changed
            ])?;
        for &(start, end) in &plan.ranges {
            self.conn
                .prepare_cached("INSERT OR IGNORE INTO ranges VALUES(?1,?2,?3)")?
                .execute(params![hash, start as i64, end as i64])?;
        }
        Ok(())
    }

    pub fn has_unchecked(&self) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM requested r WHERE NOT EXISTS
                (SELECT 1 FROM checked c WHERE c.hash=r.hash))",
            [],
            |r| r.get(0),
        )?)
    }

    /// Visit the completed plans in hash order, loading one blob's ranges.
    pub fn visit(&self, mut f: impl FnMut(&[u8], &[(usize, usize)]) -> Result<()>) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT hash FROM checked WHERE changed ORDER BY hash")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            f(&hash, &self.read_ranges("ranges", &hash)?)?;
        }
        Ok(())
    }

    fn read_ranges(&self, table: &str, hash: &[u8]) -> Result<Vec<(usize, usize)>> {
        let mut stmt = self.conn.prepare_cached(&format!(
            "SELECT start,end FROM {table} WHERE hash=?1 ORDER BY start,end"
        ))?;
        Ok(stmt
            .query_map([hash], |r| {
                Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as usize))
            })?
            .collect::<rusqlite::Result<_>>()?)
    }
}

#[cfg(test)]
mod tests;
