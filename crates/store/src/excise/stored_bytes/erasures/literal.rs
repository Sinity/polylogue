//! Literal matches preserve multiplicity; record erasure has separate normalized ranges.
use super::Erasures;
use crate::{Result, StoreError};
use rusqlite::{Connection, params};

impl Erasures {
    /// Freeze the source query's order before replacing any of its extents.
    pub(in crate::excise) fn freeze_literals(
        &self,
        archive: &Connection,
        input: &str,
    ) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS literal_blob(ordinal INTEGER PRIMARY KEY,hash BLOB NOT NULL UNIQUE,
                complete INTEGER NOT NULL DEFAULT 0);
             CREATE TABLE IF NOT EXISTS literal_range(blob INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS literal_order ON literal_range(blob,start,end);
             DELETE FROM literal_range; DELETE FROM literal_blob;",
        )?;
        let mut query = archive.prepare("SELECT DISTINCT blob FROM extent WHERE input=?1")?;
        let mut rows = query.query([input])?;
        while let Some(row) = rows.next()? {
            let hash = row.get_ref(0)?;
            let hash = hash
                .as_blob()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            self.conn
                .prepare_cached("INSERT INTO literal_blob(hash) VALUES (?1)")?
                .execute([hash])?;
        }
        Ok(())
    }

    pub(in crate::excise) fn literal_blobs(
        &self,
        mut visit: impl FnMut(&[u8], &Literal<'_>) -> Result<()>,
    ) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT ordinal,hash FROM literal_blob ORDER BY ordinal")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let hash = row.get_ref(1)?;
            visit(
                hash.as_blob()
                    .map_err(|e| StoreError::Invalid(e.to_string()))?,
                &Literal {
                    owner: self,
                    ordinal: row.get(0)?,
                },
            )?;
        }
        Ok(())
    }
}

pub(in crate::excise) struct Literal<'a> {
    owner: &'a Erasures,
    ordinal: i64,
}

impl Literal<'_> {
    pub(in crate::excise) fn begin(&self) -> Result<()> {
        self.owner.conn.execute(
            "UPDATE literal_blob SET complete=0 WHERE ordinal=?1",
            [self.ordinal],
        )?;
        self.owner
            .conn
            .execute("DELETE FROM literal_range WHERE blob=?1", [self.ordinal])?;
        Ok(())
    }

    pub(in crate::excise) fn insert(&self, start: usize, end: usize) -> Result<()> {
        crate::fault::check("excision_literals.insert", "")?;
        self.owner
            .conn
            .prepare_cached("INSERT INTO literal_range VALUES (?1,?2,?3)")?
            .execute(params![self.ordinal, start as i64, end as i64])?;
        Ok(())
    }

    pub(in crate::excise) fn finish(&self) -> Result<u64> {
        let count: i64 = self.owner.conn.query_row(
            "SELECT count(*) FROM literal_range WHERE blob=?1",
            [self.ordinal],
            |row| row.get(0),
        )?;
        crate::fault::check("excision_literals.complete", "")?;
        self.owner.conn.execute(
            "UPDATE literal_blob SET complete=1 WHERE ordinal=?1",
            params![self.ordinal],
        )?;
        Ok(count as u64)
    }

    pub(in crate::excise) fn ranges(
        &self,
        mut visit: impl FnMut(usize, usize) -> Result<()>,
    ) -> Result<()> {
        let complete: bool = self.owner.conn.query_row(
            "SELECT complete FROM literal_blob WHERE ordinal=?1",
            [self.ordinal],
            |row| row.get(0),
        )?;
        if !complete {
            return Err(StoreError::Invalid(
                "literal match scan is incomplete".into(),
            ));
        }
        let mut stmt = self
            .owner
            .conn
            .prepare("SELECT start,end FROM literal_range WHERE blob=?1 ORDER BY start,end")?;
        let mut rows = stmt.query([self.ordinal])?;
        while let Some(row) = rows.next()? {
            crate::fault::check("excision_literals.read", "")?;
            visit(
                row.get::<_, i64>(0)? as usize,
                row.get::<_, i64>(1)? as usize,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
