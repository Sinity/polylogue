use super::super::record_ranges::Error;
use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};

mod cursor;
mod literal;

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Self::Store(error.into())
    }
}

/// Private computation for one shared-input rewrite. All pieces are checked
/// before any replacement is installed; the archive owns durable recovery.
pub(super) struct Erasures {
    conn: Connection,
}

impl Erasures {
    pub fn new() -> Result<Self> {
        let conn = Connection::open("")?;
        conn.execute_batch(
            "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
             PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
             CREATE TABLE requested(hash BLOB, start INTEGER, end INTEGER,
                 PRIMARY KEY(hash,start,end)) WITHOUT ROWID;
             CREATE TABLE normalized(hash BLOB,start INTEGER,end INTEGER,PRIMARY KEY(hash,start)) WITHOUT ROWID;
             CREATE TABLE checked(hash BLOB PRIMARY KEY, documents INTEGER,
                 erased INTEGER, changed INTEGER) WITHOUT ROWID;
             CREATE TABLE ranges(hash BLOB, start INTEGER, end INTEGER,
                 PRIMARY KEY(hash,start,end)) WITHOUT ROWID;
             BEGIN;",
        )?;
        Ok(Self { conn })
    }

    pub fn request(&self, hash: &[u8], start: usize, end: usize) -> Result<()> {
        self.conn
            .prepare_cached("INSERT OR IGNORE INTO requested VALUES(?1,?2,?3)")?
            .execute(params![hash, start as i64, end as i64])?;
        Ok(())
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

    pub fn has_unchecked(&self) -> Result<bool> {
        Ok(self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM requested r WHERE NOT EXISTS
                (SELECT 1 FROM checked c WHERE c.hash=r.hash))",
            [],
            |r| r.get(0),
        )?)
    }

    pub fn plan_blob(
        &self,
        hash: &[u8],
        file: &std::fs::File,
    ) -> std::result::Result<(usize, usize), Error> {
        let mut cursor = cursor::Cursor::new(self, hash, file)?;
        let result = super::super::record_ranges::plan_file(file, &mut cursor);
        if let Some(error) = cursor.error.take() {
            return Err(error);
        }
        let counts = result?;
        cursor.finish()?;
        crate::fault::check("excision_ranges.complete", "").map_err(Error::Store)?;
        self.conn.execute(
            "INSERT INTO checked SELECT ?1,?2,?3,EXISTS(SELECT 1 FROM ranges WHERE hash=?1)",
            params![hash, counts.0 as i64, counts.1 as i64],
        )?;
        Ok(counts)
    }

    /// Only complete plans may supply replacement ranges.
    pub fn visit_ranges(
        &self,
        hash: &[u8],
        mut f: impl FnMut(usize, usize) -> Result<()>,
    ) -> Result<()> {
        if self.counts(hash)?.is_none() {
            return Err(crate::StoreError::Invalid(
                "source blob erasure plan is incomplete".into(),
            ));
        }
        let mut stmt = self.conn.prepare_cached("SELECT r.start,r.end FROM ranges r JOIN checked c ON c.hash=r.hash WHERE r.hash=?1 ORDER BY r.start,r.end")?;
        let mut rows = stmt.query([hash])?;
        while let Some(row) = rows.next()? {
            f(
                row.get::<_, i64>(0)? as usize,
                row.get::<_, i64>(1)? as usize,
            )?;
        }
        Ok(())
    }

    pub fn visit(&self, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT hash FROM checked WHERE changed ORDER BY hash")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            f(row
                .get_ref(0)?
                .as_blob()
                .map_err(|e| crate::StoreError::Invalid(e.to_string()))?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
