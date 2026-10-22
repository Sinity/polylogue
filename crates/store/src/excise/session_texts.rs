use crate::Result;
use rusqlite::Connection;

/// Exact session texts without retaining all decoded bodies in memory.
#[derive(Default)]
pub(super) struct SessionTexts(Option<Connection>);

impl SessionTexts {
    pub fn insert(&mut self, value: &str) -> Result<()> {
        if self.0.is_none() {
            let conn = Connection::open("")?;
            conn.execute_batch(
                "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE texts(value TEXT PRIMARY KEY) WITHOUT ROWID;
                CREATE TABLE found(value TEXT PRIMARY KEY) WITHOUT ROWID; BEGIN;",
            )?;
            self.0 = Some(conn);
        }
        self.0
            .as_ref()
            .expect("created session text set")
            .prepare_cached("INSERT OR IGNORE INTO texts(value) VALUES(?1)")?
            .execute([value])?;
        Ok(())
    }

    /// Preserve BINARY lexical order and byte-length semantics, including NULs.
    /// Match tracking uses another table, leaving this cursor's keys unchanged.
    pub fn visit(&self, minimum_bytes: usize, mut f: impl FnMut(&str) -> Result<()>) -> Result<()> {
        let Some(conn) = &self.0 else {
            return Ok(());
        };
        let mut query = conn.prepare_cached(
            "SELECT value FROM texts WHERE length(CAST(value AS BLOB))>=?1 ORDER BY value",
        )?;
        let mut rows = query.query([minimum_bytes as i64])?;
        while let Some(row) = rows.next()? {
            let value: String = row.get(0)?;
            f(&value)?;
        }
        Ok(())
    }

    pub fn reset_found(&self) -> Result<()> {
        if let Some(conn) = &self.0 {
            conn.execute("DELETE FROM found", [])?;
        }
        Ok(())
    }

    pub fn found(&self, value: &str) -> Result<()> {
        if let Some(conn) = &self.0 {
            conn.prepare_cached(
                "INSERT OR IGNORE INTO found SELECT value FROM texts WHERE value=?1",
            )?
            .execute([value])?;
        }
        Ok(())
    }

    pub fn missing(&self) -> Result<u64> {
        match &self.0 {
            Some(conn) => {
                Ok(conn
                    .query_row("SELECT count(*) FROM texts WHERE NOT EXISTS(SELECT 1 FROM found WHERE found.value=texts.value)", [], |r| {
                        r.get::<_, i64>(0)
                    })? as u64)
            }
            None => Ok(0),
        }
    }
}

#[cfg(test)]
mod tests;
