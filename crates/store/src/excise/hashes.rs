//! The texts an excision removed, whose vectors go, and the blobs it may free.
use crate::Result;
use rusqlite::Connection;

pub(super) struct RemovalHashes<'a>(&'a Connection);

impl<'a> RemovalHashes<'a> {
    pub(super) fn new(index: &'a Connection) -> Result<Self> {
        index.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS excision_removed_text(hash BLOB PRIMARY KEY) WITHOUT ROWID;
             CREATE TEMP TABLE IF NOT EXISTS excision_blob_to_check(hash BLOB PRIMARY KEY) WITHOUT ROWID;",
        )?;
        Ok(Self(index))
    }

    pub(super) fn text(&self, hash: &[u8]) -> Result<()> {
        self.0
            .prepare_cached("INSERT OR IGNORE INTO temp.excision_removed_text VALUES (?1)")?
            .execute([hash])?;
        Ok(())
    }

    pub(super) fn blob_to_check(&self, hash: &[u8]) -> Result<()> {
        self.0
            .prepare_cached("INSERT OR IGNORE INTO temp.excision_blob_to_check VALUES (?1)")?
            .execute([hash])?;
        Ok(())
    }

    pub(super) fn visit_texts(&self, mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT hash FROM temp.excision_removed_text ORDER BY hash")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            visit(&hash)?;
        }
        Ok(())
    }

    pub(super) fn visit_blobs_to_check(
        &self,
        mut visit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT hash FROM temp.excision_blob_to_check ORDER BY hash")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            visit(&hash)?;
        }
        Ok(())
    }

    pub(super) fn text_count(&self) -> Result<u64> {
        Ok(self
            .0
            .query_row("SELECT count(*) FROM temp.excision_removed_text", [], |r| {
                r.get::<_, i64>(0)
            })? as u64)
    }

    pub(super) fn finish(self) -> Result<()> {
        // Keep definitions while other TEMP cursors may be active.
        self.0.execute_batch(
            "DELETE FROM temp.excision_removed_text; DELETE FROM temp.excision_blob_to_check",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
