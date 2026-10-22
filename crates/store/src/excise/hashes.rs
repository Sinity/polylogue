//! Removed texts authorize vector exclusion; candidate blobs still need ownership checks.
use crate::{Result, StoreError};
use rusqlite::Connection;
use serde::Serializer as _;
use serde::ser::SerializeSeq;

pub(super) struct RemovalHashes<'a>(&'a Connection);

impl<'a> RemovalHashes<'a> {
    pub(super) fn new(index: &'a Connection) -> Result<Self> {
        index.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS excision_removed_text(hash BLOB PRIMARY KEY) WITHOUT ROWID;
             CREATE TEMP TABLE IF NOT EXISTS excision_candidate_blob(hash BLOB PRIMARY KEY) WITHOUT ROWID;",
        )?;
        Ok(Self(index))
    }

    pub(super) fn text(&self, hash: &[u8]) -> Result<()> {
        self.0
            .prepare_cached("INSERT OR IGNORE INTO temp.excision_removed_text VALUES (?1)")?
            .execute([hash])?;
        Ok(())
    }

    pub(super) fn blob_candidate(&self, hash: &[u8]) -> Result<()> {
        self.0
            .prepare_cached("INSERT OR IGNORE INTO temp.excision_candidate_blob VALUES (?1)")?
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

    pub(super) fn visit_blob_candidates(
        &self,
        mut visit: impl FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT hash FROM temp.excision_candidate_blob ORDER BY hash")?;
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

    pub(super) fn text_policy_json(&self) -> Result<String> {
        let mut bytes = Vec::new();
        let mut serializer = serde_json::Serializer::new(&mut bytes);
        let mut sequence = serializer.serialize_seq(None)?;
        self.visit_texts(|hash| {
            sequence.serialize_element(&crate::sinex_rows::hex(hash))?;
            Ok(())
        })?;
        sequence.end()?;
        String::from_utf8(bytes).map_err(|error| StoreError::Invalid(error.to_string()))
    }

    pub(super) fn finish(self) -> Result<()> {
        // Keep definitions while other TEMP cursors may be active.
        self.0.execute_batch(
            "DELETE FROM temp.excision_removed_text; DELETE FROM temp.excision_candidate_blob",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
