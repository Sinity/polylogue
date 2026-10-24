use super::*;

pub(crate) struct References<'a>(&'a Connection);

impl<'a> References<'a> {
    pub(super) fn capture(i: &'a Connection, pk: i64) -> Result<Self> {
        i.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS session_remove_text(id INTEGER PRIMARY KEY);
             CREATE TEMP TABLE IF NOT EXISTS session_remove_blob(hash BLOB PRIMARY KEY) WITHOUT ROWID;",
        )?;
        i.execute("INSERT INTO temp.session_remove_text SELECT DISTINCT text_id FROM block WHERE session_pk = ?1 AND text_id IS NOT NULL", [pk])?;
        i.execute("INSERT INTO temp.session_remove_blob SELECT DISTINCT blob_hash FROM block WHERE session_pk = ?1 AND blob_hash IS NOT NULL", [pk])?;
        Ok(Self(i))
    }

    pub(crate) fn visit_texts(&self, mut visit: impl FnMut(i64) -> Result<()>) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT id FROM temp.session_remove_text ORDER BY id")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            visit(row.get(0)?)?;
        }
        Ok(())
    }

    pub(crate) fn visit_blobs(&self, mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT hash FROM temp.session_remove_blob ORDER BY hash")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let hash: Vec<u8> = row.get(0)?;
            visit(&hash)?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<()> {
        self.0.execute_batch(
            "DELETE FROM temp.session_remove_text; DELETE FROM temp.session_remove_blob",
        )?;
        Ok(())
    }
}
