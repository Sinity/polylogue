use super::Reader;
use crate::{Result, source::Source};
use polylogue_model::signal::{Checkpoint, StoredOutbox};

impl Reader<'_> {
    pub fn signal_checkpoint(&self, owner: &str) -> Result<Option<Checkpoint>> {
        crate::signals::checkpoint(self.conn()?, Source::Cache, owner)
    }

    /// Historical records remain readable but are never eligible for effects.
    pub fn visit_signals(&self, mut visit: impl FnMut(StoredOutbox) -> Result<()>) -> Result<()> {
        let conn = self.conn()?;
        let mut query =
            conn.prepare("SELECT id FROM source_delivery WHERE signal IS NOT NULL ORDER BY at,id")?;
        let mut rows = query.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let outbox = crate::primary::authored::signal_outbox(conn, &id)?;
            visit(outbox)?;
        }
        Ok(())
    }
}
