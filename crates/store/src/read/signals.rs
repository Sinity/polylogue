use super::Reader;
use crate::{Result, source::Source};
use polylogue_model::signal::{Checkpoint, StoredPendingDelivery};

impl Reader<'_> {
    pub fn signal_checkpoint(&self, evaluator: &str) -> Result<Option<Checkpoint>> {
        crate::signals::checkpoint(self.conn()?, Source::Cache, evaluator)
    }

    /// Older records stay readable but never trigger effects.
    pub fn visit_signals(
        &self,
        mut visit: impl FnMut(StoredPendingDelivery) -> Result<()>,
    ) -> Result<()> {
        let conn = self.conn()?;
        let mut query =
            conn.prepare("SELECT id FROM source_delivery WHERE signal IS NOT NULL ORDER BY at,id")?;
        let mut rows = query.query([])?;
        while let Some(row) = rows.next()? {
            let id: String = row.get(0)?;
            let pending = crate::primary::authored::pending_delivery(conn, &id)?;
            visit(pending)?;
        }
        Ok(())
    }
}
